"""The SimNIBS side of every cross-code benchmark. Runs in SimNIBS's Python, needs no GPU.

    python simnibs_child.py run REQUEST.npz RESPONSE.npz
    python simnibs_child.py serve DIR [OP ...]  # answer the unanswered requests in DIR

SimNIBS gets the same mesh arrays, coil and 4x4 coil frames as cuNIBS. Mesh-sized work (the
conductivity field, assembly and solver setup) is done once per request and timed apart from
the per-placement work, and the coil file is read once, as a user running a sweep would.

Only stdlib, numpy and simnibs are imported here.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _mesh(a):
    from simnibs import mesh_io

    elements = mesh_io.Elements(triangles=a["tris"] + 1, tetrahedra=a["tets"] + 1)
    tags = np.concatenate([a["tri_tags"], a["tet_tags"]]).astype(np.int32)
    elements.tag1 = tags
    elements.tag2 = tags.copy()
    return mesh_io.Msh(
        nodes=mesh_io.Nodes(np.asarray(a["nodes"], np.float64)), elements=elements
    )


def _coil(name: str, didt: float):
    import simnibs
    from simnibs.simulation.tms_coil.tms_coil import TmsCoil

    path = Path(simnibs.__file__).parent / "resources/coil_models/Drakaki_BrainStim_2022"
    coil = TmsCoil.from_file(str(path / f"{name}.ccd"))
    for stimulator in coil.get_elements_grouped_by_stimulators():
        stimulator.di_dt = np.atleast_1d(didt)
    return coil


def _conductivity(msh, sigma_by_tag: dict[int, float] | None = None):
    from simnibs.simulation.sim_struct import TMSLIST

    tl = TMSLIST()
    tl.mesh = msh
    tl.check_conductivities()
    for tag, sigma in (sigma_by_tag or {}).items():
        tl.cond[int(tag) - 1].value = float(sigma)
    return tl.cond2elmdata(logger_level=0)


def _clock(timings: dict, key: str):
    class Clock:
        def __enter__(self):
            self.t0 = time.perf_counter()

        def __exit__(self, *exc):
            timings[key] = time.perf_counter() - self.t0

    return Clock()


def op_info(p, a):
    return {}, {}


def op_forward(p, a):
    """Solve every frame on one mesh. Returns E per tetrahedron and the phase timings."""
    from simnibs import mesh_io
    from simnibs.simulation import fem

    setup: dict[str, float] = {}
    # A user's run starts by reading the mesh from a .msh file, so that read is what is timed,
    # not building it from the request's arrays.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mesh.msh"
        mesh_io.write_msh(_mesh(a), str(path), mode="binary")
        with _clock(setup, "mesh_read_s"):
            msh = mesh_io.read_msh(str(path))
    with _clock(setup, "conductivity_s"):
        cond = _conductivity(msh)
    with _clock(setup, "coil_load_s"):
        coil = _coil(p["coil"], p["didt"])
    with _clock(setup, "assemble_s"):
        system = fem.TMSFEM(msh, cond, solver_options=p.get("solver", "hypre"))
    with _clock(setup, "prepare_solver_s"):
        system.prepare_solver()

    is_tet = msh.elm.elm_type == 4
    frames = a["frames"].reshape(-1, 4, 4)
    fields = p.get("fields", True)
    E = np.empty((len(frames), int(is_tet.sum()), 3), np.float32) if fields else None
    round_trip = np.empty_like(E) if p.get("node_round_trip") else None
    placements = []
    for i, frame in enumerate(frames):
        t: dict[str, float] = {}
        with _clock(t, "dadt_s"):
            dadt = coil.get_da_dt(msh, frame).node_data2elm_data()
        with _clock(t, "rhs_s"):
            b = system.assemble_rhs(dadt)
        with _clock(t, "solve_s"):
            v = system.solve(b)
        with _clock(t, "fields_s"):
            e = fem.calc_fields(mesh_io.NodeData(v, mesh=msh), "E", cond=cond, dadt=dadt)
        placements.append(t)
        if E is not None:
            E[i] = e.field["E"].value[is_tet]
        if round_trip is not None:
            round_trip[i] = e.field["E"].elm_data2node_data().node_data2elm_data().value[is_tet]

    arrays = {} if E is None else {"E": E}
    if round_trip is not None:
        arrays["E_round_trip"] = round_trip
    return {"setup_s": setup, "placements_s": placements}, arrays


def op_uq(p, a):
    """Time a conductivity draw the only way SimNIBS can: rebuild and re-solve per draw."""
    from simnibs import mesh_io
    from simnibs.simulation import fem

    msh = _mesh(a)
    coil = _coil(p["coil"], p["didt"])
    dadt = coil.get_da_dt(msh, a["frame"].reshape(4, 4)).node_data2elm_data()
    is_gm = (msh.elm.elm_type == 4) & (msh.elm.tag1 == 2)
    draws = []
    for sigmas in a["sigmas"]:
        t: dict[str, float] = {}
        with _clock(t, "conductivity_s"):
            cond = _conductivity(msh, dict(zip(p["tags"], sigmas, strict=True)))
        with _clock(t, "assemble_s"):
            system = fem.TMSFEM(msh, cond, solver_options=p.get("solver", "hypre"))
        with _clock(t, "prepare_solver_s"):
            system.prepare_solver()
        with _clock(t, "rhs_s"):
            b = system.assemble_rhs(dadt)
        with _clock(t, "solve_s"):
            v = system.solve(b)
        with _clock(t, "fields_s"):
            e = fem.calc_fields(mesh_io.NodeData(v, mesh=msh), "E", cond=cond, dadt=dadt)
        t["peak_gm"] = float(np.linalg.norm(e.field["E"].value[is_gm], axis=1).max())
        draws.append(t)
        # A PETSc solver left for the collector keeps its whole AMG hierarchy alive into the
        # next draw's setup.
        del system, e, v, b, cond
    return {"draws": draws}, {}


def op_dadt(p, a):
    """dA/dt at every node, per frame."""
    msh = _mesh(a)
    coil = _coil(p["coil"], p["didt"])
    out = np.stack([coil.get_da_dt(msh, f).value for f in a["frames"].reshape(-1, 4, 4)])
    return {}, {"dadt": out}


def op_node_recovery(p, a):
    """SimNIBS's element-to-node averaging of each element field in ``fields``."""
    from simnibs import mesh_io

    msh = _mesh({**a, "tris": np.zeros((0, 3), int), "tri_tags": np.zeros(0, int)})
    out = np.stack(
        [mesh_io.ElementData(f, mesh=msh).elm_data2node_data().value for f in a["fields"]]
    )
    return {}, {"nodal": out}


def op_sphere_field(p, a):
    """SimNIBS's closed-form E of dipoles outside a spherically symmetric conductor."""
    from simnibs.simulation import analytical_solutions

    E = analytical_solutions.tms_E_field(
        a["dipole_pos"], a["dipole_moment"], p["didt"], a["positions"]
    )
    return {}, {"E": E}


def op_adm(p, a):
    """SimNIBS's ADM placement search, run as ``TMSoptimize._ADM_optimize`` runs it.

    Returns every candidate as a 4x4 frame in mm, positions outer and rotations inner, with the
    objective in the same order. Also returns the target region as tetrahedron indices, every
    tetrahedron's barycentre, the coil's dipoles, the 17x17x2 auxiliary coil ADMmag evaluates in
    its place (one moment set per rotation), and the phase timings.
    """
    from simnibs import mesh_io
    from simnibs.optimization import ADMlib
    from simnibs.optimization.tms_optimization import define_target_region, get_opt_grid_ADM
    from simnibs.simulation import fem

    msh = _mesh(a)
    cond = _conductivity(msh)
    coil = _coil(p["coil"], p["didt"])
    dipoles = np.concatenate([e.get_points() for e in coil.elements]) * 1e-3
    moments = np.concatenate([e.get_values() for e in coil.elements])
    target = np.asarray(p["target_mm"], np.float64)

    t: dict[str, float] = {}
    with _clock(t, "grid_and_target_s"):
        region = define_target_region(msh, target, p["roi_radius_mm"], [2])
        matrices, rotations = get_opt_grid_ADM(
            msh,
            target,
            distance=p["distance_mm"],
            radius=p["search_radius_mm"],
            resolution_pos=p["resolution_pos_mm"],
            resolution_angle=p["resolution_angle_deg"],
        )
    position_frames = matrices.transpose(2, 0, 1).copy()
    matrices[:3, 3, :] *= 1e-3

    with _clock(t, "assemble_s"):
        system = fem.DipoleFEM(msh, cond, p["solver"])
        system.prepare_solver()
    vols = msh.elements_volumes_and_areas()
    bary = msh.elements_baricenters()

    def current(system, direction):
        jp = mesh_io.ElementData(np.zeros((msh.elm.nr, 3)))
        jp[region] = direction
        b = system.assemble_rhs(
            bary[region],
            np.atleast_2d(direction) * 1e-9 * np.atleast_1d(vols[region])[:, None],
            "partial integration",
        )
        if b.ndim == 2:
            b = b.sum(1)
        v = mesh_io.NodeData(system.solve(b), mesh=msh)
        j = fem.calc_fields(v, "J", cond=cond).field["J"][:] + jp[:]
        return j / np.sum(vols[region]) * vols[:, None]

    with _clock(t, "adjoint_s"):
        jx, jy, jz = (current(system, d) for d in ([1, 0, 0], [0, 1, 0], [0, 0, 1]))
    # SimNIBS frees the solver before ADMmag, which needs the memory.
    del system

    th = msh.elm.get_tetrahedra()
    with _clock(t, "evaluate_s"):
        e_roi = ADMlib.ADMmag(
            bary[th].T * 1e-3,
            jx[th].T,
            jy[th].T,
            jz[th].T,
            dipoles.T,
            moments.T,
            matrices,
            rotations,
        )
    # ADMmag returns (rotations, positions); SimNIBS transposes it this way before pairing it
    # with the frames built below.
    objective = e_roi.T.reshape(-1) * p["didt"]

    z = np.array([0.0, 0.0, 1.0])
    frames = []
    for cm in position_frames:
        for r in rotations.T:
            turn = np.eye(4)
            turn[:3, :3] = np.array([np.cross(r, z), r, z]).T
            frames.append(cm @ turn)

    # The same call ADMmag makes internally, so this is the auxiliary coil it evaluated.
    raux, kaux = ADMlib.resamplecoil(
        dipoles.T, moments.T, [17, 17, 2], rotations.shape[1], rotations
    )
    is_tet = msh.elm.elm_type == 4
    return (
        {
            "timings_s": t,
            "n_positions": len(position_frames),
            "n_rotations": int(rotations.shape[1]),
        },
        {
            "objective": objective,
            "frames": np.array(frames),
            "position_frames": position_frames,
            # Element numbers are 1-based and the request's triangles come first.
            "roi_tets": np.asarray(region, np.int64) - 1 - len(a["tris"]),
            "barycenters": bary.value[is_tet],
            "dipoles_m": dipoles,
            "moments": moments,
            "aux_positions_m": np.asarray(raux).T,
            "aux_moments": np.asarray(kaux).transpose(2, 1, 0),
        },
    )


def _onlinefem_coil(p):
    """The coil at unit dI/dt, since ``update_field`` scales by it.

    ``sampled`` precomputes dA/dt on the coil file's own grid and interpolates it per placement,
    the coil path Cao et al. use. Otherwise every dipole is evaluated at every point.
    """
    coil = _coil(p["coil"], 1.0)
    return coil.as_sampled() if p["sampled_coil"] else coil


def _onlinefem(msh, p, coil, cond=None):
    from simnibs.simulation.onlinefem import OnlineFEM

    threads = p["threads"] or len(os.sched_getaffinity(0))
    return OnlineFEM(
        msh,
        "TMS",
        roi=[],
        coil=coil,
        useElements=p["use_elements"],
        cpus=threads,
        cond=cond,
        solver_loglevel=logging.WARNING,
    )


def _onlinefem_placement(ofem, clouds: dict, frame, didt: float, t: dict) -> dict:
    """One placement as ``OnlineFEM.update_field`` runs it, with each cloud's field timed apart."""
    with _clock(t, "rhs_s"):
        b = ofem.set_rhs(matsimnibs=frame)
    with _clock(t, "solve_s"):
        v = ofem.solve(b)
    out = {}
    for name, cloud in clouds.items():
        with _clock(t, f"fields_{name}_s"):
            out[name] = cloud.calc_fields(v=v, dadt=ofem.dadt, dataType=0) * didt
    return out


def op_onlinefem(p, a):
    """Cao et al.'s pre-factorised SimNIBS path for repeated placements, phase by phase.

    Setup samples the coil if asked, factorises the stiffness matrix with PARDISO and precomputes
    the right-hand-side force integrals; each point cloud then builds its own sampling operator. A
    placement evaluates dA/dt, assembles the right-hand side, back-substitutes, and samples |E| at
    a cloud. "volume" is every tetrahedron's own field, "roi" the tetrahedra in ``roi_tets``. The
    first placement also pays numba's compilation, so callers take the steady cost from the rest.
    """
    from simnibs.simulation.onlinefem import FemTargetPointCloud

    msh = _mesh(a).crop_mesh(elm_type=4)
    setup: dict[str, float] = {}
    with _clock(setup, "conductivity_s"):
        cond = _conductivity(msh)
    with _clock(setup, "coil_s"):
        coil = _onlinefem_coil(p)
    with _clock(setup, "factorise_s"):
        ofem = _onlinefem(msh, p, coil, cond)
    bary = msh.elements_baricenters().value
    clouds = {}
    for name, tets in (("volume", np.arange(len(bary))), ("roi", a["roi_tets"])):
        with _clock(setup, f"cloud_{name}_s"):
            clouds[name] = FemTargetPointCloud(msh, center=bary[tets], nearest_neighbor=True)

    placements = []
    field = None
    for i, frame in enumerate(a["frames"].reshape(-1, 4, 4)):
        t: dict[str, float] = {}
        out = _onlinefem_placement(ofem, clouds, frame, p["didt"], t)
        placements.append(t)
        if i == 0 and p.get("field"):
            field = np.asarray(out["volume"], np.float32).reshape(-1)
    return {"setup_s": setup, "placements_s": placements}, (
        {} if field is None else {"magnE": field}
    )


def op_onlinefem_uq(p, a):
    """A conductivity draw under OnlineFEM: the factorisation is rebuilt, then one placement."""
    from simnibs.simulation.onlinefem import FemTargetPointCloud

    msh = _mesh(a).crop_mesh(elm_type=4)
    roi = FemTargetPointCloud(
        msh, center=msh.elements_baricenters().value[a["roi_tets"]], nearest_neighbor=True
    )
    # The coil does not depend on the conductivities, so it is sampled once for every draw.
    coil = _onlinefem_coil(p)
    draws = []
    for sigmas in a["sigmas"]:
        t: dict[str, float] = {}
        with _clock(t, "conductivity_s"):
            cond = _conductivity(msh, dict(zip(p["tags"], sigmas, strict=True)))
        with _clock(t, "factorise_s"):
            ofem = _onlinefem(msh, p, coil, cond)
        _onlinefem_placement(ofem, {"roi": roi}, a["frame"].reshape(4, 4), p["didt"], t)
        draws.append(t)
        del ofem
    return {"draws": draws}, {}


def op_throughput(p, a):
    """Solve one subject's request in ``processes`` SimNIBS processes at once, as a cohort would.

    The CPUs this process may use are split evenly between them. Returns the wall time for all
    of them together and each one's own phase timings.
    """
    n = p["processes"]
    threads = max(1, len(os.sched_getaffinity(0)) // n)
    header = {k: v for k, v in p.items() if k != "processes"} | {
        "op": "forward",
        "fields": False,
    }
    env = os.environ | dict.fromkeys(THREAD_VARS, str(threads))
    with tempfile.TemporaryDirectory() as tmp:
        request = Path(tmp) / "one.request.npz"
        np.savez(request, __request__=np.frombuffer(json.dumps(header).encode(), np.uint8), **a)
        responses = [Path(tmp) / f"{i}.response.npz" for i in range(n)]
        t0 = time.perf_counter()
        procs = [
            subprocess.Popen([sys.executable, __file__, "run", str(request), str(r)], env=env)
            for r in responses
        ]
        if any(q.wait() for q in procs):
            raise RuntimeError("a SimNIBS process failed")
        wall = time.perf_counter() - t0
        runs = []
        for r in responses:
            with np.load(r) as f:
                meta = json.loads(f["__response__"].tobytes())
            runs.append({"setup_s": meta["setup_s"], "placements_s": meta["placements_s"]})
    return {"processes": n, "threads_per_process": threads, "wall_s": wall, "runs": runs}, {}


def host() -> dict:
    import simnibs

    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "threads": int(os.environ.get("OMP_NUM_THREADS", "0")) or None,
        "simnibs": simnibs.__version__,
    }


def run(request: Path, response: Path) -> None:
    with np.load(request) as f:
        params = json.loads(f["__request__"].tobytes())
        arrays = {k: f[k] for k in f.files if k != "__request__"}

    # Thread pools are sized when numpy and PETSc load, so a request that names a thread count
    # is re-run in a process started with it.
    threads = params.get("threads")
    if threads == 0:
        threads = len(os.sched_getaffinity(0))
    if threads and os.environ.get("OMP_NUM_THREADS") != str(threads):
        env = os.environ | dict.fromkeys(THREAD_VARS, str(threads))
        subprocess.run(
            [sys.executable, __file__, "run", str(request), str(response)], env=env, check=True
        )
        return

    meta, out = globals()[f"op_{params.pop('op')}"](params, arrays)
    meta["host"] = host()
    header = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    partial = response.with_name(response.name + ".partial.npz")
    np.savez(partial, __response__=header, **out)
    partial.replace(response)


def serve(directory: Path, ops: list[str]) -> None:
    """Answer each unanswered request in ``directory`` whose operation is in ``ops`` (or any).

    A request is claimed before it is answered, so several servers can share one directory.
    """
    for request in sorted(directory.glob("*.request.npz")):
        response = request.with_name(request.name.replace(".request.", ".response."))
        with np.load(request) as f:
            op = json.loads(f["__request__"].tobytes())["op"]
        if response.exists() or (ops and op not in ops):
            continue
        try:
            request.with_suffix(".claim").touch(exist_ok=False)
        except FileExistsError:
            continue
        print(f"answering {request.name} ({op})", flush=True)
        subprocess.run(
            [sys.executable, __file__, "run", str(request), str(response)], check=True
        )


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["run", request, response]:
            run(Path(request), Path(response))
        case ["serve", directory, *ops]:
            serve(Path(directory), ops)
        case _:
            sys.exit(__doc__)
