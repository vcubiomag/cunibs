"""Shared test fixtures: synthetic cubes, coils, and the cropped real head-mesh patch.

The ``Subject`` fixtures are session-scoped so the GPU context and UQ precompute are built
once for the whole run. Never call ``.free()`` on one, and never mutate its ``HeadMesh``
(``skin_triangle_normals`` and ``tet_barycenters_mm`` are ``cached_property``) — take the
``fresh_subject`` factory instead.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from cunibs.mesh import HeadMesh

DATA_DIR = Path(__file__).parent / "data"
PATCH_GZ = DATA_DIR / "head_patch_r25.msh.gz"
PATCH_MANIFEST = DATA_DIR / "head_patch_r25.json"

_CUBE_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.float64,
)
_CUBE_TETS = np.array(
    [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ],
    dtype=np.int32,
)
_CUBE_TRIS = np.array(
    [
        [0, 2, 3],
        [0, 3, 1],  # z=0
        [4, 5, 7],
        [4, 7, 6],  # z=1
        [0, 1, 5],
        [0, 5, 4],  # y=0
        [2, 6, 7],
        [2, 7, 3],  # y=1
        [0, 4, 6],
        [0, 6, 2],  # x=0
        [1, 3, 7],
        [1, 7, 5],  # x=1
    ],
    dtype=np.int32,
)

_has_gpu: bool | None = None


def has_gpu() -> bool:
    global _has_gpu
    if _has_gpu is None:
        try:
            import cupy

            _has_gpu = cupy.cuda.runtime.getDeviceCount() > 0
        # cupy raises its own driver/runtime errors when no device or driver is present.
        except Exception:  # noqa: BLE001
            _has_gpu = False
    return _has_gpu


def pytest_collection_modifyitems(items):
    no_gpu = pytest.mark.skip(reason="no CUDA GPU available")
    no_patch = pytest.mark.skip(reason=f"missing {PATCH_GZ}")
    no_reference = pytest.mark.skip(reason="CUNIBS_REFERENCE_MESH is not set")
    gpu_ok = has_gpu()
    patch_ok = PATCH_GZ.exists()
    reference_ok = bool(os.environ.get("CUNIBS_REFERENCE_MESH"))
    for item in items:
        if "gpu" in item.keywords and not gpu_ok:
            item.add_marker(no_gpu)
        if "realmesh" in item.keywords and not patch_ok:
            item.add_marker(no_patch)
        if "reference" in item.keywords and not reference_ok:
            item.add_marker(no_reference)


@pytest.fixture(scope="session")
def cp():
    import cupy

    return cupy


@pytest.fixture(scope="session", autouse=True)
def _free_gpu_pool():
    yield
    if has_gpu():
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()


@pytest.fixture(scope="session")
def cube_mesh() -> HeadMesh:
    """A solvable 100 mm all-gray-matter cube."""
    return HeadMesh(
        nodes_mm=np.ascontiguousarray(_CUBE_CORNERS * 100.0),
        tet_nodes=_CUBE_TETS.copy(),
        tet_tags=np.full(_CUBE_TETS.shape[0], 2, dtype=np.int32),
        skin_tris=_CUBE_TRIS.copy(),
    )


@pytest.fixture(scope="session")
def two_tissue_cube_mesh() -> HeadMesh:
    """A 100 mm cube split into gray matter (2) and CSF (3) so |E| depends on the σ ratio."""
    return HeadMesh(
        nodes_mm=np.ascontiguousarray(_CUBE_CORNERS * 100.0),
        tet_nodes=_CUBE_TETS.copy(),
        tet_tags=np.array([2, 2, 2, 3, 3, 3], dtype=np.int32),
        skin_tris=_CUBE_TRIS.copy(),
    )


@pytest.fixture(scope="session")
def synthetic_coil():
    """A two-dipole coil in its local frame."""
    from cunibs.coil import Coil

    return Coil(
        positions_m=np.array([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64),
        moments=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float64),
        name="synthetic",
        didt_max=100e6,
    )


@pytest.fixture(scope="session")
def figure8_coil():
    """A wider dipole pair, so the field over the cube is non-trivial."""
    from cunibs.coil import Coil

    return Coil(
        positions_m=np.array([[-0.03, 0.0, 0.0], [0.03, 0.0, 0.0]], dtype=np.float64),
        moments=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float64),
        name="test",
        didt_max=150e6,
    )


@pytest.fixture(scope="session")
def d70_coil():
    from cunibs.coil import MAGSTIM_D70, Coil

    return Coil.load(MAGSTIM_D70)


@pytest.fixture(scope="session")
def patch_manifest() -> dict:
    return json.loads(PATCH_MANIFEST.read_text())


@pytest.fixture(scope="session")
def patch_mesh_path(tmp_path_factory) -> Path:
    """Decompress the committed patch once per session (the parser needs a real path)."""
    path = tmp_path_factory.mktemp("mesh") / "head_patch_r25.msh"
    path.write_bytes(gzip.decompress(PATCH_GZ.read_bytes()))
    return path


@pytest.fixture(scope="session")
def patch_mesh(patch_mesh_path) -> HeadMesh:
    from cunibs.mesh import load_mesh

    return load_mesh(patch_mesh_path)


@pytest.fixture(scope="session")
def patch_top_mm(patch_mesh) -> np.ndarray:
    """The topmost scalp node of the patch — the canonical coil target."""
    skin_nodes = np.unique(patch_mesh.skin_tris)
    coords = patch_mesh.nodes_mm[skin_nodes]
    return np.ascontiguousarray(coords[np.argmax(coords[:, 2])])


@pytest.fixture(scope="session")
def patch_placement(patch_top_mm):
    from cunibs import Placement

    return Placement(patch_top_mm, patch_top_mm + np.array([0.0, 50.0, 0.0]), 4.0)


@pytest.fixture(scope="session")
def cube_subject(cube_mesh):
    from cunibs import Subject

    return Subject(cube_mesh)


@pytest.fixture(scope="session")
def two_tissue_subject(two_tissue_cube_mesh):
    from cunibs import Subject

    return Subject(two_tissue_cube_mesh)


@pytest.fixture(scope="session")
def patch_subject(patch_mesh):
    from cunibs import Subject

    return Subject(patch_mesh)


@pytest.fixture
def fresh_subject():
    """Factory for throwaway subjects, for tests that free or mutate solver state."""
    from cunibs import Subject

    made = []

    def make(mesh) -> Subject:
        subj = Subject(mesh)
        made.append(subj)
        return subj

    yield make
    for subj in made:
        subj.free()


@pytest.fixture(scope="session")
def reference_mesh_path() -> Path:
    path = os.environ.get("CUNIBS_REFERENCE_MESH")
    if not path:
        pytest.skip("CUNIBS_REFERENCE_MESH is not set")
    return Path(path)
