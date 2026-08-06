# cuNIBS

[![pypi](https://img.shields.io/pypi/v/cunibs)](https://pypi.python.org/pypi/cunibs)

cuNIBS computes the electric field induced by transcranial magnetic stimulation
(TMS) in a tetrahedral head model. It solves the magneto-quasistatic
finite-element problem on the GPU with first-order elements, magnetic dipole coil
models, and a mixed-precision conjugate-gradient solve preconditioned by an
aggregation-AMG V-cycle. The mesh and the AMG hierarchy stay on the GPU and are
reused across coil placements, so a sweep pays the setup cost once.

It supports isotropic conductivity models, conductivity uncertainty
quantification, coil-placement optimization, and NVIDIA GPUs.

## Installation

```bash
python -m pip install cunibs
```

Requires Python 3.12 or later and an NVIDIA GPU. Wheels are published for x86-64
Linux and Windows, for CPython 3.12 through 3.14 including free-threaded 3.14.
The install pulls the CUDA 13 toolkit wheels and cupy, so no system CUDA
installation is needed, but the driver must support CUDA 13 (r580 or later).

## Quickstart

```python
from cunibs import Placement, Subject
from cunibs.coil import Coil, MAGSTIM_D70

subject = Subject.from_mesh("subject.msh")
coil = Coil.load(MAGSTIM_D70)

placement = Placement(
    center_mm=[0.0, 20.0, 80.0],
    handle_mm=[0.0, 70.0, 80.0],
    distance_mm=4.0,
)

result = subject.simulate(coil, placement, didt=1.0e6)

print(result.peak_magnE())
print(result.peak_location_mm())
print(result.focality(frac=0.5))
```

`center_mm` is the scalp target and `handle_mm` is a point in the positive
handle direction. cuNIBS projects the target onto the scalp, builds the coil
frame from the local surface normal, and offsets by `distance_mm` along that
normal. A handle lying on the normal itself leaves the rotation about the normal
undefined and is rejected.

`didt` is the stimulator's current rate of change in A/s. The field is linear in
it. Each bundled coil carries its rated peak as `coil.didt_max`.

## Input data

### Head mesh

`Subject.from_mesh` reads binary Gmsh 2.2 files with first-order tetrahedra and
an oriented scalp surface (tag `1005`). Coordinates are read as millimetres.
Volume tags select the built-in conductivities; unrecognized volume tags and
surface triangles are dropped on load.

Generate individualized models with the SimNIBS
[CHARM](https://simnibs.github.io/simnibs/build/html/documentation/command_line/charm.html)
pipeline:

```bash
charm subject_id T1w.nii.gz T2w.nii.gz
```

CHARM writes `m2m_subject_id/subject_id.msh`. A T1-weighted scan is enough, but
a T2-weighted scan improves skull segmentation. Inspect the segmentation before
simulating.

Conductivities follow the standard SimNIBS values:

| Tag | Tissue | Conductivity (S/m) | Source |
| ---: | --- | ---: | --- |
| 1 | White matter | 0.126 | Wagner et al. (2004) |
| 2 | Gray matter | 0.275 | Wagner et al. (2004) |
| 3 | Cerebrospinal fluid | 1.654 | Wagner et al. (2004) |
| 5 | Scalp | 0.465 | Wagner et al. (2004) |
| 6 | Eye | 0.500 | Opitz et al. (2015) |
| 7 | Cortical bone | 0.008 | Opitz et al. (2015) |
| 8 | Cancellous bone | 0.025 | Opitz et al. (2015) |
| 9 | Blood | 0.600 | Gabriel et al. (2009) |
| 10 | Muscle | 0.160 | Gabriel et al. (2009) |

### Coil model

The package bundles the 25 validated coil models of Drakaki et al. (2022),
available as constants in `cunibs.coil` and loaded with `Coil.load`. Dipole
positions are in metres and moments in A m².

Convert a SimNIBS CCD coil to the HDF5 dipole format:

```python
from pathlib import Path

from cunibs.coil import Coil, encode_ccd

encode_ccd(Path("coil.ccd"), Path("coil.h5"))
coil = Coil.load("coil.h5")
```

## Sweeping placements

`simulate` takes one placement. `iter_simulate` streams many, reusing the
assembled system and the AMG hierarchy:

```python
placements = [
    Placement([0.0, 20.0, 80.0], [0.0, 70.0, 80.0]),
    Placement([20.0, 0.0, 80.0], [70.0, 0.0, 80.0]),
]

for result in subject.iter_simulate(coil, placements, didt=1.0e6):
    print(result.peak_magnE())
```

It is a generator: nothing is computed until you iterate, results come back in
the order given, and each is freed as the loop advances. Peak memory is bounded
by one block rather than by the sweep length. Wrap it in `list()` to collect
everything, which is always safe for summaries.

Placements are solved in blocks that share one stiffness and hierarchy read per
block. `block_k` sets the block width (default 8, `1` solves one at a time) and
also caps peak memory when fields are retained. It is a throughput and memory
knob only; the same placement returns the same field at any width, bit for bit.

```python
for result in subject.iter_simulate(coil, placements, didt=1.0e6, block_k=4):
    ...
```

### Many subjects

A `Subject` holds its solver context and AMG hierarchy on the GPU for its
lifetime. Use the context manager (or `subject.free()`) to release that memory
between subjects:

```python
from pathlib import Path

for mesh_file in Path("subjects").glob("m2m_*/*.msh"):
    with Subject.from_mesh(mesh_file) as subject:
        result = subject.simulate(coil, placement, didt=1.0e6)
        ...  # collect results
```

## Results

Every `FieldResult` carries `summary`, the gray-matter metrics:

```python
result.peak_magnE()
result.focality(0.5)
result.summary["distribution"]["p99"]
```

The metric API reports the peak field, peak location, stimulated volume,
field-weighted centre of gravity, and volume-weighted distribution statistics.
`peak_magnE()` is the true maximum of `|E|`. Focality is measured against the
volume-weighted 99.9th percentile instead, since on a tetrahedral mesh the true
maximum is routinely set by a single sliver element at a tissue boundary:
`focality(0.5)` is the volume with `|E|` at or above half that percentile.

Full-volume arrays are opt-in, because they are what makes a result large. On a
4M-tetrahedron model the three together are about 70 MB (`E` 69%, `magnE` 23%,
`v` 8%); with none of them a result is about a kilobyte. Ask for what you need:

```python
result = subject.simulate(coil, placement, didt=1.0e6, magnitude=True)

for result in subject.iter_simulate(
    coil, placements, magnitude=True, vectors=True, potential=True
):
    ...
```

`result.magnE`, `result.E` and `result.v` are `None` when not requested.
Retaining `magnitude` also unlocks the two metrics a precomputed summary cannot
answer, `summary_for(region)` for a non-default tissue and `focality(frac)` at an
arbitrary fraction:

```python
gray_matter = result.summary_for("gray_matter")
whole_model = result.summary_for("all")
```

Results are always NumPy. For device-resident fields, call
`cunibs.fem.solve_placements_block` directly.

`FieldResult` contains:

| Attribute | Description | Units |
| --- | --- | --- |
| `E` | Electric field per tetrahedron | V/m |
| `magnE` | Electric-field magnitude per tetrahedron | V/m |
| `v` | Electric scalar potential per node | V |
| `transform` | Coil-to-head affine matrix | translation in mm |
| `vols` | Tetrahedron volumes | m³ |
| `tet_tags` | Volume tissue tags | dimensionless |
| `barycenters_mm` | Tetrahedron barycentres | mm |
| `didt` | Coil current rate of change | A/s |
| `recovery` | Which post-processing produced `E`/`magnE`/`summary` | dimensionless |
| `E_slots` | Recovered field per slot, with `nodal=True` | V/m |
| `slot_node` | Mesh node each slot belongs to, with `nodal=True` | dimensionless |
| `slot_tag` | Tissue tag each slot belongs to, or `None` when slots are nodes | dimensionless |
| `n_nodes` | Mesh node count, so `nodal_field` can size its output | dimensionless |

Save a result and its metric inputs to HDF5. Fields that were not retained are
absent from the file and load back as `None`:

```python
result.save("placement.h5")

from cunibs import FieldResult

loaded = FieldResult.load("placement.h5")
```

## Field recovery

The raw finite-element field is constant over each tetrahedron. `recovery=`
post-processes it by fitting a local patch around each node and interpolating
from there, the same step SimNIBS applies before mapping a field to a surface or
volume.

```python
result = subject.simulate(coil, placement, didt=1.0e6, magnitude=True)

result.magnE     # |E| of the recovered field
result.recovery  # "harmonic"

raw = subject.simulate(coil, placement, didt=1.0e6, magnitude=True, recovery="raw")
```

`E`, `magnE` and `summary` all describe whichever field the mode selected. The
raw field is not additionally retained.

| `recovery` | What it does |
| --- | --- |
| `"harmonic"` | Recovers the potential in the space of harmonic polynomials and differentiates it. **The default**, and the most accurate mode at a tissue interface. |
| `"raw"` | The per-tetrahedron field straight from the solve. Its peak is the largest element average rather than a value at a point, so it reads high and moves with mesh resolution. |
| `"spr_tissue"` | Zienkiewicz-Zhu superconvergent patch recovery (Zienkiewicz and Zhu, 1992) with each patch restricted to one tissue. **What SimNIBS's surface and volume overlays report**, so use it when comparing against them. Its boundary rule is SimNIBS's too: wherever the cropped tissue ends, including at every tissue interface, each side takes its own one-sided volume-weighted average rather than a fit. |
| `"spr_global"` | The same fit over patches spanning every incident tetrahedron regardless of tissue. Corresponds to SimNIBS's `continuous=True`. |

`E` is discontinuous at a tissue boundary, since `J_n = σE_n` is
continuous and the normal component of `E` jumps by the conductivity ratio. A
single value per node cannot represent a two-valued field, so `"spr_global"` does
not converge there however fine the mesh. `"spr_tissue"` averages instead of
fitting there and converges at first order; `"harmonic"` fits and converges at
roughly 1.7. On a reference head mesh 62% of nodes have a mixed-tissue patch and
90% of gray-matter tetrahedra touch one, so 81% of `"spr_tissue"`'s slots are
volume averages.

Inside a region of constant conductivity the potential is harmonic, so the fit
runs in the harmonic polynomials rather than all of them: 16 of the 20 cubics,
dropping to 9 of the 10 quadratics on a patch too small to carry the cubic. That
constraint is also what makes the fit well posed at an interface, where the
same-tissue patch is a half-ball and cannot see the curvature normal to the flat
side. The assumption is specific to this formulation, with the source outside the
head; it would not carry to a problem with interior current sources.

Recovery costs one extra pass over the mesh per placement. Its patch weights are
built once per subject on first use and reused across placements.

### Nodal fields

A recovered field also exists on the mesh nodes, which interpolating onto
a cortical surface needs:

```python
result = subject.simulate(coil, placement, magnitude=True, nodal=True)

result.nodal_field()       # (n_nodes, 3) gray matter, NaN where it does not reach
result.nodal_field("csf")  # the CSF-side value at the same nodes
```

For the tissue-restricted modes the field lives on (node, tissue) slots rather
than nodes, since it is two-valued at a boundary. `nodal_field` returns one row
per mesh node carrying the requested tissue's value, and NaN where that tissue
does not reach. `region="all"` is rejected for those modes rather than silently
picking one of the two values. `E_slots`, `slot_node` and `slot_tag` expose the
underlying slot arrays.

## Conductivity uncertainty quantification

`simulate_conductivity_uq` runs a Monte Carlo analysis over tissue
conductivities for one placement or a sequence, configured by a
`ConductivityUQConfig`. Each tissue is an independent random variable with the
nominal conductivity as its mean and a coefficient of variation you supply;
draws are lognormal by default, which keeps conductivities positive. The result
reports per-tetrahedron mean, standard deviation, and coefficient of variation of
`|E|`.

```python
from cunibs import ConductivityUQConfig, Placement, Subject
from cunibs.coil import Coil, MAGSTIM_D70

subject = Subject.from_mesh("subject.msh")
coil = Coil.load(MAGSTIM_D70)

config = ConductivityUQConfig(
    n_samples=500,
    tissue_cov={2: 0.15, 3: 0.05, 7: 0.35, 8: 0.35},
    seed=1,
)

uq_result = subject.simulate_conductivity_uq(coil, placement, config, didt=1.0e6)

print(uq_result.peak_mean_magnE())
print(uq_result.max_local_cov())
```

The result carries its `summary` the same way a `FieldResult` does. Pass
`moments=True` to also retain the per-tetrahedron moment arrays; they are kept or
dropped together, since the metrics need both the mean and the CoV and the third
follows from those two. `mean_magnE`, `std_magnE` and `cov_magnE` then use the
same tetrahedron ordering as `FieldResult.magnE`.
`iter_simulate_conductivity_uq` streams a sequence of placements the same way
`iter_simulate` does.

`peak_mean_magnE` and `max_local_cov` are metrics *of* the moment fields, not
moments of a metric. For a nonlinear metric such as the peak or focality, the
metric of the mean is not the mean of the metric over the ensemble. Use the
per-draw arrays to characterize how a scalar is distributed across draws:

```python
m1 = subject.roi([-45.0, -5.0, 25.0], radius_mm=5.0, region="gray_matter")

uq_result = subject.simulate_conductivity_uq(
    coil, placement, config, didt=1.0e6, record_rois={"M1": m1}
)

uq_result.roi_samples["M1"]       # (n_samples,) per-draw ROI mean |E| (V/m)
uq_result.peak_samples            # (n_samples,) per-draw gray-matter peak |E|
uq_result.focality_samples        # (n_samples,) per-draw stimulated volume (m³)
uq_result.peak_location_samples   # (n_samples, 3) per-draw peak location (mm)
uq_result.tissue_sensitivity("peak")  # variance share per tissue tag
```

`tissue_sensitivity` regresses the log of a per-draw scalar (`"peak"`,
`"focality"`, or an ROI name) on the log conductivity draws. It is a first-order
linear-in-log index on the i.i.d. ensemble, not a Saltelli Sobol estimate.

Results save to HDF5 with `uq_result.save(path)` and load with
`ConductivityUQResult.load(path)`.

## Coil-placement optimization

`cunibs.adm` implements the Auxiliary Dipole Method (Gomez et al., 2021). A few
one-time adjoint solves, reusing the forward AMG hierarchy, sample a reciprocity
field on a regular grid. The target E-field of any placement is then a trilinear
interpolation plus a dipole sum, with no further FEM solve. This evaluates
candidates orders of magnitude faster than a forward solve each, and matches a
forward solve at the optimum to a relative error of 4e-4.

```python
import numpy as np

from cunibs import Subject, Target, adm
from cunibs.coil import Coil, MAGSTIM_D70

subject = Subject.from_mesh("subject.msh")
coil = Coil.load(MAGSTIM_D70)

# Omit `direction` to maximize |E| (three adjoint solves), or pass one to
# maximize a directional component.
target = Target(position_mm=[-45.0, -5.0, 25.0], region="gray_matter")

# Candidate scalp positions to search (each is projected onto the scalp).
centers = np.array([[x, y, 80.0] for x in range(-30, 31, 5) for y in range(-30, 31, 5)])

result = adm.optimize(subject.context, coil, target, centers)

result.best_objective  # peak |E| at the target (V/m)
result.best_center_mm  # optimal scalp position
result.best_angle_rad  # optimal in-plane rotation
```

The in-plane rotation is optimized in closed form: the target E-field is a rigid
rotation of the coil, so each component is band-limited in the angle. It is
sampled at `n_samples` angles, trigonometrically interpolated, and `|E(θ)|²` is
maximized analytically.

For repeated queries against a fixed target, build the reciprocity field once and
reuse it:

```python
recip = adm.build_reciprocity(subject.context, coil, target, centers)
E = adm.evaluate(recip, coil, placements, didt=1.0e6)  # (P, D) target E-vectors
```

## Reproducibility

A placement's field is a function of the mesh, the coil, the placement, `didt`,
the solve tolerance and the `recovery` mode, and of nothing else. It does not
depend on `block_k`, on which other placements shared its block, or on where it
fell in the sweep. Repeating a run reproduces it bitwise. Splitting a sweep
across calls, resuming an interrupted one, or retuning `block_k` for a different
GPU all leave the numbers unchanged. A conductivity-UQ draw depends on its own
conductivity vector and not on the ensemble size or the other draws, so the same
seed reproduces the same draws at any `n_samples`.

Results can still differ across GPU architectures, CUDA versions, compiler
versions, and dependency versions.

## Citation

No archival citation is provided yet. For reproducible academic use, cite the
software by name, author, version, and Git commit, and archive the exact input
mesh, coil model, and placement parameters used.

## References

- Saturnino, G. B., Puonti, O., Nielsen, J. D., Antonenko, D., Madsen, K. H.,
  and Thielscher, A. (2019). [SimNIBS 2.1: A comprehensive pipeline for
  individualized electric field modelling for transcranial brain
  stimulation](https://doi.org/10.1007/978-3-030-21293-3_1).
- Puonti, O., Van Leemput, K., Saturnino, G. B., Siebner, H. R., Madsen, K. H.,
  and Thielscher, A. (2020). [Accurate and robust whole-head segmentation from
  magnetic resonance images for individualized head
  modeling](https://doi.org/10.1016/j.neuroimage.2020.117044). *NeuroImage*,
  219, 117044.
- Wagner, T. A., Zahn, M., Grodzinsky, A. J., and Pascual-Leone, A. (2004).
  [Three-dimensional head model simulation of transcranial magnetic
  stimulation](https://doi.org/10.1109/TBME.2004.827925). *IEEE Transactions
  on Biomedical Engineering*, 51(9), 1586-1598.
- Gomez, L. J., Dannhauer, M., and Peterchev, A. V. (2021). [Fast computational
  optimization of TMS coil placement for individualized electric field
  targeting](https://doi.org/10.1016/j.neuroimage.2020.117696). *NeuroImage*,
  228, 117696. (Auxiliary Dipole Method.)
- Zienkiewicz, O. C., and Zhu, J. Z. (1992). [The superconvergent patch recovery
  and a posteriori error estimates. Part 1: The recovery
  technique](https://doi.org/10.1002/nme.1620330702). *International Journal for
  Numerical Methods in Engineering*, 33(7), 1331-1364.
- Opitz, A., Paulus, W., Will, S., Antunes, A., and Thielscher, A. (2015).
  [Determinants of the electric field during transcranial direct current
  stimulation](https://doi.org/10.1016/j.neuroimage.2015.01.033).
  *NeuroImage*, 109, 140-150.
- Gabriel, C., Peyman, A., and Grant, E. H. (2009).
  [Electrical conductivity of tissue at frequencies below 1
  MHz](https://doi.org/10.1088/0031-9155/54/16/002). *Physics in Medicine and
  Biology*, 54(16), 4863-4878.
- Drakaki, M., Mathiesen, C., Siebner, H. R., Madsen, K., and Thielscher, A.
  (2022). [Database of 25 validated coil models for electric field simulations
  for TMS](https://doi.org/10.1016/j.brs.2022.04.017). *Brain Stimulation*,
  15(3), 697-706.
- Naumov, M., Arsaev, M., Castonguay, P., et al. (2015). [AmgX: A library for
  GPU accelerated algebraic multigrid and preconditioned iterative
  methods](https://doi.org/10.1137/140980260). *SIAM Journal on Scientific
  Computing*, 37(5), S602-S626. The aggregation selector and l1-Jacobi V-cycle
  used here follow the algorithms described there.
