# cuNIBS

[![pypi](https://img.shields.io/pypi/v/cunibs)](https://pypi.python.org/pypi/cunibs)

cuNIBS computes transcranial magnetic stimulation (TMS) electric fields in
tetrahedral head models. Use it for a single placement simulation, repeated
placement studies, conductivity uncertainty analysis, and coil-placement
optimization on NVIDIA GPUs. See the accompanying manuscript for the numerical
method and its validation.

## Installation

```bash
python -m pip install cunibs
```

cuNIBS requires Python 3.12 or later, an NVIDIA GPU, and a driver compatible
with CUDA 13 (r580 or later). Wheels are available for x86-64 Linux and Windows,
for CPython 3.12 through 3.14, including free-threaded Python 3.14. The package
installs the required CUDA and CuPy wheels; a separate CUDA toolkit installation
is not required.

## Quick start

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

`center_mm` is the intended scalp target and `handle_mm` specifies the positive
handle direction. cuNIBS projects the target onto the scalp and places the coil
`distance_mm` above the local surface. The handle point must not lie on the
surface normal through the target.

`didt` is the coil-current rate of change in A/s. Fields scale linearly with
this value. Bundled coil models provide their rated peak value as
`coil.didt_max`.

## Input data

### Head mesh

`Subject.from_mesh` reads binary Gmsh 2.2 meshes containing first-order
tetrahedra and an oriented scalp surface with tag `1005`. Coordinates must be in
millimetres. The volume tag selects the built-in tissue conductivity; unrelated
surface triangles are ignored. Invalid mesh references, non-finite coordinates,
and zero-volume tetrahedra are rejected.

Create individualized head models with the SimNIBS
[CHARM](https://simnibs.github.io/simnibs/build/html/documentation/command_line/charm.html)
pipeline:

```bash
charm subject_id T1w.nii.gz T2w.nii.gz
```

CHARM writes `m2m_subject_id/subject_id.msh`. A T1-weighted image is sufficient;
a T2-weighted image can improve skull segmentation. Inspect the segmentation
before simulation.

The built-in conductivities match the standard SimNIBS values.

| Tag | Tissue | Conductivity (S/m) | Source |
| ---: | --- | ---: | --- |
| 1 | White matter | 0.126 | Wagner et al. (2004) |
| 2 | Gray matter | 0.275 | Wagner et al. (2004) |
| 3 | Cerebrospinal fluid | 1.654 | Wagner et al. (2004) |
| 4 | Average bone | 0.010 | Wagner et al. (2004) |
| 5 | Scalp | 0.465 | Wagner et al. (2004) |
| 6 | Eye | 0.500 | Opitz et al. (2015) |
| 7 | Cortical bone | 0.008 | Opitz et al. (2015) |
| 8 | Cancellous bone | 0.025 | Opitz et al. (2015) |
| 9 | Blood | 0.600 | Gabriel et al. (2009) |
| 10 | Muscle | 0.160 | Gabriel et al. (2009) |

### Coil models

The package includes the 25 validated coil models from Drakaki et al. (2022).
Import a bundled-model constant from `cunibs.coil` and pass it to `Coil.load`.
Custom coil models use an HDF5 dipole format, with positions in metres and
moments in A m². Convert a SimNIBS CCD coil as follows:

```python
from pathlib import Path

from cunibs.coil import Coil, encode_ccd

encode_ccd(Path("coil.ccd"), Path("coil.h5"))
coil = Coil.load("coil.h5")
```

## Simulating placements

Use `simulate` for one placement. Use `iter_simulate` for a sequence; it yields
results in input order and reuses the subject setup.

```python
placements = [
    Placement([0.0, 20.0, 80.0], [0.0, 70.0, 80.0]),
    Placement([20.0, 0.0, 80.0], [70.0, 0.0, 80.0]),
]

for result in subject.iter_simulate(coil, placements, didt=1.0e6):
    print(result.peak_magnE())
```

The iterator computes results only as they are requested. It releases each
result as the loop advances unless you retain it. Use `list(...)` when all
results are needed in memory.

`block_k` controls how many placements are processed together (default `8`).
Reduce it to lower peak memory use, or increase it when GPU memory permits. It
does not change the field returned for a placement.

```python
for result in subject.iter_simulate(coil, placements, didt=1.0e6, block_k=4):
    ...
```

### Many subjects

A `Subject` retains GPU resources until it is freed. Use a context manager, or
call `subject.free()`, when processing multiple meshes.

```python
from pathlib import Path

for mesh_file in Path("subjects").glob("m2m_*/*.msh"):
    with Subject.from_mesh(mesh_file) as subject:
        result = subject.simulate(coil, placement, didt=1.0e6)
        print(result.peak_magnE())
```

## Results

Each `FieldResult` provides gray-matter summary metrics by default:

```python
result.peak_magnE()
result.focality(0.5)
result.summary["distribution"]["p99"]
```

The metric API includes peak field and location, stimulated volume,
field-weighted centre of gravity, and volume-weighted distribution statistics.
`peak_magnE()` reports the maximum `|E|`. `focality(frac)` reports the volume at
or above `frac` times the volume-weighted 99.9th percentile of `|E|`.

Full-volume arrays are not retained unless requested. Request only the data
needed for downstream analysis:

```python
result = subject.simulate(coil, placement, didt=1.0e6, magnitude=True)

for result in subject.iter_simulate(
    coil, placements, magnitude=True, vectors=True, potential=True
):
    ...
```

`result.magnE`, `result.E`, and `result.v` are `None` unless the corresponding
array was requested. Retaining `magnitude` also enables `summary_for(region)`
for a non-default tissue and `focality(frac)` at an arbitrary fraction.

```python
gray_matter = result.summary_for("gray_matter")
whole_model = result.summary_for("all")
```

Results are returned as NumPy arrays. For device-resident fields, use
`cunibs.fem.solve_placements_block` directly.

| Attribute | Description | Units |
| --- | --- | --- |
| `E` | Electric field per tetrahedron | V/m |
| `magnE` | Electric-field magnitude per tetrahedron | V/m |
| `v` | Electric scalar potential per node | V |
| `transform` | Coil-to-head affine transform | translation in mm |
| `vols` | Tetrahedron volumes | m³ |
| `tet_tags` | Volume tissue tags | dimensionless |
| `barycenters_mm` | Tetrahedron barycentres | mm |
| `didt` | Coil-current rate of change | A/s |
| `recovery` | Recovery method used for `E`, `magnE`, and `summary` | dimensionless |
| `E_slots` | Recovered field per slot when `nodal=True` | V/m |
| `slot_node` | Mesh node for each slot when `nodal=True` | dimensionless |
| `slot_tag` | Tissue tag for each slot, or `None` for node slots | dimensionless |
| `n_nodes` | Mesh node count | dimensionless |

Save a result and its retained fields to HDF5:

```python
result.save("placement.h5")

from cunibs import FieldResult

loaded = FieldResult.load("placement.h5")
```

Fields that were not retained are absent from the file and load as `None`.

## Field recovery

The raw finite-element field is constant within each tetrahedron. The
`recovery` argument selects the reported field:

| `recovery` | Use case |
| --- | --- |
| `"harmonic"` | Default. Use for cuNIBS analyses, particularly near tissue interfaces. |
| `"raw"` | Use the unprocessed per-tetrahedron field. |
| `"spr_tissue"` | Use when comparing with SimNIBS surface or volume overlays. |
| `"spr_global"` | Use the SimNIBS `continuous=True` convention. |

```python
result = subject.simulate(coil, placement, didt=1.0e6, magnitude=True)
raw = subject.simulate(coil, placement, didt=1.0e6, magnitude=True, recovery="raw")
```

`E`, `magnE`, and `summary` always describe the selected recovery method. The
raw field is not retained alongside a recovered field.

### Nodal fields

Request `nodal=True` when interpolating a recovered field to a cortical surface
or another node-based representation:

```python
result = subject.simulate(coil, placement, magnitude=True, nodal=True)

gray_field = result.nodal_field()
csf_field = result.nodal_field("csf")
```

For tissue-restricted recovery, a boundary node can hold one value for each
incident tissue. `nodal_field(region)` returns the requested tissue value and
uses `NaN` where that tissue does not reach the node. `region="all"` is not
available for these recovery modes. `E_slots`, `slot_node`, and `slot_tag`
provide the underlying arrays.

## Conductivity uncertainty quantification

`simulate_conductivity_uq` estimates the effect of uncertain tissue
conductivities for one placement or a placement sequence. Configure independent
tissue distributions with `ConductivityUQConfig`; conductivities are sampled
from lognormal distributions by default.

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

The result includes summary metrics for the per-tetrahedron mean field. Pass
`moments=True` to retain `mean_magnE`, `std_magnE`, and `cov_magnE` arrays.
Use `iter_simulate_conductivity_uq` to stream placements.

For statistics of scalar outcomes across draws, record regions of interest or
use the supplied sample arrays:

```python
m1 = subject.roi([-45.0, -5.0, 25.0], radius_mm=5.0, region="gray_matter")

uq_result = subject.simulate_conductivity_uq(
    coil, placement, config, didt=1.0e6, record_rois={"M1": m1}
)

uq_result.roi_samples["M1"]
uq_result.peak_samples
uq_result.focality_samples
uq_result.peak_location_samples
uq_result.tissue_sensitivity("peak")
```

`peak_mean_magnE` and `max_local_cov` summarize the moment fields, rather than
the distribution of a metric across draws. `tissue_sensitivity` is a
prior-weighted linear-in-log sensitivity index, not a Sobol estimator.

Save and load uncertainty results with `uq_result.save(path)` and
`ConductivityUQResult.load(path)`.

## Coil-placement optimization

`cunibs.adm` implements the Auxiliary Dipole Method (Gomez et al., 2021) for
searching candidate coil positions and in-plane rotations without a forward FEM
solve for every candidate.

```python
import numpy as np

from cunibs import Subject, Target, adm
from cunibs.coil import Coil, MAGSTIM_D70

subject = Subject.from_mesh("subject.msh")
coil = Coil.load(MAGSTIM_D70)

target = Target(position_mm=[-45.0, -5.0, 25.0], region="gray_matter")
centers = np.array([[x, y, 80.0] for x in range(-30, 31, 5) for y in range(-30, 31, 5)])

result = adm.optimize(subject.context, coil, target, centers)

result.best_objective
result.best_center_mm
result.best_angle_rad
```

Omit `Target.direction` to maximize `|E|`; provide a direction to maximize its
component along that direction. Candidate centres are projected onto the scalp.

For repeated searches with the same target and candidate centres, build and
reuse the reciprocity field:

```python
recip = adm.build_reciprocity(subject.context, coil, target, centers)
E = adm.evaluate(recip, coil, placements, didt=1.0e6)
```

## Reproducibility

A placement result is determined by the mesh, coil model, placement, `didt`,
solve tolerance, and recovery method. It is independent of `block_k`, placement
order, and other placements in a sweep. Repeating the same calculation on the
same software and hardware configuration reproduces the result bitwise.

Results can differ across GPU architectures, CUDA versions, compiler versions,
and dependency versions. A conductivity-UQ draw depends only on its conductivity
vector; the same seed reproduces the same draws at any `n_samples`.

## Citation

No archival citation is available yet. For reproducible academic use, cite the
software name, author, version, and Git commit. Archive the input mesh, coil
model, placement parameters, and analysis settings with the study data.

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
  228, 117696.
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
