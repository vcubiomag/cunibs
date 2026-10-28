# cuNIBS

[![pypi](https://img.shields.io/pypi/v/cunibs)](https://pypi.python.org/pypi/cunibs)

cuNIBS computes the electric field that transcranial magnetic stimulation (TMS)
induces in an individual head model. It solves the full-volume finite-element
model on an NVIDIA GPU and reads head models built with the SimNIBS
[CHARM](https://simnibs.github.io/simnibs/build/html/documentation/command_line/charm.html)
pipeline.

Many TMS analyses solve the same head model hundreds or thousands of times.
cuNIBS builds the solver for a head once, keeps it on the GPU, and reuses it for
every solve. Setup takes under a second, and each additional placement takes
milliseconds. Use it to:

- simulate one coil placement or a sweep of placements,
- run the same analysis across a cohort of subjects,
- estimate how uncertain tissue conductivities affect the field,
- find the coil position and orientation that best stimulates a target.

The accompanying manuscript describes the numerical method and its validation
against analytic solutions and SimNIBS.

## Installation

```bash
python -m pip install cunibs
```

cuNIBS needs Python 3.12 or later, an NVIDIA GPU, and a driver that supports
CUDA 13 (r580 or later). Wheels are available for x86-64 Linux and Windows, for
CPython 3.12 through 3.14, including free-threaded Python 3.14. The package
installs its own CUDA libraries, so a separate CUDA toolkit is not needed.

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
print(result.focality(0.5))
```

`center_mm` is the target on the scalp and `handle_mm` sets the direction of the
coil handle. cuNIBS projects the target onto the scalp and places the coil
`distance_mm` above it. `didt` is the rate of change of the coil current in A/s.
The field scales linearly with it.

## Inputs

### Head models

Build a head model from MRI with CHARM:

```bash
charm subject_id T1w.nii.gz T2w.nii.gz
```

This writes `m2m_subject_id/subject_id.msh`. A T1-weighted image is enough; a
T2-weighted image improves the skull segmentation. Check the segmentation before
you simulate.

cuNIBS assigns each tissue the standard SimNIBS conductivity. To check a mesh
before using it, call `validate_mesh("subject.msh")`. It returns a report instead
of raising, so you can screen a whole cohort up front.

A mesh can be saved as HDF5, which is smaller and loads faster than the `.msh`.
Pass the CHARM `eeg_positions` directory to keep the electrode positions in the
same file. `load_mesh`, `Subject.from_mesh` and `validate_mesh` read either format,
and `read_cap` reads the positions back.

```python
from cunibs import Subject, load_mesh
from cunibs.head import read_cap

load_mesh("m2m_subject_id/subject_id.msh").save(
    "subject_id.h5", eeg_positions="m2m_subject_id/eeg_positions"
)
subject = Subject.from_mesh("subject_id.h5")
c3 = read_cap("subject_id.h5")["C3"]
```

### Coils

cuNIBS includes the 25 validated coil models from Drakaki et al. (2022). Load
one by constant or by name, and call `bundled_coils()` to list them. SimNIBS
`.ccd` files load directly.

```python
coil = Coil.load("MagStim_D70")
coil = Coil.load("my_coil.ccd")
```

## Placement sweeps and cohorts

Pass a list of placements to `iter_simulate`. It yields results in input order
and reuses the subject setup for all of them.

```python
placements = [
    Placement([0.0, 20.0, 80.0], [0.0, 70.0, 80.0]),
    Placement([20.0, 0.0, 80.0], [70.0, 0.0, 80.0]),
]

for result in subject.iter_simulate(coil, placements, didt=1.0e6):
    print(result.peak_magnE())
```

A `Subject` holds GPU memory until it is freed. Use it as a context manager when
you process many subjects:

```python
from pathlib import Path

for mesh_file in Path("subjects").glob("m2m_*/*.msh"):
    with Subject.from_mesh(mesh_file) as subject:
        result = subject.simulate(coil, placement, didt=1.0e6)
```

## Results

Each result carries summary metrics for gray matter: peak field and its
location, focality, centre of gravity, and distribution statistics. Results stay
small because full-volume arrays are kept only when you ask for them:

```python
result = subject.simulate(
    coil, placement, didt=1.0e6, magnitude=True, vectors=True, potential=True
)

result.magnE  # |E| per tetrahedron, V/m
result.E  # E per tetrahedron, V/m
result.v  # potential per node, V
```

Save and load results as HDF5 with `result.save(path)` and
`FieldResult.load(path)`. Saved files record the software versions, GPU, mesh,
and coil used.

The `recovery` argument selects how the reported field is computed from the
finite-element solution. The default, `"harmonic"`, is accurate at tissue
interfaces. Use `"spr_tissue"` or `"spr_global"` to compare with SimNIBS
outputs, or `"raw"` for the unprocessed field.

## Conductivity uncertainty

`simulate_conductivity_uq` samples tissue conductivities and reports the mean
field and its variability.

```python
from cunibs import ConductivityUQConfig

config = ConductivityUQConfig(
    n_samples=500,
    tissue_cov={"gray_matter": 0.15, "csf": 0.05, "cortical_bone": 0.35},
    seed=1,
)

uq = subject.simulate_conductivity_uq(coil, placement, config, didt=1.0e6)

print(uq.peak_mean_magnE())
print(uq.max_local_cov())
print(uq.peak_samples)
```

`tissue_cov` sets the coefficient of variation for each tissue. Only the tissues
it names are varied. Samples are lognormal by default.

## Coil-placement optimization

`cunibs.adm` searches candidate coil positions and orientations for the one that
maximizes the field at a target. It uses the auxiliary dipole method (Gomez et
al., 2021), so it does not need a separate solve for each candidate.

```python
from cunibs import Target, adm, head

target = Target(position_mm=[-45.0, -5.0, 25.0], region="gray_matter")
centers = head.scalp_grid(subject, [-50.0, -10.0, 80.0], radius_mm=20.0, spacing_mm=2.0)

search = adm.optimize(subject, coil, target, centers)

result = subject.simulate(coil, search.best_placement)
```

Give `Target` a `direction` to maximize the field along that direction instead
of its magnitude.

`cunibs.head` also measures scalp-to-cortex distance and tissue thickness, and
reads the electrode positions and fiducials that CHARM writes.

## Reproducibility

A result depends only on the mesh, coil, placement, `didt`, solver tolerance,
and recovery method. It does not depend on the other placements in a sweep or
their order. The same calculation on the same hardware and software gives
bitwise-identical results. Results can differ across GPU models and software
versions.

## Citation

No archival citation is available yet. Cite the software name, author, version,
and Git commit, and archive the mesh, coil, placements, and settings with your
study data.

## License

cuNIBS is released under the GNU General Public License v3.0.

## References

- Drakaki, M., Mathiesen, C., Siebner, H. R., Madsen, K., and Thielscher, A.
  (2022). [Database of 25 validated coil models for electric field simulations
  for TMS](https://doi.org/10.1016/j.brs.2022.04.017). *Brain Stimulation*,
  15(3), 697-706.
- Gomez, L. J., Dannhauer, M., and Peterchev, A. V. (2021). [Fast computational
  optimization of TMS coil placement for individualized electric field
  targeting](https://doi.org/10.1016/j.neuroimage.2020.117696). *NeuroImage*,
  228, 117696.
- Puonti, O., Van Leemput, K., Saturnino, G. B., Siebner, H. R., Madsen, K. H.,
  and Thielscher, A. (2020). [Accurate and robust whole-head segmentation from
  magnetic resonance images for individualized head
  modeling](https://doi.org/10.1016/j.neuroimage.2020.117044). *NeuroImage*,
  219, 117044.
- Saturnino, G. B., Puonti, O., Nielsen, J. D., Antonenko, D., Madsen, K. H.,
  and Thielscher, A. (2019). [SimNIBS 2.1: A comprehensive pipeline for
  individualized electric field modelling for transcranial brain
  stimulation](https://doi.org/10.1007/978-3-030-21293-3_1).
