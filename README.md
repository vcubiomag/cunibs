# cuNIBS

[![pypi](https://img.shields.io/pypi/v/cunibs)](https://pypi.python.org/pypi/cunibs)

cuNIBS computes the electric field induced by transcranial magnetic stimulation
(TMS) in a tetrahedral head model. It uses first-order finite elements, magnetic
dipole coil models, CUDA kernels, and an AMGx-preconditioned linear solve. Mesh
state and the AMG hierarchy remain on the GPU and are reused across coil
placements.

The package is intended for computational research. It currently supports
isotropic conductivity models, conductivity uncertainty quantification, and
NVIDIA GPUs.

## Numerical method

Under the magneto-quasistatic approximation, the electric field is

$$\mathbf{E} = -\nabla v - \frac{\partial \mathbf{A}}{\partial t}$$

where $v$ is the electric scalar potential and $\mathbf{A}$ is the magnetic
vector potential. For piecewise constant isotropic conductivity $\sigma$, the
potential satisfies

$$\nabla \cdot \left(\sigma \nabla v\right) = -\nabla \cdot \left(\sigma \frac{\partial \mathbf{A}}{\partial t}\right)$$

cuNIBS discretizes this equation with linear basis functions on tetrahedra. For
tetrahedron $e$, the element matrix is

$$K_{ij}^{(e)} = V_e \sigma_e \nabla \lambda_i \cdot \nabla \lambda_j$$

The right-hand side uses the mean nodal value of
$\partial\mathbf{A}/\partial t$ in each tetrahedron. One potential degree of
freedom is fixed to remove the additive null space. The resulting symmetric
positive-definite system is solved with preconditioned conjugate gradients and
an aggregation AMG preconditioner.

The coil field is evaluated from magnetic dipoles:

$$\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi} \sum_j \frac{\mathbf{m}_j \times (\mathbf{r} - \mathbf{s}_j)}{\lVert \mathbf{r} - \mathbf{s}_j \rVert^3}$$

The implementation uses float64 for stiffness assembly and the scalar
potential. Placement-dependent field kernels use float32. Electric-field
reconstruction accumulates $\nabla v$ in float64 before conversion to float32.
The right-hand side uses a fixed per-node corner reduction order.

## Installation

cuNIBS requires Python 3.12 or later and an NVIDIA GPU with a compatible
driver. Install the repository with pip:

```bash
python -m pip install cunibs
```

The build installs the required Python packages, CUDA components, and the
bundled AMGx library.

## Input data

### Head mesh

`Subject.from_mesh` reads binary Gmsh 2.2 files. The mesh must contain
first-order tetrahedra and an oriented scalp surface. Coordinates are interpreted
in millimetres. Volume tags select the built-in isotropic tissue
conductivities. The scalp surface must use tag `1005`.

Generate individualized head models with the SimNIBS
[CHARM](https://simnibs.github.io/simnibs/build/html/documentation/command_line/charm.html)
pipeline:

```bash
charm subject_id T1w.nii.gz T2w.nii.gz
```

CHARM writes the final mesh to `m2m_subject_id/subject_id.msh`. A T1-weighted
scan is sufficient, but a T2-weighted scan improves skull segmentation. Inspect
the generated segmentation before simulation. The method is described by
Puonti et al. (2020).

The conductivity assignments follow the standard SimNIBS values. The loader
recognizes the following volume tags:

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

Unsupported volume tags are removed when the mesh is loaded. Surface triangles
that do not use a recognized surface tag are also removed.

### Coil model

`Coil.load` reads the HDF5 dipole format used by the bundled coil models.
Dipole positions use metres and dipole moments use A m². Models are available as
constants in `cunibs.coil`. The package includes the 25 validated coil models
reported by Drakaki et al. (2022), covering common coils from several
manufacturers.

Import a SimNIBS CCD coil by converting it to HDF5:

```python
from pathlib import Path

from cunibs.coil import Coil, encode_ccd

encode_ccd(Path("coil.ccd"), Path("coil.h5"))
coil = Coil.load("coil.h5")
```

## Usage

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
print(result.summary)
```

`center_mm` specifies the scalp target. `handle_mm` specifies a point in the
positive coil-handle direction. cuNIBS projects the target onto the scalp,
constructs the coil frame from the local surface normal, and applies
`distance_mm` along the outward normal.

Pass a sequence of placements to reuse the assembled system and AMG hierarchy:

```python
placements = [
    Placement([0.0, 20.0, 80.0], [0.0, 70.0, 80.0]),
    Placement([20.0, 0.0, 80.0], [70.0, 0.0, 80.0]),
]

results = subject.simulate(coil, placements, didt=1.0e6)
```

The first call builds the GPU solver state. Later calls on the same `Subject`
reuse it. By default, `simulate` returns compact CPU-side summaries and does not
retain full-volume field arrays.

## Results

`FieldSummary` contains the placement metadata, the coil-to-head transform, and
the gray-matter metric summary. Retain raw fields explicitly when per-tetrahedron
arrays are needed:

```python
field = subject.simulate(coil, placement, didt=1.0e6, retain_fields=True)
gpu_field = subject.simulate(
    coil, placement, didt=1.0e6, retain_fields=True, device="gpu"
)
```

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

The metric API reports the peak field, peak location, stimulated volume,
field-weighted centre of gravity, and volume-weighted distribution statistics.
Metrics can be computed over gray matter or the complete volume when fields are
retained:

```python
gray_matter = result.summary("gray_matter")
whole_model = result.summary("all")
```

Save a result and its metric inputs to HDF5:

```python
field.save("placement.h5")

from cunibs import FieldResult

loaded = FieldResult.load("placement.h5")
```

## Conductivity uncertainty quantification

`ConductivityUQConfig` runs a Monte Carlo analysis over tissue conductivities for
one coil placement or a sequence of placements. Each sampled conductivity vector
is solved with the same finite-element model, and `ConductivityUQResult` reports
per-tetrahedron moments of the electric-field magnitude.

For tissue tag $t$, the default model treats the conductivity as an independent
random variable with nominal value $\sigma_{0,t}$ and coefficient of variation
$c_t$. The default distribution is lognormal:

$$\sigma_t^{(k)} = \sigma_{0,t}\exp\left(s_t z_k - \frac{s_t^2}{2}\right), \qquad
s_t = \sqrt{\log(1 + c_t^2)}, \qquad z_k \sim \mathcal{N}(0,1)$$

This parameterization keeps conductivities positive and preserves the nominal
mean, $\mathbb{E}[\sigma_t] = \sigma_{0,t}$. The result stores the sampled
conductivities and the Monte Carlo estimates

$$\bar{E}_e = \frac{1}{N}\sum_{k=1}^{N} |E_e^{(k)}|, \qquad
s_e = \sqrt{\frac{1}{N-1}\sum_{k=1}^{N}(|E_e^{(k)}|-\bar{E}_e)^2}, \qquad
\mathrm{CoV}_e = \frac{s_e}{\bar{E}_e}$$

where $e$ indexes tetrahedra. The finite-element matrix and right-hand side are
linear in the tissue conductivities, so cuNIBS precomputes per-tissue stiffness
and right-hand-side components once and reuses the matrix sparsity pattern across
samples.

```python
from cunibs import ConductivityUQConfig, Placement, Subject
from cunibs.coil import Coil, MAGSTIM_D70

subject = Subject.from_mesh("subject.msh")
coil = Coil.load(MAGSTIM_D70)

placement = Placement(
    center_mm=[0.0, 20.0, 80.0],
    handle_mm=[0.0, 70.0, 80.0],
    distance_mm=4.0,
)

config = ConductivityUQConfig(
    n_samples=500,
    tissue_cov={2: 0.15, 3: 0.05, 7: 0.35, 8: 0.35},
    seed=1,
)

uq_result = subject.simulate(coil, placement, didt=1.0e6, conductivity_uq=config)

print(uq_result.peak_mean_magnE())
print(uq_result.peak_cov())
```

By default, conductivity UQ returns compact CPU-side metrics. Retain the
per-tetrahedron moment arrays explicitly:

```python
uq_fields = subject.simulate(
    coil,
    placement,
    didt=1.0e6,
    conductivity_uq=config,
    retain_fields=True,
)
```

`mean_magnE`, `std_magnE`, and `cov_magnE` use the same tetrahedron ordering as
`FieldResult.magnE` when fields are retained. `peak_mean_magnE` and `peak_cov`
accept the same region names as the deterministic metric API.

Save a conductivity-UQ result to HDF5:

```python
uq_fields.save("conductivity_uq.h5")

from cunibs import ConductivityUQResult

loaded = ConductivityUQResult.load("conductivity_uq.h5")
```

## Coil-placement optimization (ADM)

`cunibs.adm` implements the Auxiliary Dipole Method for fast coil-placement
optimization. A few one-time adjoint solves, reusing the forward AMG hierarchy,
sample a reciprocity field on a regular grid. The target E-field of any placement
is then a trilinear interpolation plus a dipole sum, with no further FEM solve.
This evaluates candidate placements orders of magnitude faster than a forward
solve per candidate, and matches a forward solve at the optimum to a relative
error of 4e-4.

```python
from cunibs import Subject, Target, adm
from cunibs.coil import Coil, MAGSTIM_D70
import numpy as np

subject = Subject.from_mesh("subject.msh")
coil = Coil.load(MAGSTIM_D70)

# Omit `direction` to maximize |E| (three adjoint solves), or pass one to
# maximize a directional component.
target = Target(position_mm=[-45.0, -5.0, 25.0], region="gray_matter")

# Candidate scalp positions to search (each is projected onto the scalp).
centers = np.array([[x, y, 80.0] for x in range(-30, 31, 5) for y in range(-30, 31, 5)])

result = adm.optimize(subject.context, coil, target, centers)

print(result.best_objective)     # peak |E| at the target (V/m)
print(result.best_center_mm)     # optimal scalp position
print(result.best_angle_rad)     # optimal in-plane rotation
```

The in-plane rotation is optimized in closed form: the target E-field is a rigid
rotation of the coil, so each component is band-limited in the angle. It is
sampled at `n_samples` angles, trigonometrically interpolated, and `|E(θ)|²` is
maximized analytically.

For repeated queries against a fixed target, such as uncertainty quantification
over a distribution of placements, build the reciprocity field once and reuse it:

```python
recip = adm.build_reciprocity(subject.context, coil, target, centers)
E = adm.evaluate(recip, coil, placements, didt=1.0e6)   # (P, D) target E-vectors
```

## Reproducibility

The solver configuration uses deterministic AMGx execution and a fixed
right-hand-side reduction order. Floating-point results can still vary across
GPU architectures, CUDA versions, compiler versions, and dependency versions.
The ADM adjoint solves use a tighter tolerance (`1e-9`) than the forward solve
because their near-point-source right-hand side is more sensitive.

## Citation

No archival citation is provided yet. For reproducible academic use, cite the
software by name, author, version, and Git commit, and archive the exact input
mesh, coil model, and placement parameters used in the analysis.

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
  Computing*, 37(5), S602-S626.
