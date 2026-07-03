# cuNIBS

cuNIBS computes the electric field induced by transcranial magnetic stimulation
(TMS) in a tetrahedral head model. It uses first-order finite elements, magnetic
dipole coil models, CUDA kernels, and an AMGx-preconditioned linear solve. Mesh
state and the AMG hierarchy remain on the GPU and are reused across coil
placements.

The package is intended for computational research. It currently supports
isotropic conductivity models and NVIDIA GPUs.

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
print(result.summary())
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
reuse it.

## Results

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
Metrics can be computed over gray matter or the complete volume:

```python
gray_matter = result.summary("gray_matter")
whole_model = result.summary("all")
```

Save a result and its metric inputs to HDF5:

```python
result.save("placement.h5")

from cunibs import FieldResult

loaded = FieldResult.load("placement.h5")
```

## Reproducibility

The solver configuration uses deterministic AMGx execution and a fixed
right-hand-side reduction order. Floating-point results can still vary across
GPU architectures, CUDA versions, compiler versions, and dependency versions.

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
