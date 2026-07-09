#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <new>
#include <string>

#include "kernels.hpp"
#include "solver.hpp"

namespace nb = nanobind;

// Contiguity is required because these arrays are passed to CUDA and AMGx as raw pointers.
using f64_cuda = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
using f64_cuda_2d = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cuda>;
using i32_cuda = nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cuda>;

using f32_cuda_1d = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
using f32_cuda_2d = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cuda>;
using f32_cuda_3d = nb::ndarray<float, nb::ndim<3>, nb::c_contig, nb::device::cuda>;
using i32_cuda_2d = nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cuda>;

NB_MODULE(_solver_ext, m) {
    nb::class_<AMGXSolver>(m, "AMGXSolver")
        .def(nb::init<const std::string&>(), nb::arg("config"))
        .def(
            "setup",
            [](AMGXSolver& self, i32_cuda row_ptr, i32_cuda col_idx, f64_cuda values) {
                int n = static_cast<int>(row_ptr.shape(0)) - 1;
                int nnz = static_cast<int>(values.shape(0));
                self.setup(n, nnz, row_ptr.data(), col_idx.data(), values.data());
            },
            nb::arg("row_ptr"), nb::arg("col_idx"), nb::arg("values"),
            "Upload the reduced CSR (device pointers) and build the AMG hierarchy once.")
        .def(
            "update_coefficients",
            [](AMGXSolver& self, f64_cuda values) {
                self.update_coefficients(static_cast<int>(values.shape(0)), values.data());
            },
            nb::arg("values"),
            "Replace matrix values (device pointer) keeping the sparsity pattern; no re-analysis.")
        .def("resetup", &AMGXSolver::resetup,
             "Rebuild the numeric AMG hierarchy for the current values (reuses structure per config).")
        .def("iterations", &AMGXSolver::iterations,
             "Return the iteration count from the most recent solve.")
        .def(
            "solve",
            [](AMGXSolver& self, f64_cuda b, f64_cuda x) {
                int n = static_cast<int>(b.shape(0));
                self.solve(n, b.data(), x.data());
            },
            nb::arg("b"), nb::arg("x"),
            "Solve A x = b on device; x (length n) is overwritten with the solution.")
        .def(
            "apply",
            [](AMGXSolver& self, f64_cuda b, f64_cuda x) {
                int n = static_cast<int>(b.shape(0));
                self.apply(n, b.data(), x.data());
            },
            nb::arg("b"), nb::arg("x"),
            "Run the configured solver once without enforcing convergence.");

    nb::class_<AMGXFloatSolver>(m, "AMGXFloatSolver")
        .def(nb::init<const std::string&>(), nb::arg("config"))
        .def(
            "setup",
            [](AMGXFloatSolver& self, i32_cuda row_ptr, i32_cuda col_idx, f32_cuda_1d values) {
                int n = static_cast<int>(row_ptr.shape(0)) - 1;
                int nnz = static_cast<int>(values.shape(0));
                self.setup(n, nnz, row_ptr.data(), col_idx.data(), values.data());
            },
            nb::arg("row_ptr"), nb::arg("col_idx"), nb::arg("values"))
        .def(
            "apply",
            [](AMGXFloatSolver& self, f32_cuda_1d b, f32_cuda_1d x) {
                int n = static_cast<int>(b.shape(0));
                self.apply(n, b.data(), x.data());
            },
            nb::arg("b"), nb::arg("x"))
        .def("iterations", &AMGXFloatSolver::iterations);

    nb::class_<PcgAmgSolver>(m, "PcgAmgSolver")
        .def(
            "__init__",
            [](PcgAmgSolver* self, i32_cuda row_ptr, i32_cuda col_idx, f64_cuda values) {
                int n = static_cast<int>(row_ptr.shape(0)) - 1;
                int nnz = static_cast<int>(values.shape(0));
                new (self) PcgAmgSolver(n, nnz, row_ptr.data(), col_idx.data(), values.data());
            },
            nb::arg("row_ptr"), nb::arg("col_idx"), nb::arg("values"))
        .def(
            "update_values",
            [](PcgAmgSolver& self, f64_cuda values) { self.update_values(values.data()); },
            nb::arg("values"))
        .def(
            "solve",
            [](PcgAmgSolver& self, AMGXSolver& preconditioner, f64_cuda b, f64_cuda x,
               double tolerance, int max_iters) {
                PcgResult result =
                    self.solve(preconditioner, b.data(), x.data(), tolerance, max_iters);
                return nb::make_tuple(result.iterations, result.relative_residual);
            },
            nb::arg("preconditioner"), nb::arg("b"), nb::arg("x"), nb::arg("tolerance"),
            nb::arg("max_iters"))
        .def(
            "solve_mixed",
            [](PcgAmgSolver& self, AMGXFloatSolver& preconditioner, f64_cuda b, f64_cuda x,
               double tolerance, int max_iters) {
                PcgResult result =
                    self.solve_mixed(preconditioner, b.data(), x.data(), tolerance, max_iters);
                return nb::make_tuple(result.iterations, result.relative_residual);
            },
            nb::arg("preconditioner"), nb::arg("b"), nb::arg("x"), nb::arg("tolerance"),
            nb::arg("max_iters"));

    m.def(
        "pcg_amg_solve",
        [](i32_cuda row_ptr, i32_cuda col_idx, f64_cuda values, AMGXSolver& preconditioner,
           f64_cuda b, f64_cuda x, double tolerance, int max_iters) {
            int n = static_cast<int>(row_ptr.shape(0)) - 1;
            int nnz = static_cast<int>(values.shape(0));
            PcgResult result = pcg_amg_solve(n, nnz, row_ptr.data(), col_idx.data(),
                                             values.data(), preconditioner, b.data(), x.data(),
                                             tolerance, max_iters);
            return nb::make_tuple(result.iterations, result.relative_residual);
        },
        nb::arg("row_ptr"), nb::arg("col_idx"), nb::arg("values"), nb::arg("preconditioner"),
        nb::arg("b"), nb::arg("x"), nb::arg("tolerance"), nb::arg("max_iters"),
        "Run double outer PCG with an AMGx preconditioner apply.");

    m.def(
        "dadt_nbody",
        [](f32_cuda_2d s, f32_cuda_2d mp, f32_cuda_1d sn, f32_cuda_2d r, f32_cuda_2d out,
           float didt, float mu0_4pi, uintptr_t stream) {
            int n_dip = static_cast<int>(s.shape(0));
            int n_nodes = static_cast<int>(r.shape(0));
            launch_dadt(s.data(), mp.data(), sn.data(), r.data(), out.data(), n_dip, n_nodes,
                        didt, mu0_4pi, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("s"), nb::arg("mp"), nb::arg("sn"), nb::arg("r"), nb::arg("out"), nb::arg("didt"),
        nb::arg("mu0_4pi"), nb::arg("stream"),
        "Fused dA/dt at nodes from placed magnetic dipoles; writes into caller-allocated out.");

    m.def(
        "dadt_node_to_element",
        [](f32_cuda_2d dadt_nodes, i32_cuda_2d tet_nodes, f32_cuda_2d out, uintptr_t stream) {
            int n_tet = static_cast<int>(tet_nodes.shape(0));
            launch_dadt_element_average(dadt_nodes.data(), tet_nodes.data(), out.data(), n_tet,
                                        reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("dadt_nodes"), nb::arg("tet_nodes"), nb::arg("out"), nb::arg("stream"),
        "Average nodal dA/dt onto tetrahedra; writes into caller-allocated out.");

    m.def(
        "rhs_assemble",
        [](f32_cuda_2d dadt_elm, f32_cuda_3d g, f32_cuda_1d neg_vc, i32_cuda ptr, i32_cuda idx,
           f32_cuda_1d b, uintptr_t stream) {
            int n_nodes = static_cast<int>(b.shape(0));
            launch_rhs(dadt_elm.data(), g.data(), neg_vc.data(), ptr.data(), idx.data(), b.data(),
                       n_nodes, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("dadt_elm"), nb::arg("g"), nb::arg("neg_vc"), nb::arg("ptr"), nb::arg("idx"),
        nb::arg("b"), nb::arg("stream"),
        "Deterministic node-centric RHS assembly; writes into caller-allocated b.");

    m.def(
        "rhs_assemble_weighted",
        [](f32_cuda_2d dadt_elm, f32_cuda_3d wg, i32_cuda ptr, i32_cuda idx, f32_cuda_1d b,
           uintptr_t stream) {
            int n_nodes = static_cast<int>(b.shape(0));
            launch_rhs_weighted(dadt_elm.data(), wg.data(), ptr.data(), idx.data(), b.data(),
                                n_nodes, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("dadt_elm"), nb::arg("wg"), nb::arg("ptr"), nb::arg("idx"), nb::arg("b"),
        nb::arg("stream"),
        "Deterministic node-centric RHS assembly with preweighted gradients.");

    m.def(
        "weighted_gradient",
        [](f32_cuda_3d g, f32_cuda_1d neg_vc, f32_cuda_3d wg, uintptr_t stream) {
            int n_tet = static_cast<int>(neg_vc.shape(0));
            launch_weighted_gradient(g.data(), neg_vc.data(), wg.data(), n_tet,
                                     reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("g"), nb::arg("neg_vc"), nb::arg("wg"), nb::arg("stream"),
        "Precompute neg_vc-scaled gradients for repeated RHS assembly.");

    m.def(
        "reconstruct_e",
        [](f64_cuda v, i32_cuda_2d tet_nodes, f32_cuda_3d g, f32_cuda_2d dadt_elm,
           f32_cuda_2d e_out, f32_cuda_1d magn_out, uintptr_t stream) {
            int n_tet = static_cast<int>(tet_nodes.shape(0));
            launch_reconstruct(v.data(), tet_nodes.data(), g.data(), dadt_elm.data(), e_out.data(),
                               magn_out.data(), n_tet, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("v"), nb::arg("tet_nodes"), nb::arg("g"), nb::arg("dadt_elm"), nb::arg("e_out"),
        nb::arg("magn_out"), nb::arg("stream"),
        "Element-centric E/magnE reconstruction; writes into caller-allocated e_out/magn_out.");

    m.def(
        "place_transforms",
        [](f64_cuda_2d centers, f64_cuda_2d handles, f64_cuda dists, f64_cuda_2d a, f64_cuda_2d b,
           f64_cuda_2d c, f64_cuda_2d tnorm, f64_cuda_2d out, uintptr_t stream) {
            int n_pl = static_cast<int>(centers.shape(0));
            int n_tri = static_cast<int>(a.shape(0));
            launch_place(centers.data(), handles.data(), dists.data(), a.data(), b.data(), c.data(),
                         tnorm.data(), out.data(), n_pl, n_tri,
                         reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("centers"), nb::arg("handles"), nb::arg("dists"), nb::arg("a"), nb::arg("b"),
        nb::arg("c"), nb::arg("tnorm"), nb::arg("out"), nb::arg("stream"),
        "Batched closest-point scalp projection + coil frame; writes (P,16) row-major 4x4 out.");
}
