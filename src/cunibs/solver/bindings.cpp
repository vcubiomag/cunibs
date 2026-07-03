#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>

#include "kernels.hpp"
#include "solver.hpp"

namespace nb = nanobind;

// Contiguity is required because these arrays are passed to CUDA and AMGx as raw pointers.
using f64_cuda = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
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
            "solve",
            [](AMGXSolver& self, f64_cuda b, f64_cuda x) {
                int n = static_cast<int>(b.shape(0));
                self.solve(n, b.data(), x.data());
            },
            nb::arg("b"), nb::arg("x"),
            "Solve A x = b on device; x (length n) is overwritten with the solution.");

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
}
