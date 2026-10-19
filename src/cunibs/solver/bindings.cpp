#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>

#include "amg/aggregate.hpp"
#include "amg/l1.hpp"
#include "amg/solver.hpp"
#include "amg/vcycle.hpp"
#include "coil/coil.hpp"
#include "fem/fem.hpp"

namespace nb = nanobind;

// Contiguity is required because these arrays are passed to CUDA as raw pointers.
using f64_cuda = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
using f64_cuda_2d = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cuda>;
using f64_cuda_3d = nb::ndarray<double, nb::ndim<3>, nb::c_contig, nb::device::cuda>;
using i32_cuda = nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cuda>;

using f32_cuda_1d = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
using f32_cuda_2d = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cuda>;
using f32_cuda_3d = nb::ndarray<float, nb::ndim<3>, nb::c_contig, nb::device::cuda>;
using i32_cuda_2d = nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cuda>;
using u8_cuda_1d = nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
using u8_cuda_2d = nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cuda>;

namespace {

// A grid's origin and spacing (metres) and shape, as the ADM kernels take them.
struct GridArgs {
    double origin[3];
    double spacing[3];
    int shape[3];
};

GridArgs grid_args(const std::vector<double>& origin, const std::vector<double>& spacing,
                   const std::vector<int>& shape) {
    if (origin.size() != 3 || spacing.size() != 3 || shape.size() != 3) {
        throw std::invalid_argument("grid origin, spacing and shape must each have 3 entries");
    }
    GridArgs g;
    for (int i = 0; i < 3; ++i) {
        if (shape[i] < 2 || !(spacing[i] > 0.0)) {
            throw std::invalid_argument("grid needs at least 2 samples and a positive spacing");
        }
        g.origin[i] = origin[i];
        g.spacing[i] = spacing[i];
        g.shape[i] = shape[i];
    }
    return g;
}

// A host-to-host copy run in stream order by the CUDA driver's own callback thread, which is what
// lets a pinned staging buffer drain into a pageable array while the calling thread goes on to
// queue GPU work. It never touches Python, so it needs no interpreter lock.
struct HostCopy {
    const void* src;
    void* dst;
    std::size_t bytes;
};

void CUDART_CB run_host_copy(void* arg) {
    auto* copy = static_cast<HostCopy*>(arg);
    std::memcpy(copy->dst, copy->src, copy->bytes);
    delete copy;
}

}  // namespace

NB_MODULE(_solver_ext, m) {
    // Compiled-in limits, exported so the Python layer never restates them.
    m.attr("MAX_STAGE_BLOCK") = kMaxStageBlock;
    nb::list widths;
    for (int w : kBlockWidths) widths.append(w);
    m.attr("BLOCK_SIZES") = nb::tuple(widths);

    nb::class_<NativeVCycle>(m, "NativeVCycle")
        .def(nb::init<>())
        .def("set_smoother", &NativeVCycle::set_smoother,
             nb::call_guard<nb::gil_scoped_release>(),
             nb::arg("degree"), nb::arg("lower_ratio"),
             "Set the Chebyshev degree and the lower end of its interval, as a divisor of the "
             "smoother's spectral upper bound. Must precede the first add_level.")
        .def(
            "add_level",
            [](NativeVCycle& self, i32_cuda row_ptr, i32_cuda col_idx, f32_cuda_1d values,
               f32_cuda_1d dinv, i32_cuda p_row_ptr, i32_cuda p_col_idx, f32_cuda_1d p_values,
               i32_cuda r_row_ptr, i32_cuda r_col_idx, f32_cuda_1d r_values, uintptr_t stream) {
                int n_rows = static_cast<int>(row_ptr.shape(0)) - 1;
                int nnz = static_cast<int>(values.shape(0));
                int n_coarse = static_cast<int>(r_row_ptr.shape(0)) - 1;
                int p_nnz = static_cast<int>(p_values.shape(0));
                if (static_cast<int>(r_values.shape(0)) != p_nnz) {
                    throw std::invalid_argument("R must be the transpose of P");
                }
                if (static_cast<int>(p_row_ptr.shape(0)) - 1 != n_rows) {
                    throw std::invalid_argument("P must have one row per fine row");
                }
                self.add_level(n_rows, nnz, n_coarse, p_nnz, row_ptr.data(), col_idx.data(),
                               values.data(), dinv.data(), p_row_ptr.data(), p_col_idx.data(),
                               p_values.data(), r_row_ptr.data(), r_col_idx.data(),
                               r_values.data(), reinterpret_cast<cudaStream_t>(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("row_ptr").noconvert(), nb::arg("col_idx").noconvert(),
            nb::arg("values").noconvert(), nb::arg("dinv").noconvert(),
            nb::arg("p_row_ptr").noconvert(), nb::arg("p_col_idx").noconvert(),
            nb::arg("p_values").noconvert(), nb::arg("r_row_ptr").noconvert(),
            nb::arg("r_col_idx").noconvert(), nb::arg("r_values").noconvert(),
            nb::arg("stream") = 0,
            "Append one non-coarsest level (fp32 CSR, 1/guard(d), prolongator CSR and its "
            "transpose); device arrays are copied into solver-owned buffers.")
        .def(
            "set_coarse",
            [](NativeVCycle& self, f32_cuda_2d ainv, uintptr_t stream) {
                if (ainv.shape(0) != ainv.shape(1)) {
                    throw std::invalid_argument("coarse inverse must be square");
                }
                self.set_coarse(static_cast<int>(ainv.shape(0)), ainv.data(),
                                reinterpret_cast<cudaStream_t>(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("ainv").noconvert(), nb::arg("stream") = 0,
            "Set the dense (row-major) inverse of the coarsest matrix.")
        .def("finalize", &NativeVCycle::finalize,
             "Validate level chain consistency; required before apply.")
        .def("n_levels", &NativeVCycle::n_levels,
             "Number of coarsening levels, excluding the coarsest (dense-inverse) one.");

    m.def(
        "select_size4",
        [](i32_cuda row_ptr, i32_cuda col_idx, f32_cuda_1d values, i32_cuda aggregates,
           uintptr_t stream) {
            int n_rows = static_cast<int>(row_ptr.shape(0)) - 1;
            int nnz = static_cast<int>(values.shape(0));
            if (static_cast<int>(aggregates.shape(0)) != n_rows) {
                throw std::invalid_argument("aggregates must have one entry per row");
            }
            if (col_idx.shape(0) != values.shape(0)) {
                throw std::invalid_argument("col_idx and values must have one entry per nonzero");
            }
            return select_size4(n_rows, nnz, row_ptr.data(), col_idx.data(), values.data(),
                                aggregates.data(), reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("row_ptr").noconvert(), nb::arg("col_idx").noconvert(),
        nb::arg("values").noconvert(), nb::arg("aggregates").noconvert(),
        nb::arg("stream") = 0,
        "Unsmoothed pairwise aggregation (AMGx SIZE_4). Fills `aggregates` with a surjective "
        "row -> aggregate map and returns the aggregate count. Synchronises the stream.");

    nb::class_<PcgAmgSolver>(m, "PcgAmgSolver")
        .def(
            "__init__",
            [](PcgAmgSolver* self, i32_cuda row_ptr, i32_cuda col_idx, f64_cuda values,
               uintptr_t stream) {
                int n = static_cast<int>(row_ptr.shape(0)) - 1;
                int nnz = static_cast<int>(values.shape(0));
                new (self) PcgAmgSolver(n, nnz, row_ptr.data(), col_idx.data(), values.data(),
                                        reinterpret_cast<cudaStream_t>(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("row_ptr").noconvert(), nb::arg("col_idx").noconvert(),
            nb::arg("values").noconvert(), nb::arg("stream") = 0)
        .def(
            "update_values",
            [](PcgAmgSolver& self, f64_cuda values, uintptr_t stream) {
                if (static_cast<int>(values.shape(0)) != self.nnz()) {
                    throw std::invalid_argument("values must have one entry per nonzero");
                }
                self.update_values(values.data(), reinterpret_cast<cudaStream_t>(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("values").noconvert(), nb::arg("stream"))
        .def(
            "solve_mixed",
            [](PcgAmgSolver& self, NativeVCycle& preconditioner, f64_cuda b, f64_cuda x,
               double tolerance, int max_iters, uintptr_t stream, std::optional<f64_cuda> x0) {
                const int n = self.n();
                if (static_cast<int>(b.shape(0)) != n || static_cast<int>(x.shape(0)) != n ||
                    (x0.has_value() && static_cast<int>(x0->shape(0)) != n)) {
                    throw std::invalid_argument("b, x and x0 must have one entry per row");
                }
                PcgResult result;
                {
                    nb::gil_scoped_release release;
                    result = self.solve_mixed(preconditioner, b.data(), x.data(), tolerance,
                                              max_iters, reinterpret_cast<cudaStream_t>(stream),
                                              x0.has_value() ? x0->data() : nullptr);
                }
                return nb::make_tuple(result.iterations, result.relative_residual);
            },
            nb::arg("preconditioner"), nb::arg("b").noconvert(), nb::arg("x").noconvert(),
            nb::arg("tolerance"), nb::arg("max_iters"), nb::arg("stream"),
            nb::arg("x0").noconvert() = nb::none())
        .def(
            "solve_mixed_block",
            [](PcgAmgSolver& self, NativeVCycle& preconditioner, f64_cuda_2d B, f64_cuda_2d X,
               double tolerance, int max_iters, uintptr_t stream,
               std::optional<f64_cuda_2d> X0) {
                const int k = static_cast<int>(B.shape(1));
                const int n = self.n();
                const bool shaped = static_cast<int>(B.shape(0)) == n &&
                                    static_cast<int>(X.shape(0)) == n &&
                                    static_cast<int>(X.shape(1)) == k &&
                                    (!X0.has_value() || (static_cast<int>(X0->shape(0)) == n &&
                                                         static_cast<int>(X0->shape(1)) == k));
                if (!shaped) {
                    throw std::invalid_argument("B, X and X0 must all be (n, k)");
                }
                PcgBlockResult result;
                {
                    nb::gil_scoped_release release;
                    result = self.solve_mixed_block(preconditioner, B.data(), X.data(), k,
                                                    tolerance, max_iters,
                                                    reinterpret_cast<cudaStream_t>(stream),
                                                    X0.has_value() ? X0->data() : nullptr);
                }
                nb::list rels;
                for (double r : result.relative_residual) rels.append(r);
                nb::list col_iters;
                for (int i : result.column_iterations) col_iters.append(i);
                return nb::make_tuple(result.iterations, rels, col_iters);
            },
            nb::arg("preconditioner"), nb::arg("B").noconvert(), nb::arg("X").noconvert(),
            nb::arg("tolerance"), nb::arg("max_iters"), nb::arg("stream"),
            nb::arg("X0").noconvert() = nb::none(),
            "Lockstep k-RHS mixed-precision PCG over row-major (n, k) operands; returns "
            "(block iterations, per-column relative residuals, per-column iteration counts). A "
            "column freezes once its own residual meets tolerance, so it stops where it would "
            "have solved alone. k in {1, 2, 4, 8}.")
        .def(
            "set_affine_terms",
            [](PcgAmgSolver& self, f64_cuda_2d terms, uintptr_t stream) {
                if (static_cast<int>(terms.shape(1)) != self.nnz()) {
                    throw std::invalid_argument("affine terms must be (n_terms, nnz)");
                }
                self.set_affine_terms(static_cast<int>(terms.shape(0)), terms.data(),
                                      reinterpret_cast<cudaStream_t>(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("terms").noconvert(), nb::arg("stream") = 0,
            "Set an operator affine in per-column coefficients, A(sigma) = sum_t sigma_t A_t, "
            "from its parts' values in entry order (n_terms, nnz).")
        .def(
            "solve_affine_block",
            [](PcgAmgSolver& self, NativeVCycle& preconditioner, f64_cuda_2d B, f64_cuda_2d X,
               nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> sigma,
               double tolerance, int max_iters, uintptr_t stream,
               std::optional<f64_cuda_2d> X0) {
                const int k = static_cast<int>(B.shape(1));
                const int n = self.n();
                const bool shaped = static_cast<int>(B.shape(0)) == n &&
                                    static_cast<int>(X.shape(0)) == n &&
                                    static_cast<int>(X.shape(1)) == k &&
                                    static_cast<int>(sigma.shape(0)) == k &&
                                    static_cast<int>(sigma.shape(1)) == self.affine_terms() &&
                                    (!X0.has_value() || (static_cast<int>(X0->shape(0)) == n &&
                                                         static_cast<int>(X0->shape(1)) == k));
                if (!shaped) {
                    throw std::invalid_argument(
                        "B, X and X0 must be (n, k) and sigma (k, n_terms)");
                }
                PcgBlockResult result;
                {
                    nb::gil_scoped_release release;
                    result = self.solve_affine_block(preconditioner, B.data(), X.data(), k,
                                                     sigma.data(), tolerance, max_iters,
                                                     reinterpret_cast<cudaStream_t>(stream),
                                                     X0.has_value() ? X0->data() : nullptr);
                }
                nb::list rels;
                for (double r : result.relative_residual) rels.append(r);
                nb::list col_iters;
                for (int i : result.column_iterations) col_iters.append(i);
                return nb::make_tuple(result.iterations, rels, col_iters);
            },
            nb::arg("preconditioner"), nb::arg("B").noconvert(), nb::arg("X").noconvert(),
            nb::arg("sigma").noconvert(), nb::arg("tolerance"), nb::arg("max_iters"),
            nb::arg("stream"), nb::arg("X0").noconvert() = nb::none(),
            "The block solve with column c's operator A(sigma[c]) from set_affine_terms: every "
            "column its own matrix, sharing the parts' reads and the preconditioner. sigma is a "
            "host (k, n_terms) float64 array. Returns what solve_mixed_block does.");

    m.def(
        "p1_gradients",
        [](f64_cuda_2d nodes_mm, i32_cuda_2d tet_nodes, f64_cuda_3d g, f64_cuda vols,
           uintptr_t stream) {
            const int n_tet = static_cast<int>(tet_nodes.shape(0));
            if (static_cast<int>(g.shape(0)) != n_tet || g.shape(1) != 4 || g.shape(2) != 3) {
                throw std::invalid_argument("p1_gradients: g must be (n_tet, 4, 3)");
            }
            if (static_cast<int>(vols.shape(0)) != n_tet) {
                throw std::invalid_argument("p1_gradients: vols must have one entry per tet");
            }
            launch_p1_gradients(nodes_mm.data(), tet_nodes.data(), g.data(), vols.data(), n_tet,
                                reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("nodes_mm").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("g").noconvert(), nb::arg("vols").noconvert(), nb::arg("stream"),
        "P1 basis-function gradients (n_tet, 4, 3) in 1/m and element volumes in m^3, from node "
        "coordinates in millimetres; writes into caller-allocated g/vols.");

    m.def(
        "dadt_nbody",
        [](f32_cuda_2d s, f32_cuda_2d mp, f32_cuda_1d sn, f32_cuda_2d r, f32_cuda_2d out,
           float didt, float mu0_4pi, uintptr_t stream, int column) {
            int n_dip = static_cast<int>(s.shape(0));
            int n_nodes = static_cast<int>(r.shape(0));
            const int stride = static_cast<int>(out.shape(1));
            if (static_cast<int>(out.shape(0)) != n_nodes || stride % 3 != 0 || column < 0 ||
                3 * column >= stride) {
                throw std::invalid_argument(
                    "dadt_nbody: out must be (n_nodes, 3k) with column < k");
            }
            launch_dadt(s.data(), mp.data(), sn.data(), r.data(), out.data() + 3 * column, stride,
                        n_dip, n_nodes, didt, mu0_4pi, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("s").noconvert(), nb::arg("mp").noconvert(), nb::arg("sn").noconvert(),
        nb::arg("r").noconvert(), nb::arg("out").noconvert(), nb::arg("didt"), nb::arg("mu0_4pi"),
        nb::arg("stream"), nb::arg("column") = 0,
        "Fused dA/dt at nodes from placed magnetic dipoles; writes into caller-allocated out. out "
        "is (n_nodes, 3k), an interleaved (n_nodes, k, 3) block, and this fills placement column.");

    m.def(
        "dadt_node_to_element_block",
        [](f32_cuda_3d dadt_nodes, i32_cuda_2d tet_nodes, std::vector<f32_cuda_2d> out,
           uintptr_t stream) {
            const int k = static_cast<int>(out.size());
            if (k < 1 || k > kMaxStageBlock || static_cast<int>(dadt_nodes.shape(1)) != k ||
                dadt_nodes.shape(2) != 3) {
                throw std::invalid_argument("dadt_node_to_element_block: bad block shape");
            }
            const int n_tet = static_cast<int>(tet_nodes.shape(0));
            float* out_ptrs[kMaxStageBlock];
            for (int c = 0; c < k; ++c) {
                if (static_cast<int>(out[c].shape(0)) != n_tet || out[c].shape(1) != 3) {
                    throw std::invalid_argument("dadt_node_to_element_block: bad output shape");
                }
                out_ptrs[c] = out[c].data();
            }
            launch_dadt_element_average_block(dadt_nodes.data(), tet_nodes.data(), out_ptrs,
                                              n_tet, k, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("dadt_nodes").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("out").noconvert(), nb::arg("stream"),
        "Average an interleaved (n_nodes, k, 3) nodal dA/dt onto tetrahedra, one (n_tet, 3) "
        "output per placement; tet_nodes is read once for the block.");

    m.def(
        "rhs_operator_fill",
        [](f64_cuda_3d g, f64_cuda scale, i32_cuda_2d tet_nodes, i32_cuda ptr, i32_cuda idx,
           i32_cuda indptr, i32_cuda indices, f32_cuda_2d op, uintptr_t stream) {
            const int n_rows = static_cast<int>(indptr.shape(0)) - 1;
            if (op.shape(0) != indices.shape(0) || op.shape(1) != 4 ||
                static_cast<int>(ptr.shape(0)) != n_rows + 1) {
                throw std::invalid_argument("rhs_operator_fill: shape mismatch");
            }
            launch_rhs_operator_fill(g.data(), scale.data(), tet_nodes.data(), ptr.data(),
                                     idx.data(), indptr.data(), indices.data(), op.data(), n_rows,
                                     reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("g").noconvert(), nb::arg("scale").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(), nb::arg("indptr").noconvert(),
        nb::arg("indices").noconvert(), nb::arg("op").noconvert(), nb::arg("stream"),
        "Build the forward RHS operator over the stiffness pattern: op is (nnz, 4), each row a "
        "packed (Bx, By, Bz, column); scale is neg_vc / 4 in float64.");

    m.def(
        "rhs_operator_apply",
        [](i32_cuda indptr, f32_cuda_2d op, f32_cuda_3d a, f32_cuda_2d b, uintptr_t stream) {
            const int n_nodes = static_cast<int>(indptr.shape(0)) - 1;
            const int k = static_cast<int>(a.shape(1));
            if (k < 1 || k > kMaxStageBlock || static_cast<int>(a.shape(0)) != n_nodes ||
                a.shape(2) != 3 || static_cast<int>(b.shape(0)) != n_nodes ||
                static_cast<int>(b.shape(1)) != k || op.shape(1) != 4) {
                throw std::invalid_argument("rhs_operator_apply: shape mismatch");
            }
            launch_rhs_operator(indptr.data(), op.data(), a.data(), b.data(), n_nodes, k,
                                reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("indptr").noconvert(), nb::arg("op").noconvert(), nb::arg("a").noconvert(),
        nb::arg("b").noconvert(), nb::arg("stream"),
        "b = B a for an interleaved (n_nodes, k, 3) nodal dA/dt; writes row-major (n_nodes, k) "
        "float32. Each column's sum order is independent of k.");

    m.def(
        "dadt_node_to_element",
        [](f32_cuda_2d dadt_nodes, i32_cuda_2d tet_nodes, f32_cuda_2d out, uintptr_t stream) {
            int n_tet = static_cast<int>(tet_nodes.shape(0));
            launch_dadt_element_average(dadt_nodes.data(), tet_nodes.data(), out.data(), n_tet,
                                        reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("dadt_nodes").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("out").noconvert(), nb::arg("stream"),
        "Average nodal dA/dt onto tetrahedra; writes into caller-allocated out.");

    m.def(
        "rhs_assemble",
        [](f32_cuda_2d dadt_elm, f32_cuda_3d g, f32_cuda_1d neg_vc, i32_cuda ptr, i32_cuda idx,
           f32_cuda_1d b, uintptr_t stream) {
            int n_nodes = static_cast<int>(b.shape(0));
            launch_rhs(dadt_elm.data(), g.data(), neg_vc.data(), ptr.data(), idx.data(), b.data(),
                       n_nodes, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("dadt_elm").noconvert(), nb::arg("g").noconvert(), nb::arg("neg_vc").noconvert(),
        nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(), nb::arg("b").noconvert(),
        nb::arg("stream"),
        "Deterministic node-centric RHS assembly; writes into caller-allocated b.");

    m.def(
        "rhs_assemble_staged",
        [](f32_cuda_2d dadt_elm, f32_cuda_3d g, f32_cuda_1d neg_vc, i32_cuda ptr, i32_cuda idx,
           f32_cuda_1d b, uintptr_t stream) {
            int n_nodes = static_cast<int>(b.shape(0));
            int n_tet = static_cast<int>(dadt_elm.shape(0));
            launch_rhs_staged(dadt_elm.data(), g.data(), neg_vc.data(), ptr.data(), idx.data(),
                              b.data(), n_nodes, n_tet, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("dadt_elm").noconvert(), nb::arg("g").noconvert(), nb::arg("neg_vc").noconvert(),
        nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(), nb::arg("b").noconvert(),
        nb::arg("stream"),
        "Deterministic node-centric RHS assembly, corner pass then gather.");

    m.def(
        "tet_lowest_node",
        [](i32_cuda inverse, i32_cuda_2d tet_nodes, i32_cuda lowest, uintptr_t stream) {
            const int n_tet = static_cast<int>(tet_nodes.shape(0));
            if (tet_nodes.shape(1) != 4 || static_cast<int>(lowest.shape(0)) != n_tet) {
                throw std::invalid_argument(
                    "tet_lowest_node: tet_nodes must be (n_tet, 4) and lowest (n_tet,)");
            }
            launch_tet_lowest_node(inverse.data(), tet_nodes.data(), lowest.data(), n_tet,
                                   reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("inverse").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("lowest").noconvert(), nb::arg("stream"),
        "Smallest permuted node index of each tetrahedron, the sort key of the Morton layout.");

    m.def(
        "count_incident_node_csr",
        [](i32_cuda_2d tet_nodes, i32_cuda ptr, i32_cuda idx, i32_cuda out_ptr,
           uintptr_t stream) {
            if (ptr.shape(0) != out_ptr.shape(0)) {
                throw std::invalid_argument(
                    "count_incident_node_csr: ptr and out_ptr must both be n_seg + 1");
            }
            return count_incident_node_csr(tet_nodes.data(), ptr.data(), idx.data(),
                                           out_ptr.data(),
                                           static_cast<int>(out_ptr.shape(0)) - 1,
                                           reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("tet_nodes").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("out_ptr").noconvert(), nb::arg("stream"),
        "Row offsets of the distinct nodes of the tetrahedra in each segment of a corner CSR. "
        "Fills out_ptr and returns nnz, the length to allocate the column index at. "
        "Synchronises the stream.");

    m.def(
        "fill_incident_node_csr",
        [](i32_cuda_2d tet_nodes, i32_cuda ptr, i32_cuda idx, i32_cuda out_ptr, i32_cuda out_idx,
           uintptr_t stream) {
            if (ptr.shape(0) != out_ptr.shape(0)) {
                throw std::invalid_argument(
                    "fill_incident_node_csr: ptr and out_ptr must both be n_seg + 1");
            }
            fill_incident_node_csr(tet_nodes.data(), ptr.data(), idx.data(), out_ptr.data(),
                                   out_idx.data(), static_cast<int>(out_ptr.shape(0)) - 1,
                                   reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("tet_nodes").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("out_ptr").noconvert(), nb::arg("out_idx").noconvert(), nb::arg("stream"),
        "Fill the column index for the rows count_incident_node_csr sized.");

    m.def(
        "count_patch_csr",
        [](i32_cuda r1_ptr, i32_cuda r1_idx, i32_cuda neighbour, int min_nodes, i32_cuda out_ptr,
           uintptr_t stream) {
            if (neighbour.shape(0) != r1_idx.shape(0)) {
                throw std::invalid_argument(
                    "count_patch_csr: neighbour must have one entry per first-ring node");
            }
            if (out_ptr.shape(0) != r1_ptr.shape(0)) {
                throw std::invalid_argument(
                    "count_patch_csr: r1_ptr and out_ptr must both be n_slots + 1");
            }
            return count_patch_csr(r1_ptr.data(), r1_idx.data(), neighbour.data(), min_nodes,
                                   out_ptr.data(), static_cast<int>(r1_ptr.shape(0)) - 1,
                                   reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("r1_ptr").noconvert(), nb::arg("r1_idx").noconvert(),
        nb::arg("neighbour").noconvert(), nb::arg("min_nodes"), nb::arg("out_ptr").noconvert(),
        nb::arg("stream"),
        "Row offsets of the recovery patches: a slot reaching fewer than min_nodes grows to the "
        "union of its neighbours' rings. Same convention as count_incident_node_csr.");

    m.def(
        "fill_patch_csr",
        [](i32_cuda r1_ptr, i32_cuda r1_idx, i32_cuda neighbour, int min_nodes, i32_cuda out_ptr,
           i32_cuda out_idx, uintptr_t stream) {
            if (neighbour.shape(0) != r1_idx.shape(0)) {
                throw std::invalid_argument(
                    "fill_patch_csr: neighbour must have one entry per first-ring node");
            }
            if (out_ptr.shape(0) != r1_ptr.shape(0)) {
                throw std::invalid_argument(
                    "fill_patch_csr: r1_ptr and out_ptr must both be n_slots + 1");
            }
            fill_patch_csr(r1_ptr.data(), r1_idx.data(), neighbour.data(), min_nodes,
                           out_ptr.data(), out_idx.data(),
                           static_cast<int>(r1_ptr.shape(0)) - 1,
                           reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("r1_ptr").noconvert(), nb::arg("r1_idx").noconvert(),
        nb::arg("neighbour").noconvert(), nb::arg("min_nodes"), nb::arg("out_ptr").noconvert(),
        nb::arg("out_idx").noconvert(), nb::arg("stream"),
        "Fill the column index for the rows count_patch_csr sized.");

    m.def(
        "assemble_stiffness_values",
        [](f64_cuda_3d g, f64_cuda scale, i32_cuda_2d tet_nodes, i32_cuda ptr, i32_cuda idx,
           i32_cuda indptr, i32_cuda indices, f64_cuda data, uintptr_t stream) {
            const int n_rows = static_cast<int>(indptr.shape(0)) - 1;
            launch_stiffness_rows(g.data(), scale.data(), tet_nodes.data(), ptr.data(), idx.data(),
                                  indptr.data(), indices.data(), data.data(), n_rows,
                                  reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("g").noconvert(), nb::arg("scale").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(), nb::arg("indptr").noconvert(),
        nb::arg("indices").noconvert(), nb::arg("data").noconvert(), nb::arg("stream"),
        "Fill a CSR pattern's values with the stiffness matrix, one thread per row.");

    m.def(
        "l1_dinv",
        [](i32_cuda indptr, i32_cuda indices, f32_cuda_1d data, f32_cuda_1d dinv,
           uintptr_t stream) {
            const int n_rows = static_cast<int>(indptr.shape(0)) - 1;
            if (static_cast<int>(dinv.shape(0)) != n_rows) {
                throw std::invalid_argument("dinv must have one entry per row");
            }
            if (indices.shape(0) != data.shape(0)) {
                throw std::invalid_argument("indices and data must have one entry per nonzero");
            }
            launch_l1_dinv(indptr.data(), indices.data(), data.data(), dinv.data(), n_rows,
                           reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("indptr").noconvert(), nb::arg("indices").noconvert(), nb::arg("data").noconvert(),
        nb::arg("dinv").noconvert(), nb::arg("stream"),
        "l1-Jacobi smoother scaling 1 / (sign(a_ii) * sum_j |a_ij|), one thread per row.");

    m.def(
        "rhs_assemble_staged_block",
        [](std::vector<f32_cuda_2d> dadt_elm, f32_cuda_3d g, f32_cuda_1d neg_vc, i32_cuda ptr,
           i32_cuda idx, f32_cuda_2d b_block, uintptr_t stream) {
            const int k = static_cast<int>(dadt_elm.size());
            if (k < 1 || k > kMaxStageBlock ||
                static_cast<int>(b_block.shape(1)) != k) {
                throw std::invalid_argument("rhs_assemble_staged_block: bad list sizes");
            }
            const float* in_ptrs[kMaxStageBlock];
            for (int c = 0; c < k; ++c) in_ptrs[c] = dadt_elm[c].data();
            int n_nodes = static_cast<int>(b_block.shape(0));
            int n_tet = static_cast<int>(dadt_elm[0].shape(0));
            launch_rhs_staged_block(in_ptrs, g.data(), neg_vc.data(), ptr.data(), idx.data(),
                                    b_block.data(), n_nodes, n_tet, k,
                                    reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("dadt_elm").noconvert(), nb::arg("g").noconvert(), nb::arg("neg_vc").noconvert(),
        nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(), nb::arg("b_block").noconvert(),
        nb::arg("stream"),
        "Block RHS assembly for <=8 placements: the gradients and node2corner are read once for "
        "the whole block; writes row-major (n_nodes, k) float32.");

    m.def(
        "reconstruct_e_block",
        [](f64_cuda_2d v_block, i32_cuda_2d tet_nodes, f32_cuda_3d g,
           std::vector<f32_cuda_2d> dadt_elm, std::vector<f32_cuda_2d> e_out,
           std::vector<f32_cuda_1d> magn_out, uintptr_t stream) {
            const int k = static_cast<int>(dadt_elm.size());
            if (k < 1 || k > kMaxStageBlock || e_out.size() != dadt_elm.size() ||
                magn_out.size() != dadt_elm.size() ||
                static_cast<int>(v_block.shape(1)) != k) {
                throw std::invalid_argument("reconstruct_e_block: bad list sizes");
            }
            const float* de_ptrs[kMaxStageBlock];
            float* e_ptrs[kMaxStageBlock];
            float* m_ptrs[kMaxStageBlock];
            for (int c = 0; c < k; ++c) {
                de_ptrs[c] = dadt_elm[c].data();
                e_ptrs[c] = e_out[c].data();
                m_ptrs[c] = magn_out[c].data();
            }
            int n_tet = static_cast<int>(tet_nodes.shape(0));
            launch_reconstruct_block(v_block.data(), tet_nodes.data(), g.data(), de_ptrs,
                                     e_ptrs, m_ptrs, n_tet, k,
                                     reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("v_block").noconvert(), nb::arg("tet_nodes").noconvert(), nb::arg("g").noconvert(),
        nb::arg("dadt_elm").noconvert(), nb::arg("e_out").noconvert(),
        nb::arg("magn_out").noconvert(), nb::arg("stream"),
        "Block E/magnE reconstruction for <=8 placements: tet_nodes/g read once; "
        "v_block is row-major (n_nodes, k) float64.");

    m.def(
        "reconstruct_e",
        [](f64_cuda v, i32_cuda_2d tet_nodes, f32_cuda_3d g, f32_cuda_2d dadt_elm,
           f32_cuda_2d e_out, f32_cuda_1d magn_out, uintptr_t stream) {
            int n_tet = static_cast<int>(tet_nodes.shape(0));
            launch_reconstruct(v.data(), tet_nodes.data(), g.data(), dadt_elm.data(), e_out.data(),
                               magn_out.data(), n_tet, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("v").noconvert(), nb::arg("tet_nodes").noconvert(), nb::arg("g").noconvert(),
        nb::arg("dadt_elm").noconvert(), nb::arg("e_out").noconvert(),
        nb::arg("magn_out").noconvert(), nb::arg("stream"),
        "Element-centric E/magnE reconstruction; writes into caller-allocated e_out/magn_out.");

    m.def(
        "mark_outer_boundary",
        [](i32_cuda_2d tet_nodes, i32_cuda ptr, i32_cuda idx, i32_cuda is_boundary,
           uintptr_t stream) {
            const int n_tet = static_cast<int>(tet_nodes.shape(0));
            if (idx.shape(0) != static_cast<size_t>(n_tet) * 4) {
                throw std::invalid_argument(
                    "mark_outer_boundary: idx must hold one entry per tet corner");
            }
            if (ptr.shape(0) != is_boundary.shape(0) + 1) {
                throw std::invalid_argument(
                    "mark_outer_boundary: ptr must hold n_nodes + 1 entries");
            }
            launch_mark_outer_boundary(tet_nodes.data(), ptr.data(), idx.data(),
                                       is_boundary.data(), n_tet,
                                       reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("tet_nodes").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("is_boundary").noconvert(), nb::arg("stream"),
        "Mark the nodes of every face owned by exactly one tet, counted over the node2corner "
        "incidence CSR. is_boundary is only ever set, so it arrives zeroed.");

    m.def(
        "spr_weights",
        [](f64_cuda_2d nodes_mm, i32_cuda_2d tet_nodes, f32_cuda_1d vols, i32_cuda ptr,
           i32_cuda idx, std::optional<i32_cuda> slot_node, std::optional<i32_cuda> is_boundary,
           f32_cuda_1d w, std::optional<i32_cuda> n_fallback, uintptr_t stream) {
            const int n_slots = static_cast<int>(ptr.shape(0)) - 1;
            if (w.shape(0) != idx.shape(0)) {
                throw std::invalid_argument("spr_weights: w must have one entry per corner");
            }
            if (slot_node && static_cast<int>(slot_node->shape(0)) != n_slots) {
                throw std::invalid_argument("spr_weights: slot_node must have one entry per slot");
            }
            launch_spr_weights(nodes_mm.data(), tet_nodes.data(), vols.data(), ptr.data(),
                               idx.data(), slot_node ? slot_node->data() : nullptr,
                               is_boundary ? is_boundary->data() : nullptr, w.data(),
                               n_fallback ? n_fallback->data() : nullptr, n_slots,
                               reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("nodes_mm").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("vols").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("slot_node").noconvert(), nb::arg("is_boundary").noconvert(),
        nb::arg("w").noconvert(), nb::arg("n_fallback").noconvert(), nb::arg("stream"),
        "One scalar patch weight per corner. Pass slot_node=None when a slot is a node, "
        "is_boundary=None to fit every slot, n_fallback=None to skip the counter.");

    m.def(
        "skin_normals",
        [](f64_cuda_2d nodes_mm, i32_cuda_2d tris, i32_cuda node_ptr, i32_cuda node_idx,
           f64_cuda_2d out, uintptr_t stream) {
            const int n_tri = static_cast<int>(tris.shape(0));
            const int n_nodes = static_cast<int>(node_ptr.shape(0)) - 1;
            if (tris.shape(1) != 3 || out.shape(1) != 3 ||
                static_cast<int>(out.shape(0)) != n_tri) {
                throw std::invalid_argument("skin_normals: tris and out must be (n_tri, 3)");
            }
            if (static_cast<int>(node_idx.shape(0)) != 3 * n_tri) {
                throw std::invalid_argument("skin_normals: node_idx must hold every corner");
            }
            launch_skin_normals(nodes_mm.data(), tris.data(), node_ptr.data(), node_idx.data(),
                                out.data(), n_tri, n_nodes,
                                reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("nodes_mm").noconvert(), nb::arg("tris").noconvert(),
        nb::arg("node_ptr").noconvert(), nb::arg("node_idx").noconvert(),
        nb::arg("out").noconvert(), nb::arg("stream"),
        "Smoothed outward skin-triangle normals. node_ptr/node_idx is a node-to-corner CSR over "
        "the triangles, and its order is the order every node's smoothing sums in.");

    m.def(
        "ray_first_hit",
        [](f64_cuda_2d origins, f64_cuda_2d directions, f64_cuda_3d verts, f64_cuda out_t,
           double min_t, uintptr_t stream) {
            const int n_rays = static_cast<int>(origins.shape(0));
            if (origins.shape(1) != 3 || directions.shape(0) != origins.shape(0) ||
                directions.shape(1) != 3) {
                throw std::invalid_argument(
                    "ray_first_hit: origins and directions must both be (n_rays, 3)");
            }
            if (verts.shape(1) != 3 || verts.shape(2) != 3) {
                throw std::invalid_argument("ray_first_hit: verts must be (n_tri, 3, 3)");
            }
            if (static_cast<int>(out_t.shape(0)) != n_rays) {
                throw std::invalid_argument("ray_first_hit: out_t must have one entry per ray");
            }
            launch_ray_first_hit(origins.data(), directions.data(), verts.data(), out_t.data(),
                                 n_rays, static_cast<int>(verts.shape(0)), min_t,
                                 reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("origins").noconvert(), nb::arg("directions").noconvert(),
        nb::arg("verts").noconvert(), nb::arg("out_t").noconvert(), nb::arg("min_t"),
        nb::arg("stream"),
        "Moller-Trumbore first-hit distance per ray against a triangle surface; +inf where the "
        "ray misses. Hits nearer than min_t are skipped, so a ray leaving one surface does not "
        "hit the one it started on.");

    m.def(
        "hpr_weights",
        [](f64_cuda_2d nodes_mm, i32_cuda pptr, i32_cuda pidx, i32_cuda slot_node, f32_cuda_2d w,
           i32_cuda status, uintptr_t stream, std::optional<i32_cuda> active) {
            const int n_slots = static_cast<int>(pptr.shape(0)) - 1;
            if (w.shape(0) != pidx.shape(0) || w.shape(1) != 3) {
                throw std::invalid_argument("hpr_weights: w must be (nnz, 3)");
            }
            if (static_cast<int>(slot_node.shape(0)) != n_slots ||
                static_cast<int>(status.shape(0)) != n_slots) {
                throw std::invalid_argument("hpr_weights: slot_node/status must be per slot");
            }
            launch_hpr_weights(nodes_mm.data(), pptr.data(), pidx.data(), slot_node.data(),
                               w.data(), status.data(),
                               active.has_value() ? active->data() : nullptr,
                               active.has_value() ? static_cast<int>(active->shape(0)) : n_slots,
                               reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("nodes_mm").noconvert(), nb::arg("pptr").noconvert(), nb::arg("pidx").noconvert(),
        nb::arg("slot_node").noconvert(), nb::arg("w").noconvert(), nb::arg("status").noconvert(),
        nb::arg("stream"), nb::arg("active").noconvert() = nb::none(),
        "Harmonic-constrained potential-recovery weights: grad_v[s] = sum_m w[s,m] v[m]. "
        "status is the rung the fit took, counted down from the widest basis: 0 harmonic cubic, "
        "1 harmonic quadratic, 2 linear, 3 no gradient determined.");

    m.def(
        "hpr_grad",
        [](f64_cuda_2d v_block, f32_cuda_2d w, i32_cuda pptr, i32_cuda pidx, i32_cuda slot_node,
           std::vector<f32_cuda_2d> dadt_nodes, std::vector<f32_cuda_2d> e_slots,
           uintptr_t stream, std::optional<i32_cuda> active) {
            const int k = static_cast<int>(dadt_nodes.size());
            if (k < 1 || k > kMaxStageBlock || e_slots.size() != dadt_nodes.size()) {
                throw std::invalid_argument("hpr_grad: bad list sizes");
            }
            const int stride = static_cast<int>(v_block.shape(1));
            if (stride < k) {
                throw std::invalid_argument("hpr_grad: v_block has fewer columns than placements");
            }
            const float* da_ptrs[kMaxStageBlock];
            float* out_ptrs[kMaxStageBlock];
            for (int c = 0; c < k; ++c) {
                da_ptrs[c] = dadt_nodes[c].data();
                out_ptrs[c] = e_slots[c].data();
            }
            const int n_slots = static_cast<int>(pptr.shape(0)) - 1;
            launch_hpr_grad(v_block.data(), w.data(), pptr.data(), pidx.data(), slot_node.data(),
                            da_ptrs, out_ptrs, n_slots,
                            active.has_value() ? active->data() : nullptr,
                            active.has_value() ? static_cast<int>(active->shape(0)) : n_slots, k,
                            stride, static_cast<int>(v_block.shape(0)),
                            reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("v_block").noconvert(), nb::arg("w").noconvert(), nb::arg("pptr").noconvert(),
        nb::arg("pidx").noconvert(), nb::arg("slot_node").noconvert(),
        nb::arg("dadt_nodes").noconvert(), nb::arg("e_slots").noconvert(), nb::arg("stream"),
        nb::arg("active").noconvert() = nb::none(),
        "Recovered E on slots from the nodal potential: E = -grad_v - dA/dt, with dA/dt taken "
        "exactly at the node rather than averaged over an element.");

    m.def(
        "recover_nodes",
        [](std::vector<f32_cuda_2d> e_in, f32_cuda_1d w, i32_cuda ptr, i32_cuda idx,
           std::vector<f32_cuda_2d> e_slots, uintptr_t stream) {
            const int k = static_cast<int>(e_in.size());
            if (k < 1 || k > kMaxStageBlock || e_slots.size() != e_in.size()) {
                throw std::invalid_argument("recover_nodes: bad list sizes");
            }
            const float* in_ptrs[kMaxStageBlock];
            float* out_ptrs[kMaxStageBlock];
            for (int c = 0; c < k; ++c) {
                in_ptrs[c] = e_in[c].data();
                out_ptrs[c] = e_slots[c].data();
            }
            const int n_slots = static_cast<int>(ptr.shape(0)) - 1;
            launch_recover_nodes(in_ptrs, w.data(), ptr.data(), idx.data(), out_ptrs, n_slots, k,
                                 reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("e_in").noconvert(), nb::arg("w").noconvert(), nb::arg("ptr").noconvert(),
        nb::arg("idx").noconvert(), nb::arg("e_slots").noconvert(), nb::arg("stream"),
        "Patch gather E*[s] = sum_c w[c] E[c>>2] for <=8 placements; the operator is read once "
        "for the whole block and each column matches its serial counterpart bitwise.");

    m.def(
        "recover_elements",
        [](std::vector<f32_cuda_2d> e_slots, i32_cuda slot_of_corner,
           std::vector<f32_cuda_2d> e_out, std::vector<f32_cuda_1d> magn_out, uintptr_t stream,
           std::optional<i32_cuda> active) {
            const int k = static_cast<int>(e_slots.size());
            if (k < 1 || k > kMaxStageBlock || e_out.size() != e_slots.size() ||
                magn_out.size() != e_slots.size()) {
                throw std::invalid_argument("recover_elements: bad list sizes");
            }
            const float* in_ptrs[kMaxStageBlock];
            float* e_ptrs[kMaxStageBlock];
            float* m_ptrs[kMaxStageBlock];
            for (int c = 0; c < k; ++c) {
                in_ptrs[c] = e_slots[c].data();
                e_ptrs[c] = e_out[c].data();
                m_ptrs[c] = magn_out[c].data();
            }
            const int n_tet = static_cast<int>(slot_of_corner.shape(0)) / 4;
            launch_recover_elements(in_ptrs, slot_of_corner.data(), e_ptrs, m_ptrs, n_tet,
                                    active.has_value() ? active->data() : nullptr,
                                    active.has_value() ? static_cast<int>(active->shape(0)) : n_tet,
                                    k, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("e_slots").noconvert(), nb::arg("slot_of_corner").noconvert(),
        nb::arg("e_out").noconvert(), nb::arg("magn_out").noconvert(), nb::arg("stream"),
        nb::arg("active").noconvert() = nb::none(),
        "Sample the recovered field back at the barycentres and take its magnitude.");

    m.def(
        "element_weight",
        [](f64_cuda values, i32_cuda_2d tet_nodes, f32_cuda_3d g, f32_cuda_1d neg_vc,
           f64_cuda_2d w_e, uintptr_t stream) {
            int n_tet = static_cast<int>(tet_nodes.shape(0));
            launch_element_weight(values.data(), tet_nodes.data(), g.data(), neg_vc.data(),
                                  w_e.data(), n_tet, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("values").noconvert(), nb::arg("tet_nodes").noconvert(), nb::arg("g").noconvert(),
        nb::arg("neg_vc").noconvert(), nb::arg("w_e").noconvert(), nb::arg("stream"),
        "Per-element reciprocity weight w_e = vol*sigma*(G_e values); into caller-allocated w_e.");

    m.def(
        "node_gather",
        [](f64_cuda corner, i32_cuda ptr, i32_cuda idx, f64_cuda out, uintptr_t stream) {
            int n_nodes = static_cast<int>(ptr.shape(0)) - 1;
            if (static_cast<int>(out.shape(0)) != n_nodes) {
                throw std::invalid_argument("node_gather: out must have one entry per node");
            }
            launch_node_gather(corner.data(), ptr.data(), idx.data(), out.data(), n_nodes,
                               reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("corner").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("out").noconvert(), nb::arg("stream"),
        "Node-centric scalar corner gather over the node2corner CSR; into caller-allocated out.");

    m.def(
        "node_scatter3",
        [](f64_cuda_2d w_e, i32_cuda ptr, i32_cuda idx, f64_cuda_2d node_w, uintptr_t stream) {
            int n_nodes = static_cast<int>(ptr.shape(0)) - 1;
            launch_node_scatter3(w_e.data(), ptr.data(), idx.data(), node_w.data(), n_nodes,
                                 reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("w_e").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("node_w").noconvert(), nb::arg("stream"),
        "Node-centric 3-vector corner gather with 1/4 weight; into caller-allocated node_w.");

    m.def(
        "dadt_nbody_multi",
        [](f32_cuda_2d s, f32_cuda_3d mp, f32_cuda_1d sn, f32_cuda_2d r, f32_cuda_3d out,
           float didt, float mu0_4pi, uintptr_t stream) {
            const int n_sets = static_cast<int>(mp.shape(0));
            const int n_dip = static_cast<int>(s.shape(0));
            const int n_nodes = static_cast<int>(r.shape(0));
            if (static_cast<int>(mp.shape(1)) != n_dip || mp.shape(2) != 6 ||
                static_cast<int>(out.shape(0)) != n_nodes ||
                static_cast<int>(out.shape(1)) != n_sets || out.shape(2) != 3) {
                throw std::invalid_argument("dadt_nbody_multi: shape mismatch");
            }
            launch_dadt_multi(s.data(), mp.data(), sn.data(), r.data(), out.data(), n_sets,
                              n_dip, n_nodes, didt, mu0_4pi,
                              reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("s").noconvert(), nb::arg("mp").noconvert(), nb::arg("sn").noconvert(),
        nb::arg("r").noconvert(), nb::arg("out").noconvert(), nb::arg("didt"), nb::arg("mu0_4pi"),
        nb::arg("stream"),
        "The dA/dt N-body for 1 or 3 moment sets over shared positions: mp is (n_sets, n_dip, 6) "
        "and out (n_targets, n_sets, 3).");

    m.def(
        "adm_evaluate",
        [](f64_cuda_3d frames, f32_cuda_2d coil_pos, f32_cuda_2d coil_mom, f32_cuda_1d q,
           u8_cuda_1d cell_ok, std::vector<double> origin, std::vector<double> spacing,
           std::vector<int> shape, int n_dir, double didt, f64_cuda_2d out, i32_cuda fault,
           uintptr_t stream) {
            const GridArgs g = grid_args(origin, spacing, shape);
            const int n_frames = static_cast<int>(frames.shape(0));
            const int n_dip = static_cast<int>(coil_pos.shape(0));
            const std::int64_t n_pts = static_cast<std::int64_t>(shape[0]) * shape[1] * shape[2];
            const std::int64_t n_cells =
                static_cast<std::int64_t>(shape[0] - 1) * (shape[1] - 1) * (shape[2] - 1);
            if (frames.shape(1) != 4 || frames.shape(2) != 4 || coil_pos.shape(1) != 3 ||
                coil_mom.shape(0) != coil_pos.shape(0) || coil_mom.shape(1) != 3 ||
                static_cast<std::int64_t>(q.shape(0)) != n_pts * n_dir * 3 ||
                static_cast<std::int64_t>(cell_ok.shape(0)) != n_cells ||
                static_cast<int>(out.shape(0)) != n_frames ||
                static_cast<int>(out.shape(1)) != n_dir || fault.shape(0) != 2) {
                throw std::invalid_argument("adm_evaluate: shape mismatch");
            }
            launch_adm_evaluate(frames.data(), n_frames, coil_pos.data(), coil_mom.data(), n_dip,
                                q.data(), cell_ok.data(), g.origin, g.spacing, g.shape, n_dir,
                                didt, out.data(), fault.data(),
                                reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("frames").noconvert(), nb::arg("coil_pos").noconvert(),
        nb::arg("coil_mom").noconvert(), nb::arg("q").noconvert(), nb::arg("cell_ok").noconvert(),
        nb::arg("origin"), nb::arg("spacing"), nb::arg("shape"), nb::arg("n_dir"),
        nb::arg("didt"), nb::arg("out").noconvert(), nb::arg("fault").noconvert(),
        nb::arg("stream"),
        "Target E-vectors (n_frames, n_dir) of coil frames against a sampled reciprocity field. "
        "fault (2 ints, zeroed by the caller) receives a fault kind and the first candidate.");

    m.def(
        "adm_mark",
        [](f64_cuda_3d frames, f32_cuda_2d coil_pos, std::vector<double> origin,
           std::vector<double> spacing, std::vector<int> shape, u8_cuda_1d need,
           uintptr_t stream) {
            const GridArgs g = grid_args(origin, spacing, shape);
            if (static_cast<std::int64_t>(need.shape(0)) !=
                static_cast<std::int64_t>(shape[0]) * shape[1] * shape[2]) {
                throw std::invalid_argument("adm_mark: need must have one byte per grid point");
            }
            launch_adm_mark(frames.data(), static_cast<int>(frames.shape(0)), coil_pos.data(),
                            static_cast<int>(coil_pos.shape(0)), g.origin, g.spacing, g.shape,
                            need.data(), reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("frames").noconvert(), nb::arg("coil_pos").noconvert(), nb::arg("origin"),
        nb::arg("spacing"), nb::arg("shape"), nb::arg("need").noconvert(), nb::arg("stream"),
        "Sets need to 1 at every grid point a frame's dipoles interpolate from.");

    m.def(
        "adm_reach",
        [](std::vector<double> origin, std::vector<double> spacing, std::vector<int> shape,
           f64_cuda_2d origins, f64_cuda_2d axes, u8_cuda_2d profile, double h0, double cell,
           u8_cuda_1d mask, uintptr_t stream) {
            const GridArgs g = grid_args(origin, spacing, shape);
            if (origins.shape(1) != 3 || axes.shape(1) != 3 ||
                origins.shape(0) != axes.shape(0) ||
                static_cast<std::int64_t>(mask.shape(0)) !=
                    static_cast<std::int64_t>(shape[0]) * shape[1] * shape[2]) {
                throw std::invalid_argument("adm_reach: shape mismatch");
            }
            launch_adm_reach(g.origin, g.spacing, g.shape, origins.data(), axes.data(),
                             static_cast<int>(origins.shape(0)), profile.data(),
                             static_cast<int>(profile.shape(0)), static_cast<int>(profile.shape(1)),
                             h0, cell, mask.data(), reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("origin"), nb::arg("spacing"), nb::arg("shape"), nb::arg("origins").noconvert(),
        nb::arg("axes").noconvert(), nb::arg("profile").noconvert(), nb::arg("h0"),
        nb::arg("cell"), nb::arg("mask").noconvert(), nb::arg("stream"),
        "Marks the grid points a coil's swept (rho, h) profile reaches about any centre's axis.");

    m.def(
        "host_copy_async",
        [](uintptr_t src, uintptr_t dst, std::size_t bytes, uintptr_t stream) {
            auto* copy = new HostCopy{reinterpret_cast<const void*>(src),
                                      reinterpret_cast<void*>(dst), bytes};
            const cudaError_t err = cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream),
                                                       run_host_copy, copy);
            if (err != cudaSuccess) delete copy;
            check_cuda(err, "host_copy_async", "launch host func");
        },
        nb::call_guard<nb::gil_scoped_release>(), nb::arg("src"), nb::arg("dst"),
        nb::arg("bytes"), nb::arg("stream"),
        "Copy bytes from src to dst (host pointers) once the stream reaches this point. The "
        "caller keeps both buffers alive until then.");

    m.attr("SUMMARY_LAYOUT") = nb::make_tuple(kSummaryPeak, kSummaryPeakLocation,
                                              kSummaryCenterOfGravity, kSummaryMean, kSummaryStd,
                                              kSummaryQuantiles);
    m.attr("MAX_SUMMARY_QUANTILES") = kMaxSummaryQuantiles;
    m.def(
        "region_summary_work_size",
        [](int n, int k, bool f64) { return region_summary_work_size(n, k, f64); },
        nb::call_guard<nb::gil_scoped_release>(), nb::arg("n"), nb::arg("k"), nb::arg("f64"),
        "Bytes of workspace region_summary needs.");
    m.def(
        "region_summary",
        [](std::vector<nb::ndarray<nb::ndim<1>, nb::c_contig, nb::device::cuda>> fields,
           i32_cuda index, f64_cuda w, f64_cuda_2d bary, std::vector<double> qs, int anchor,
           std::vector<double> fracs, u8_cuda_1d work, f64_cuda_2d out, uintptr_t stream) {
            const int k = static_cast<int>(fields.size());
            const int n = static_cast<int>(index.shape(0));
            if (k < 1 || k > kMaxStageBlock) {
                throw std::invalid_argument("region_summary: between 1 and 8 fields");
            }
            const bool f64 = fields[0].dtype() == nb::dtype<double>();
            const void* ptrs[kMaxStageBlock];
            for (int c = 0; c < k; ++c) {
                const bool ok = fields[c].dtype() == (f64 ? nb::dtype<double>()
                                                          : nb::dtype<float>()) &&
                                fields[c].shape(0) == fields[0].shape(0);
                if (!ok) {
                    throw std::invalid_argument(
                        "region_summary: fields must share one length and be float32 or float64");
                }
                ptrs[c] = fields[c].data();
            }
            const int n_q = static_cast<int>(qs.size());
            const int n_f = static_cast<int>(fracs.size());
            if (static_cast<int>(w.shape(0)) != n || static_cast<int>(bary.shape(0)) != n ||
                bary.shape(1) != 3 || static_cast<int>(out.shape(0)) != k ||
                static_cast<int>(out.shape(1)) != kSummaryQuantiles + n_q + n_f) {
                throw std::invalid_argument("region_summary: shape mismatch");
            }
            launch_region_summary(ptrs, f64, k, n, index.data(), w.data(), bary.data(),
                                  qs.data(), n_q, anchor, fracs.data(), n_f, work.data(),
                                  work.shape(0), out.data(),
                                  reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(), nb::arg("fields"),
        nb::arg("index").noconvert(), nb::arg("w").noconvert(), nb::arg("bary").noconvert(),
        nb::arg("qs"), nb::arg("anchor"), nb::arg("fracs"), nb::arg("work").noconvert(),
        nb::arg("out").noconvert(), nb::arg("stream"),
        "Volume-weighted metrics of up to 8 fields over one region; see fem/fem.hpp for the "
        "layout of out.");

    m.def(
        "accumulate_moments",
        [](f32_cuda_1d magn, f64_cuda sum_e, f64_cuda sumsq_e, uintptr_t stream) {
            int n = static_cast<int>(magn.shape(0));
            launch_accumulate_moments(magn.data(), sum_e.data(), sumsq_e.data(), n,
                                      reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("magn").noconvert(), nb::arg("sum_e").noconvert(), nb::arg("sumsq_e").noconvert(),
        nb::arg("stream"),
        "Fused streaming |E| moments: sum_e += magn; sumsq_e += magn^2 (in place).");

    m.def(
        "accumulate_moments_block",
        [](std::vector<f32_cuda_1d> magn, f64_cuda sum_e, f64_cuda sumsq_e, uintptr_t stream) {
            const int k = static_cast<int>(magn.size());
            const int n = static_cast<int>(sum_e.shape(0));
            if (k < 1 || k > kMaxStageBlock || sumsq_e.shape(0) != sum_e.shape(0)) {
                throw std::invalid_argument("accumulate_moments_block: bad sizes");
            }
            const float* ptrs[kMaxStageBlock];
            for (int c = 0; c < k; ++c) {
                if (static_cast<int>(magn[c].shape(0)) != n) {
                    throw std::invalid_argument("accumulate_moments_block: bad sizes");
                }
                ptrs[c] = magn[c].data();
            }
            launch_accumulate_moments_block(ptrs, k, sum_e.data(), sumsq_e.data(), n,
                                            reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("magn").noconvert(), nb::arg("sum_e").noconvert(), nb::arg("sumsq_e").noconvert(),
        nb::arg("stream"),
        "accumulate_moments for up to 8 draws at once, folded in the order given.");

    m.def(
        "row_dot",
        [](f64_cuda_2d a, f64_cuda_2d s, f64_cuda_2d out, uintptr_t stream) {
            const int n = static_cast<int>(a.shape(0));
            const int p = static_cast<int>(a.shape(1));
            const int k = static_cast<int>(s.shape(0));
            if (static_cast<int>(s.shape(1)) != p || static_cast<int>(out.shape(0)) != n ||
                static_cast<int>(out.shape(1)) != k) {
                throw std::invalid_argument("row_dot: a must be (n, p), s (k, p) and out (n, k)");
            }
            launch_row_dot(a.data(), s.data(), out.data(), n, p, k,
                           reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("a").noconvert(), nb::arg("s").noconvert(), nb::arg("out").noconvert(),
        nb::arg("stream"),
        "out = a @ s.T for a tall, narrow a, each entry summed in term order. A column's result "
        "does not depend on k.");

    m.def(
        "sweep_seed_work_size",
        [](int n, int m_basis, int k) { return sweep_seed_work_size(n, m_basis, k); },
        nb::call_guard<nb::gil_scoped_release>(), nb::arg("n"), nb::arg("m"), nb::arg("k"),
        "Doubles of scratch sweep_seed needs for an (n, m) basis and k columns.");

    m.def(
        "sweep_seed",
        [](f32_cuda_2d q, f64_cuda_2d b, f64_cuda work, f64_cuda_2d x0, uintptr_t stream) {
            const int n = static_cast<int>(q.shape(0));
            const int m_basis = static_cast<int>(q.shape(1));
            const int k = static_cast<int>(b.shape(1));
            if (static_cast<int>(b.shape(0)) != n || static_cast<int>(x0.shape(0)) != n ||
                static_cast<int>(x0.shape(1)) != k ||
                static_cast<std::int64_t>(work.shape(0)) < sweep_seed_work_size(n, m_basis, k)) {
                throw std::invalid_argument(
                    "sweep_seed: q (n, m), b and x0 (n, k), work sweep_seed_work_size(n, m, k)");
            }
            launch_sweep_seed(q.data(), b.data(), n, m_basis, k, work.data(), x0.data(),
                              reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("q").noconvert(), nb::arg("b").noconvert(), nb::arg("work").noconvert(),
        nb::arg("x0").noconvert(), nb::arg("stream"),
        "x0 = q (q^T b) per column for an A-orthonormal fp32 basis; a column's result does not "
        "depend on k.");

    m.def(
        "place_transforms",
        [](f64_cuda_2d centers, std::optional<f64_cuda_2d> handles, f64_cuda dists, f64_cuda_2d a,
           f64_cuda_2d b, f64_cuda_2d c, f64_cuda_2d tnorm, f64_cuda_2d out,
           std::optional<i32_cuda> degenerate, uintptr_t stream) {
            int n_pl = static_cast<int>(centers.shape(0));
            int n_tri = static_cast<int>(a.shape(0));
            launch_place(centers.data(), handles ? handles->data() : nullptr, dists.data(),
                         a.data(), b.data(), c.data(), tnorm.data(), out.data(),
                         degenerate ? degenerate->data() : nullptr, n_pl, n_tri,
                         reinterpret_cast<cudaStream_t>(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("centers").noconvert(), nb::arg("handles").noconvert(),
        nb::arg("dists").noconvert(), nb::arg("a").noconvert(), nb::arg("b").noconvert(),
        nb::arg("c").noconvert(), nb::arg("tnorm").noconvert(), nb::arg("out").noconvert(),
        nb::arg("degenerate").noconvert(), nb::arg("stream"),
        "Batched closest-point scalp projection + coil frame; writes (P,16) row-major 4x4 out "
        "and a (P,) flag marking placements whose handle left the in-plane axis undefined. "
        "Pass handles=None for the projection and normal alone, with degenerate=None.");
}
