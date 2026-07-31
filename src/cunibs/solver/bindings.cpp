#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <new>
#include <stdexcept>

#include "aggregate.hpp"
#include "kernels.hpp"
#include "solver.hpp"
#include "vcycle.hpp"

namespace nb = nanobind;

// Contiguity is required because these arrays are passed to CUDA as raw pointers.
using f64_cuda = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
using f64_cuda_2d = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cuda>;
using i32_cuda = nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cuda>;

using f32_cuda_1d = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
using f32_cuda_2d = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cuda>;
using f32_cuda_3d = nb::ndarray<float, nb::ndim<3>, nb::c_contig, nb::device::cuda>;
using i32_cuda_2d = nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cuda>;

NB_MODULE(_solver_ext, m) {
    // Compiled-in limits, exported so the Python layer never restates them.
    m.attr("MAX_STAGE_BLOCK") = kMaxStageBlock;
    nb::list widths;
    for (int w : kBlockWidths) widths.append(w);
    m.attr("BLOCK_SIZES") = nb::tuple(widths);

    nb::class_<NativeVCycle>(m, "NativeVCycle")
        .def(nb::init<>())
        .def(
            "add_level",
            [](NativeVCycle& self, i32_cuda row_ptr, i32_cuda col_idx, f32_cuda_1d values,
               f32_cuda_1d dinv, i32_cuda p_row_ptr, i32_cuda p_col_idx, f32_cuda_1d p_values,
               i32_cuda r_row_ptr, i32_cuda r_col_idx, f32_cuda_1d r_values) {
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
                               r_values.data());
            },
            nb::arg("row_ptr").noconvert(), nb::arg("col_idx").noconvert(),
            nb::arg("values").noconvert(), nb::arg("dinv").noconvert(),
            nb::arg("p_row_ptr").noconvert(), nb::arg("p_col_idx").noconvert(),
            nb::arg("p_values").noconvert(), nb::arg("r_row_ptr").noconvert(),
            nb::arg("r_col_idx").noconvert(), nb::arg("r_values").noconvert(),
            "Append one non-coarsest level (fp32 CSR, 1/guard(d), prolongator CSR and its "
            "transpose); device arrays are copied into solver-owned buffers.")
        .def(
            "set_coarse",
            [](NativeVCycle& self, f32_cuda_2d ainv) {
                if (ainv.shape(0) != ainv.shape(1)) {
                    throw std::invalid_argument("coarse inverse must be square");
                }
                self.set_coarse(static_cast<int>(ainv.shape(0)), ainv.data());
            },
            nb::arg("ainv").noconvert(),
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
            return select_size4(n_rows, nnz, row_ptr.data(), col_idx.data(), values.data(),
                                aggregates.data(), reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("row_ptr").noconvert(), nb::arg("col_idx").noconvert(),
        nb::arg("values").noconvert(), nb::arg("aggregates").noconvert(),
        nb::arg("stream") = 0,
        "Unsmoothed pairwise aggregation (AMGx SIZE_4). Fills `aggregates` with a surjective "
        "row -> aggregate map and returns the aggregate count. Synchronises the stream.");

    nb::class_<PcgAmgSolver>(m, "PcgAmgSolver")
        .def(
            "__init__",
            [](PcgAmgSolver* self, i32_cuda row_ptr, i32_cuda col_idx, f64_cuda values) {
                int n = static_cast<int>(row_ptr.shape(0)) - 1;
                int nnz = static_cast<int>(values.shape(0));
                new (self) PcgAmgSolver(n, nnz, row_ptr.data(), col_idx.data(), values.data());
            },
            nb::arg("row_ptr").noconvert(), nb::arg("col_idx").noconvert(),
            nb::arg("values").noconvert())
        .def(
            "update_values",
            [](PcgAmgSolver& self, f64_cuda values, uintptr_t stream) {
                if (static_cast<int>(values.shape(0)) != self.nnz()) {
                    throw std::invalid_argument("values must have one entry per nonzero");
                }
                self.update_values(values.data(), reinterpret_cast<cudaStream_t>(stream));
            },
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
                PcgResult result = self.solve_mixed(
                    preconditioner, b.data(), x.data(), tolerance, max_iters,
                    reinterpret_cast<cudaStream_t>(stream), x0.has_value() ? x0->data() : nullptr);
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
                PcgBlockResult result = self.solve_mixed_block(
                    preconditioner, B.data(), X.data(), k, tolerance, max_iters,
                    reinterpret_cast<cudaStream_t>(stream),
                    X0.has_value() ? X0->data() : nullptr);
                nb::list rels;
                for (double r : result.relative_residual) rels.append(r);
                return nb::make_tuple(result.iterations, rels);
            },
            nb::arg("preconditioner"), nb::arg("B").noconvert(), nb::arg("X").noconvert(),
            nb::arg("tolerance"), nb::arg("max_iters"), nb::arg("stream"),
            nb::arg("X0").noconvert() = nb::none(),
            "Lockstep k-RHS mixed-precision PCG over row-major (n, k) operands; returns "
            "(iterations, per-column relative residuals). k in {2, 4, 8}.");

    m.def(
        "dadt_nbody",
        [](f32_cuda_2d s, f32_cuda_2d mp, f32_cuda_1d sn, f32_cuda_2d r, f32_cuda_2d out,
           float didt, float mu0_4pi, uintptr_t stream) {
            int n_dip = static_cast<int>(s.shape(0));
            int n_nodes = static_cast<int>(r.shape(0));
            launch_dadt(s.data(), mp.data(), sn.data(), r.data(), out.data(), n_dip, n_nodes,
                        didt, mu0_4pi, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("s").noconvert(), nb::arg("mp").noconvert(), nb::arg("sn").noconvert(),
        nb::arg("r").noconvert(), nb::arg("out").noconvert(), nb::arg("didt"), nb::arg("mu0_4pi"),
        nb::arg("stream"),
        "Fused dA/dt at nodes from placed magnetic dipoles; writes into caller-allocated out.");

    m.def(
        "dadt_node_to_element",
        [](f32_cuda_2d dadt_nodes, i32_cuda_2d tet_nodes, f32_cuda_2d out, uintptr_t stream) {
            int n_tet = static_cast<int>(tet_nodes.shape(0));
            launch_dadt_element_average(dadt_nodes.data(), tet_nodes.data(), out.data(), n_tet,
                                        reinterpret_cast<cudaStream_t>(stream));
        },
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
        nb::arg("dadt_elm").noconvert(), nb::arg("g").noconvert(), nb::arg("neg_vc").noconvert(),
        nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(), nb::arg("b").noconvert(),
        nb::arg("stream"),
        "Deterministic node-centric RHS assembly; writes into caller-allocated b.");

    m.def(
        "rhs_assemble_weighted",
        [](f32_cuda_2d dadt_elm, f32_cuda_3d wg, i32_cuda ptr, i32_cuda idx, f32_cuda_1d b,
           uintptr_t stream) {
            int n_nodes = static_cast<int>(b.shape(0));
            int n_tet = static_cast<int>(dadt_elm.shape(0));
            launch_rhs_weighted(dadt_elm.data(), wg.data(), ptr.data(), idx.data(), b.data(),
                                n_nodes, n_tet, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("dadt_elm").noconvert(), nb::arg("wg").noconvert(), nb::arg("ptr").noconvert(),
        nb::arg("idx").noconvert(), nb::arg("b").noconvert(), nb::arg("stream"),
        "Deterministic node-centric RHS assembly with preweighted gradients.");

    m.def(
        "rhs_assemble_weighted_block",
        [](std::vector<f32_cuda_2d> dadt_elm, f32_cuda_3d wg, i32_cuda ptr, i32_cuda idx,
           f32_cuda_2d b_block, uintptr_t stream) {
            const int k = static_cast<int>(dadt_elm.size());
            if (k < 1 || k > kMaxStageBlock ||
                static_cast<int>(b_block.shape(1)) != k) {
                throw std::invalid_argument("rhs_assemble_weighted_block: bad list sizes");
            }
            const float* in_ptrs[kMaxStageBlock];
            for (int c = 0; c < k; ++c) in_ptrs[c] = dadt_elm[c].data();
            int n_nodes = static_cast<int>(b_block.shape(0));
            int n_tet = static_cast<int>(dadt_elm[0].shape(0));
            launch_rhs_weighted_block(in_ptrs, wg.data(), ptr.data(), idx.data(),
                                      b_block.data(), n_nodes, n_tet, k,
                                      reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("dadt_elm").noconvert(), nb::arg("wg").noconvert(), nb::arg("ptr").noconvert(),
        nb::arg("idx").noconvert(), nb::arg("b_block").noconvert(), nb::arg("stream"),
        "Block RHS assembly for <=8 placements: wg/node2corner read once; writes "
        "row-major (n_nodes, k) float32.");

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
        nb::arg("v_block").noconvert(), nb::arg("tet_nodes").noconvert(), nb::arg("g").noconvert(),
        nb::arg("dadt_elm").noconvert(), nb::arg("e_out").noconvert(),
        nb::arg("magn_out").noconvert(), nb::arg("stream"),
        "Block E/magnE reconstruction for <=8 placements: tet_nodes/g read once; "
        "v_block is row-major (n_nodes, k) float64.");

    m.def(
        "weighted_gradient",
        [](f32_cuda_3d g, f32_cuda_1d neg_vc, f32_cuda_3d wg, uintptr_t stream) {
            int n_tet = static_cast<int>(neg_vc.shape(0));
            launch_weighted_gradient(g.data(), neg_vc.data(), wg.data(), n_tet,
                                     reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("g").noconvert(), nb::arg("neg_vc").noconvert(), nb::arg("wg").noconvert(),
        nb::arg("stream"),
        "Precompute neg_vc-scaled gradients for repeated RHS assembly.");

    m.def(
        "reconstruct_e",
        [](f64_cuda v, i32_cuda_2d tet_nodes, f32_cuda_3d g, f32_cuda_2d dadt_elm,
           f32_cuda_2d e_out, f32_cuda_1d magn_out, uintptr_t stream) {
            int n_tet = static_cast<int>(tet_nodes.shape(0));
            launch_reconstruct(v.data(), tet_nodes.data(), g.data(), dadt_elm.data(), e_out.data(),
                               magn_out.data(), n_tet, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("v").noconvert(), nb::arg("tet_nodes").noconvert(), nb::arg("g").noconvert(),
        nb::arg("dadt_elm").noconvert(), nb::arg("e_out").noconvert(),
        nb::arg("magn_out").noconvert(), nb::arg("stream"),
        "Element-centric E/magnE reconstruction; writes into caller-allocated e_out/magn_out.");

    m.def(
        "element_weight",
        [](f64_cuda values, i32_cuda_2d tet_nodes, f32_cuda_3d g, f32_cuda_1d neg_vc,
           f64_cuda_2d w_e, uintptr_t stream) {
            int n_tet = static_cast<int>(tet_nodes.shape(0));
            launch_element_weight(values.data(), tet_nodes.data(), g.data(), neg_vc.data(),
                                  w_e.data(), n_tet, reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("values").noconvert(), nb::arg("tet_nodes").noconvert(), nb::arg("g").noconvert(),
        nb::arg("neg_vc").noconvert(), nb::arg("w_e").noconvert(), nb::arg("stream"),
        "Per-element reciprocity weight w_e = vol*sigma*(G_e values); into caller-allocated w_e.");

    m.def(
        "node_scatter3",
        [](f64_cuda_2d w_e, i32_cuda ptr, i32_cuda idx, f64_cuda_2d node_w, uintptr_t stream) {
            int n_nodes = static_cast<int>(ptr.shape(0)) - 1;
            launch_node_scatter3(w_e.data(), ptr.data(), idx.data(), node_w.data(), n_nodes,
                                 reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("w_e").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("node_w").noconvert(), nb::arg("stream"),
        "Node-centric 3-vector corner gather with 1/4 weight; into caller-allocated node_w.");

    m.def(
        "accumulate_moments",
        [](f32_cuda_1d magn, f64_cuda sum_e, f64_cuda sumsq_e, uintptr_t stream) {
            int n = static_cast<int>(magn.shape(0));
            launch_accumulate_moments(magn.data(), sum_e.data(), sumsq_e.data(), n,
                                      reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("magn").noconvert(), nb::arg("sum_e").noconvert(), nb::arg("sumsq_e").noconvert(),
        nb::arg("stream"),
        "Fused streaming |E| moments: sum_e += magn; sumsq_e += magn^2 (in place).");

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
        nb::arg("centers").noconvert(), nb::arg("handles").noconvert(),
        nb::arg("dists").noconvert(), nb::arg("a").noconvert(), nb::arg("b").noconvert(),
        nb::arg("c").noconvert(), nb::arg("tnorm").noconvert(), nb::arg("out").noconvert(),
        nb::arg("degenerate").noconvert(), nb::arg("stream"),
        "Batched closest-point scalp projection + coil frame; writes (P,16) row-major 4x4 out "
        "and a (P,) flag marking placements whose handle left the in-plane axis undefined. "
        "Pass handles=None for the projection and normal alone, with degenerate=None.");
}
