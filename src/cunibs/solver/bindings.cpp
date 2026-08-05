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
using f64_cuda_3d = nb::ndarray<double, nb::ndim<3>, nb::c_contig, nb::device::cuda>;
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
        .def("set_smoother", &NativeVCycle::set_smoother, nb::arg("degree"),
             nb::arg("lower_ratio"),
             "Set the Chebyshev degree and the lower end of its interval, as a divisor of the "
             "smoother's spectral upper bound. Must precede the first add_level.")
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
            if (col_idx.shape(0) != values.shape(0)) {
                throw std::invalid_argument("col_idx and values must have one entry per nonzero");
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
            "(block iterations, per-column relative residuals). A column freezes once its own "
            "residual meets tolerance, so it stops where it would have solved alone. "
            "k in {1, 2, 4, 8}.");

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
        nb::arg("nodes_mm").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("g").noconvert(), nb::arg("vols").noconvert(), nb::arg("stream"),
        "P1 basis-function gradients (n_tet, 4, 3) in 1/m and element volumes in m^3, from node "
        "coordinates in millimetres; writes into caller-allocated g/vols.");

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
        "build_incident_node_csr",
        [](i32_cuda_2d tet_nodes, i32_cuda ptr, i32_cuda idx, i32_cuda cand, i32_cuda sorted,
           i32_cuda out_ptr, uintptr_t stream) {
            const int n_corner = static_cast<int>(idx.shape(0));
            const int n_seg = static_cast<int>(out_ptr.shape(0)) - 1;
            const size_t needed = static_cast<size_t>(n_corner) * 4;
            if (cand.shape(0) < needed || sorted.shape(0) < needed) {
                throw std::invalid_argument(
                    "build_incident_node_csr: work buffers must hold 4 * n_corner entries");
            }
            if (ptr.shape(0) != out_ptr.shape(0)) {
                throw std::invalid_argument(
                    "build_incident_node_csr: ptr and out_ptr must both be n_seg + 1");
            }
            return build_incident_node_csr(tet_nodes.data(), ptr.data(), idx.data(), cand.data(),
                                           sorted.data(), out_ptr.data(), n_seg, n_corner,
                                           reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("tet_nodes").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("cand").noconvert(), nb::arg("sorted").noconvert(),
        nb::arg("out_ptr").noconvert(), nb::arg("stream"),
        "Distinct nodes of the tetrahedra in each segment of a corner CSR. Fills out_ptr and "
        "leaves the sorted, distinct entries in the first nnz slots of cand, which it returns. "
        "Synchronises the stream.");

    m.def(
        "build_patch_csr",
        [](i32_cuda r1_ptr, i32_cuda r1_idx, i32_cuda neighbour, int min_nodes, i32_cuda cand,
           i32_cuda sorted, i32_cuda out_ptr, uintptr_t stream) {
            const int n_slots = static_cast<int>(r1_ptr.shape(0)) - 1;
            if (neighbour.shape(0) != r1_idx.shape(0)) {
                throw std::invalid_argument(
                    "build_patch_csr: neighbour must have one entry per first-ring node");
            }
            if (out_ptr.shape(0) != r1_ptr.shape(0)) {
                throw std::invalid_argument(
                    "build_patch_csr: r1_ptr and out_ptr must both be n_slots + 1");
            }
            if (cand.shape(0) != sorted.shape(0)) {
                throw std::invalid_argument("build_patch_csr: work buffers must match in size");
            }
            return build_patch_csr(r1_ptr.data(), r1_idx.data(), neighbour.data(), min_nodes,
                                   cand.data(), sorted.data(), out_ptr.data(), n_slots,
                                   static_cast<int>(cand.shape(0)),
                                   reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("r1_ptr").noconvert(), nb::arg("r1_idx").noconvert(),
        nb::arg("neighbour").noconvert(), nb::arg("min_nodes"), nb::arg("cand").noconvert(),
        nb::arg("sorted").noconvert(), nb::arg("out_ptr").noconvert(), nb::arg("stream"),
        "Recovery patches from the first-ring CSR: a slot reaching fewer than min_nodes grows to "
        "the union of its neighbours' rings. Same output convention as build_incident_node_csr.");

    m.def(
        "assemble_stiffness_values",
        [](f64_cuda_3d g, f64_cuda scale, i32_cuda_2d tet_nodes, i32_cuda ptr, i32_cuda idx,
           i32_cuda indptr, i32_cuda indices, f64_cuda data, uintptr_t stream) {
            const int n_rows = static_cast<int>(indptr.shape(0)) - 1;
            launch_stiffness_rows(g.data(), scale.data(), tet_nodes.data(), ptr.data(), idx.data(),
                                  indptr.data(), indices.data(), data.data(), n_rows,
                                  reinterpret_cast<cudaStream_t>(stream));
        },
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
        nb::arg("indptr").noconvert(), nb::arg("indices").noconvert(), nb::arg("data").noconvert(),
        nb::arg("dinv").noconvert(), nb::arg("stream"),
        "l1-Jacobi smoother scaling 1 / (sign(a_ii) * sum_j |a_ij|), one thread per row.");

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
        nb::arg("nodes_mm").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("vols").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("slot_node").noconvert(), nb::arg("is_boundary").noconvert(),
        nb::arg("w").noconvert(), nb::arg("n_fallback").noconvert(), nb::arg("stream"),
        "One scalar patch weight per corner. Pass slot_node=None when a slot is a node, "
        "is_boundary=None to fit every slot, n_fallback=None to skip the counter.");

    m.def(
        "hpr_weights",
        [](f64_cuda_2d nodes_mm, i32_cuda pptr, i32_cuda pidx, i32_cuda slot_node,
           f32_cuda_2d w, i32_cuda status, uintptr_t stream) {
            const int n_slots = static_cast<int>(pptr.shape(0)) - 1;
            if (w.shape(0) != pidx.shape(0) || w.shape(1) != 3) {
                throw std::invalid_argument("hpr_weights: w must be (nnz, 3)");
            }
            if (static_cast<int>(slot_node.shape(0)) != n_slots ||
                static_cast<int>(status.shape(0)) != n_slots) {
                throw std::invalid_argument("hpr_weights: slot_node/status must be per slot");
            }
            launch_hpr_weights(nodes_mm.data(), pptr.data(), pidx.data(), slot_node.data(),
                               w.data(), status.data(), n_slots,
                               reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("nodes_mm").noconvert(), nb::arg("pptr").noconvert(), nb::arg("pidx").noconvert(),
        nb::arg("slot_node").noconvert(), nb::arg("w").noconvert(), nb::arg("status").noconvert(),
        nb::arg("stream"),
        "Harmonic-constrained potential-recovery weights: grad_v[s] = sum_m w[s,m] v[m]. "
        "status is 0 harmonic, 1 linear fallback, 2 no gradient determined.");

    m.def(
        "hpr_grad",
        [](f64_cuda_2d v_block, f32_cuda_2d w, i32_cuda pptr, i32_cuda pidx, i32_cuda slot_node,
           std::vector<f32_cuda_2d> dadt_nodes, std::vector<f32_cuda_2d> e_slots,
           uintptr_t stream) {
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
                            da_ptrs, out_ptrs, n_slots, k, stride,
                            reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("v_block").noconvert(), nb::arg("w").noconvert(), nb::arg("pptr").noconvert(),
        nb::arg("pidx").noconvert(), nb::arg("slot_node").noconvert(),
        nb::arg("dadt_nodes").noconvert(), nb::arg("e_slots").noconvert(), nb::arg("stream"),
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
        nb::arg("e_in").noconvert(), nb::arg("w").noconvert(), nb::arg("ptr").noconvert(),
        nb::arg("idx").noconvert(), nb::arg("e_slots").noconvert(), nb::arg("stream"),
        "Patch gather E*[s] = sum_c w[c] E[c>>2] for <=8 placements; the operator is read once "
        "for the whole block and each column matches its serial counterpart bitwise.");

    m.def(
        "recover_elements",
        [](std::vector<f32_cuda_2d> e_slots, i32_cuda slot_of_corner,
           std::vector<f32_cuda_2d> e_out, std::vector<f32_cuda_1d> magn_out, uintptr_t stream) {
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
            launch_recover_elements(in_ptrs, slot_of_corner.data(), e_ptrs, m_ptrs, n_tet, k,
                                    reinterpret_cast<cudaStream_t>(stream));
        },
        nb::arg("e_slots").noconvert(), nb::arg("slot_of_corner").noconvert(),
        nb::arg("e_out").noconvert(), nb::arg("magn_out").noconvert(), nb::arg("stream"),
        "Sample the recovered field back at the barycentres and take its magnitude.");

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
