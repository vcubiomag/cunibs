#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>

#include "amg/aggregate.hpp"
#include "amg/l1.hpp"
#include "amg/solver.hpp"
#include "amg/vcycle.hpp"
#include "coil/coil.hpp"
#include "fem/fem.hpp"

namespace nb = nanobind;

// Device arrays reach CUDA as raw pointers, so every one must be C-contiguous. A fixed trailing
// extent is part of the type, so nanobind rejects a wrong shape before the call is made.
template <typename T, nb::ssize_t... Shape>
using Dev = nb::ndarray<T, nb::shape<Shape...>, nb::c_contig, nb::device::cuda>;

using f64_1d = Dev<double, -1>;
using f64_2d = Dev<double, -1, -1>;
using f32_1d = Dev<float, -1>;
using f32_2d = Dev<float, -1, -1>;
using i32_1d = Dev<int32_t, -1>;
using i64_1d = Dev<int64_t, -1>;
using u8_1d = Dev<uint8_t, -1>;
using u8_2d = Dev<uint8_t, -1, -1>;
// Per-row 3-vectors: node coordinates, fields, triangle corners.
using Rows3d = Dev<double, -1, 3>;
using Rows3f = Dev<float, -1, 3>;
using Tets = Dev<int32_t, -1, 4>;
using Tris = Dev<int32_t, -1, 3>;
using TetGradd = Dev<double, -1, 4, 3>;
using TetGradf = Dev<float, -1, 4, 3>;
using TriVerts = Dev<double, -1, 3, 3>;
using Frames = Dev<double, -1, 4, 4>;
// An interleaved (n_nodes, k, 3) nodal block.
using Interleaved = Dev<float, -1, -1, 3>;
// Per-set source weights (n_sets, n_src, 3).
using SetRows3f = Dev<float, -1, -1, 3>;

namespace {

cudaStream_t as_stream(uintptr_t stream) { return reinterpret_cast<cudaStream_t>(stream); }

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(what);
}

template <typename T>
auto data_or_null(const std::optional<T>& a) -> decltype(a->data()) {
    return a ? a->data() : nullptr;
}

int block_width(std::size_t k, const char* what) {
    require(k >= 1 && k <= static_cast<std::size_t>(kMaxStageBlock), what);
    return static_cast<int>(k);
}

// The per-placement pointers of a list of k arrays, each `rows` long.
template <typename Ptr, typename Array>
std::array<Ptr, kMaxStageBlock> pack(const std::vector<Array>& arrays, int k, std::size_t rows,
                                     const char* what) {
    require(arrays.size() == static_cast<std::size_t>(k), what);
    std::array<Ptr, kMaxStageBlock> out{};
    for (int c = 0; c < k; ++c) {
        require(arrays[c].shape(0) == rows, what);
        out[c] = arrays[c].data();
    }
    return out;
}

// The same for an output a caller may skip: an empty list leaves every pointer null.
template <typename Array>
std::array<float*, kMaxStageBlock> optional_pack(const std::vector<Array>& arrays, int k,
                                                 std::size_t rows, const char* what) {
    if (arrays.empty()) return {};
    return pack<float*>(arrays, k, rows, what);
}

// A grid's origin and spacing (metres) and shape, as the ADM kernels take them.
struct GridArgs {
    double origin[3];
    double spacing[3];
    int shape[3];

    std::int64_t points() const {
        return static_cast<std::int64_t>(shape[0]) * shape[1] * shape[2];
    }
};

GridArgs grid_args(const std::vector<double>& origin, const std::vector<double>& spacing,
                   const std::vector<int>& shape) {
    require(origin.size() == 3 && spacing.size() == 3 && shape.size() == 3,
            "grid origin, spacing and shape must each have 3 entries");
    GridArgs g;
    for (int i = 0; i < 3; ++i) {
        require(shape[i] >= 2 && spacing[i] > 0.0,
                "grid needs at least 2 samples and a positive spacing");
        g.origin[i] = origin[i];
        g.spacing[i] = spacing[i];
        g.shape[i] = shape[i];
    }
    return g;
}

// The ADM sampling and near-source kernels work in grid index units and scale by 1 / spacing^2,
// which needs the same spacing on every axis.
GridArgs isotropic_grid_args(const std::vector<double>& origin, const std::vector<double>& spacing,
                             const std::vector<int>& shape, const char* what) {
    const GridArgs g = grid_args(origin, spacing, shape);
    require(g.spacing[1] == g.spacing[0] && g.spacing[2] == g.spacing[0],
            (std::string(what) + ": the grid spacing must be equal on every axis").c_str());
    return g;
}

void require_sources(const Rows3f& src, const SetRows3f& src_w, int n_dir, const char* what) {
    require(static_cast<int>(src_w.shape(0)) == n_dir && src_w.shape(1) == src.shape(0),
            (std::string(what) + ": src_w must be (n_dir, n_src, 3)").c_str());
}

void require_slots(const i32_1d& list_ptr, std::size_t n_slots, const f32_1d& q_near,
                   const f32_1d& entries, int n_dir, const char* what) {
    const std::size_t width = static_cast<std::size_t>(adm_entry_floats(n_dir));
    require(list_ptr.shape(0) == n_slots + 1 && q_near.shape(0) == n_slots * 8 * n_dir * 3 &&
                entries.shape(0) % width == 0 &&
                reinterpret_cast<std::uintptr_t>(entries.data()) % 16 == 0,
            (std::string(what) + ": list_ptr, q_near and entries must match the slots").c_str());
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
    const std::unique_ptr<HostCopy> copy(static_cast<HostCopy*>(arg));
    std::memcpy(copy->dst, copy->src, copy->bytes);
}

using BlockResult = std::tuple<int, std::vector<double>, std::vector<int>>;

BlockResult as_tuple(PcgBlockResult r) {
    return {r.iterations, std::move(r.relative_residual), std::move(r.column_iterations)};
}

}  // namespace

NB_MODULE(_solver_ext, m) {
    nb::exception<CudaOutOfMemory>(m, "CudaOutOfMemory", PyExc_MemoryError);
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
            [](NativeVCycle& self, const i32_1d& row_ptr, const i32_1d& col_idx,
               const f32_1d& values, const f32_1d& dinv, const i32_1d& p_row_ptr,
               const i32_1d& p_col_idx, const f32_1d& p_values, const i32_1d& r_row_ptr,
               const i32_1d& r_col_idx, const f32_1d& r_values, uintptr_t stream) {
                const int n_rows = static_cast<int>(row_ptr.shape(0)) - 1;
                require(r_values.shape(0) == p_values.shape(0), "R must be the transpose of P");
                require(p_row_ptr.shape(0) == row_ptr.shape(0),
                        "P must have one row per fine row");
                self.add_level(n_rows, static_cast<int>(values.shape(0)),
                               static_cast<int>(r_row_ptr.shape(0)) - 1,
                               static_cast<int>(p_values.shape(0)), row_ptr.data(),
                               col_idx.data(), values.data(), dinv.data(), p_row_ptr.data(),
                               p_col_idx.data(), p_values.data(), r_row_ptr.data(),
                               r_col_idx.data(), r_values.data(), as_stream(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("row_ptr").noconvert(), nb::arg("col_idx").noconvert(),
            nb::arg("values").noconvert(), nb::arg("dinv").noconvert(),
            nb::arg("p_row_ptr").noconvert(), nb::arg("p_col_idx").noconvert(),
            nb::arg("p_values").noconvert(), nb::arg("r_row_ptr").noconvert(),
            nb::arg("r_col_idx").noconvert(), nb::arg("r_values").noconvert(),
            nb::arg("stream"),
            "Append one non-coarsest level (fp32 CSR, 1/guard(d), prolongator CSR and its "
            "transpose); device arrays are copied into solver-owned buffers.")
        .def(
            "set_coarse",
            [](NativeVCycle& self, const f32_2d& ainv, uintptr_t stream) {
                require(ainv.shape(0) == ainv.shape(1), "coarse inverse must be square");
                self.set_coarse(static_cast<int>(ainv.shape(0)), ainv.data(), as_stream(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("ainv").noconvert(), nb::arg("stream"),
            "Set the dense (row-major) inverse of the coarsest matrix.")
        .def("finalize", &NativeVCycle::finalize,
             "Validate level chain consistency; required before apply.")
        .def("n_levels", &NativeVCycle::n_levels,
             "Number of coarsening levels, excluding the coarsest (dense-inverse) one.");

    m.def(
        "select_size4",
        [](const i32_1d& row_ptr, const i32_1d& col_idx, const f32_1d& values,
           const i32_1d& aggregates, uintptr_t stream) {
            const int n_rows = static_cast<int>(row_ptr.shape(0)) - 1;
            require(static_cast<int>(aggregates.shape(0)) == n_rows,
                    "aggregates must have one entry per row");
            require(col_idx.shape(0) == values.shape(0),
                    "col_idx and values must have one entry per nonzero");
            return select_size4(n_rows, static_cast<int>(values.shape(0)), row_ptr.data(),
                                col_idx.data(), values.data(), aggregates.data(),
                                as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("row_ptr").noconvert(), nb::arg("col_idx").noconvert(),
        nb::arg("values").noconvert(), nb::arg("aggregates").noconvert(), nb::arg("stream"),
        "Unsmoothed pairwise aggregation (AMGx SIZE_4). Fills `aggregates` with a surjective "
        "row -> aggregate map and returns the aggregate count. Synchronises the stream.");

    nb::class_<PcgAmgSolver>(m, "PcgAmgSolver")
        .def(
            "__init__",
            [](PcgAmgSolver* self, const i32_1d& row_ptr, const i32_1d& col_idx,
               const f64_1d& values, uintptr_t stream) {
                new (self) PcgAmgSolver(static_cast<int>(row_ptr.shape(0)) - 1,
                                        static_cast<int>(values.shape(0)), row_ptr.data(),
                                        col_idx.data(), values.data(), as_stream(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("row_ptr").noconvert(), nb::arg("col_idx").noconvert(),
            nb::arg("values").noconvert(), nb::arg("stream"))
        .def_prop_ro("n", &PcgAmgSolver::n, "Rows of the matrix.")
        .def_prop_ro("nnz", &PcgAmgSolver::nnz, "Nonzeros of the matrix's entry CSR.")
        .def(
            "export_csr",
            [](PcgAmgSolver& self, const i32_1d& row_ptr, const std::optional<i32_1d>& col_idx,
               const std::optional<f64_1d>& values, uintptr_t stream) {
                require(static_cast<int>(row_ptr.shape(0)) == self.n() + 1 &&
                            (!col_idx || static_cast<int>(col_idx->shape(0)) == self.nnz()) &&
                            (!values || static_cast<int>(values->shape(0)) == self.nnz()),
                        "export_csr: row_ptr must be (n + 1), col_idx and values (nnz)");
                self.export_csr(row_ptr.data(), data_or_null(col_idx), data_or_null(values),
                                as_stream(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("row_ptr").noconvert(), nb::arg("col_idx").noconvert().none(),
            nb::arg("values").noconvert().none(), nb::arg("stream"),
            "Write the matrix's entry CSR into caller arrays; col_idx or values may be None.")
        .def(
            "update_values",
            [](PcgAmgSolver& self, const f64_1d& values, uintptr_t stream) {
                require(static_cast<int>(values.shape(0)) == self.nnz(),
                        "values must have one entry per nonzero");
                self.update_values(values.data(), as_stream(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("values").noconvert(), nb::arg("stream"))
        .def(
            "solve_mixed",
            [](PcgAmgSolver& self, NativeVCycle& preconditioner, const f64_1d& b, const f64_1d& x,
               double tolerance, int max_iters, uintptr_t stream, const std::optional<f64_1d>& x0) {
                const std::size_t n = self.n();
                require(b.shape(0) == n && x.shape(0) == n && (!x0 || x0->shape(0) == n),
                        "b, x and x0 must have one entry per row");
                const PcgResult r =
                    self.solve_mixed(preconditioner, b.data(), x.data(), tolerance, max_iters,
                                     as_stream(stream), data_or_null(x0));
                return std::make_tuple(r.iterations, r.relative_residual);
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("preconditioner"), nb::arg("b").noconvert(), nb::arg("x").noconvert(),
            nb::arg("tolerance"), nb::arg("max_iters"), nb::arg("stream"),
            nb::arg("x0").noconvert() = nb::none())
        .def(
            "solve_mixed_block",
            [](PcgAmgSolver& self, NativeVCycle& preconditioner, const f64_2d& B, const f64_2d& X,
               double tolerance, int max_iters, uintptr_t stream, const std::optional<f64_2d>& X0) {
                const std::size_t n = self.n();
                const std::size_t k = B.shape(1);
                require(B.shape(0) == n && X.shape(0) == n && X.shape(1) == k &&
                            (!X0 || (X0->shape(0) == n && X0->shape(1) == k)),
                        "B, X and X0 must all be (n, k)");
                return as_tuple(self.solve_mixed_block(preconditioner, B.data(), X.data(),
                                                       static_cast<int>(k), tolerance, max_iters,
                                                       as_stream(stream), data_or_null(X0)));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("preconditioner"), nb::arg("B").noconvert(), nb::arg("X").noconvert(),
            nb::arg("tolerance"), nb::arg("max_iters"), nb::arg("stream"),
            nb::arg("X0").noconvert() = nb::none(),
            "Lockstep k-RHS mixed-precision PCG over row-major (n, k) operands; returns "
            "(block iterations, per-column relative residuals, per-column iteration counts). A "
            "column freezes once its own residual meets tolerance, so it stops where it would "
            "have solved alone. k in {1, 2, 4, 8}.")
        .def(
            "solve_affine_block",
            [](PcgAmgSolver& self, NativeVCycle& preconditioner, AffineTerms& terms,
               const f64_2d& B, const f64_2d& X,
               const nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>& sigma,
               double tolerance, int max_iters, uintptr_t stream, const std::optional<f64_2d>& X0) {
                const std::size_t n = self.n();
                const std::size_t k = B.shape(1);
                require(B.shape(0) == n && X.shape(0) == n && X.shape(1) == k &&
                            sigma.shape(0) == k &&
                            static_cast<int>(sigma.shape(1)) == terms.n_terms() &&
                            (!X0 || (X0->shape(0) == n && X0->shape(1) == k)),
                        "B, X and X0 must be (n, k) and sigma (k, n_terms)");
                return as_tuple(self.solve_affine_block(
                    preconditioner, terms, B.data(), X.data(), static_cast<int>(k), sigma.data(),
                    tolerance, max_iters, as_stream(stream), data_or_null(X0)));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("preconditioner"), nb::arg("terms"), nb::arg("B").noconvert(),
            nb::arg("X").noconvert(), nb::arg("sigma").noconvert(), nb::arg("tolerance"),
            nb::arg("max_iters"), nb::arg("stream"), nb::arg("X0").noconvert() = nb::none(),
            "The block solve with column c's operator A(sigma[c]) from `terms`: every column its "
            "own matrix, sharing the parts' reads and the preconditioner. sigma is a host "
            "(k, n_terms) float64 array. X0 may be X itself. Returns what solve_mixed_block "
            "does.");

    nb::class_<AffineTerms>(m, "AffineTerms")
        .def(
            "__init__",
            [](AffineTerms* self, const PcgAmgSolver& solver, const f64_2d& terms,
               uintptr_t stream) {
                require(static_cast<int>(terms.shape(1)) == solver.nnz(),
                        "affine terms must be (n_terms, nnz)");
                new (self) AffineTerms(solver, static_cast<int>(terms.shape(0)), terms.data(),
                                       as_stream(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(), nb::keep_alive<1, 2>(),
            nb::arg("solver"), nb::arg("terms").noconvert(), nb::arg("stream"),
            "An operator affine in per-column coefficients over solver's pattern, "
            "A(sigma) = sum_t sigma_t A_t, from its parts' values in entry order (n_terms, nnz). "
            "Keeps solver alive.")
        .def_prop_ro("n_terms", &AffineTerms::n_terms)
        .def(
            "values",
            [](const AffineTerms& self,
               const nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>& sigma,
               const f64_1d& out, uintptr_t stream) {
                require(static_cast<int>(sigma.shape(0)) == self.n_terms(),
                        "values: sigma must have one entry per term");
                self.values(sigma.data(), out.data(), as_stream(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("sigma").noconvert(), nb::arg("out").noconvert(), nb::arg("stream"),
            "out = sum_t sigma[t] A_t in the solver's entry order, for host sigma (n_terms,).")
        .def(
            "matmul",
            [](const AffineTerms& self, int term, const f64_2d& x, const f64_2d& y,
               uintptr_t stream) {
                require(static_cast<int>(x.shape(0)) == self.n() && y.shape(0) == x.shape(0) &&
                            y.shape(1) == x.shape(1),
                        "matmul: x and y must both be (n, m)");
                self.matmul(term, x.data(), y.data(), static_cast<int>(x.shape(1)),
                            as_stream(stream));
            },
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("term"), nb::arg("x").noconvert(), nb::arg("y").noconvert(), nb::arg("stream"),
            "y = A_term x for row-major (n, m) float64 x, each row summed over the part's own "
            "nonzeros in column order with one fused multiply-add per entry.");

    m.def(
        "p1_gradients",
        [](const Rows3d& nodes_mm, const Tets& tet_nodes, const TetGradd& g, const f64_1d& vols,
           uintptr_t stream) {
            const std::size_t n_tet = tet_nodes.shape(0);
            require(g.shape(0) == n_tet && vols.shape(0) == n_tet,
                    "p1_gradients: g and vols must have one entry per tet");
            launch_p1_gradients(nodes_mm.data(), tet_nodes.data(), g.data(), vols.data(),
                                static_cast<int>(n_tet), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("nodes_mm").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("g").noconvert(), nb::arg("vols").noconvert(), nb::arg("stream"),
        "P1 basis-function gradients (n_tet, 4, 3) in 1/m and element volumes in m^3, from node "
        "coordinates in millimetres; writes into caller-allocated g/vols.");

    m.def(
        "dadt_nbody",
        [](const Rows3f& s, const Dev<float, -1, 6>& mp, const f32_1d& sn, const Rows3f& r,
           const f32_2d& out, float didt, float mu0_4pi, uintptr_t stream, int column) {
            const std::size_t n_dip = s.shape(0);
            const std::size_t n_nodes = r.shape(0);
            const int stride = static_cast<int>(out.shape(1));
            require(mp.shape(0) == n_dip && sn.shape(0) == n_dip,
                    "dadt_nbody: s, mp and sn must have one entry per dipole");
            require(out.shape(0) == n_nodes && stride % 3 == 0 && column >= 0 &&
                        3 * (column + 1) <= stride,
                    "dadt_nbody: out must be (n_nodes, 3k) with room for column");
            launch_dadt(s.data(), mp.data(), sn.data(), r.data(), out.data() + 3 * column, stride,
                        static_cast<int>(n_dip), static_cast<int>(n_nodes), didt, mu0_4pi,
                        as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("s").noconvert(), nb::arg("mp").noconvert(), nb::arg("sn").noconvert(),
        nb::arg("r").noconvert(), nb::arg("out").noconvert(), nb::arg("didt"), nb::arg("mu0_4pi"),
        nb::arg("stream"), nb::arg("column") = 0,
        "Fused dA/dt at nodes from placed magnetic dipoles: mp is (n_dip, 6), each moment then its "
        "cross product with the position. out is (n_nodes, 3k), an interleaved (n_nodes, k, 3) "
        "block, and the field fills column column.");

    m.def(
        "rhs_operator_fill",
        [](const TetGradd& g, const f64_1d& scale, const Tets& tet_nodes, const i32_1d& ptr,
           const i32_1d& idx, const i32_1d& indptr, const i32_1d& indices,
           const Dev<float, -1, 4>& op, uintptr_t stream) {
            require(op.shape(0) == indices.shape(0) && ptr.shape(0) == indptr.shape(0),
                    "rhs_operator_fill: shape mismatch");
            launch_rhs_operator_fill(g.data(), scale.data(), tet_nodes.data(), ptr.data(),
                                     idx.data(), indptr.data(), indices.data(), op.data(),
                                     static_cast<int>(indptr.shape(0)) - 1, as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("g").noconvert(), nb::arg("scale").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(), nb::arg("indptr").noconvert(),
        nb::arg("indices").noconvert(), nb::arg("op").noconvert(), nb::arg("stream"),
        "Build the forward RHS operator over the stiffness pattern: op is (nnz, 4), each row a "
        "packed (Bx, By, Bz, column); scale is neg_vc / 4 in float64.");

    m.def(
        "rhs_operator_apply",
        [](const i32_1d& indptr, const Dev<float, -1, 4>& op, const Interleaved& a, const f32_2d& b,
           uintptr_t stream) {
            const std::size_t n_nodes = indptr.shape(0) - 1;
            const int k = block_width(a.shape(1), "rhs_operator_apply: bad block size");
            require(a.shape(0) == n_nodes && b.shape(0) == n_nodes &&
                        static_cast<int>(b.shape(1)) == k,
                    "rhs_operator_apply: shape mismatch");
            launch_rhs_operator(indptr.data(), op.data(), a.data(), b.data(),
                                static_cast<int>(n_nodes), k, as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("indptr").noconvert(), nb::arg("op").noconvert(), nb::arg("a").noconvert(),
        nb::arg("b").noconvert(), nb::arg("stream"),
        "b = B a for an interleaved (n_nodes, k, 3) nodal dA/dt; writes row-major (n_nodes, k) "
        "float32. Each column's sum order is independent of k.");

    m.def(
        "dadt_node_to_element",
        [](const Rows3f& dadt_nodes, const Tets& tet_nodes, const Rows3f& out, uintptr_t stream) {
            require(out.shape(0) == tet_nodes.shape(0),
                    "dadt_node_to_element: out must have one row per tet");
            launch_dadt_element_average(dadt_nodes.data(), tet_nodes.data(), out.data(),
                                        static_cast<int>(tet_nodes.shape(0)), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("dadt_nodes").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("out").noconvert(), nb::arg("stream"),
        "Average nodal dA/dt onto tetrahedra; writes into caller-allocated out.");

    m.def(
        "rhs_assemble",
        [](const Rows3f& dadt_elm, const TetGradf& g, const f32_1d& neg_vc, const i32_1d& ptr,
           const i32_1d& idx, const f32_1d& b, uintptr_t stream) {
            require(ptr.shape(0) == b.shape(0) + 1, "rhs_assemble: ptr must be n_nodes + 1");
            launch_rhs(dadt_elm.data(), g.data(), neg_vc.data(), ptr.data(), idx.data(), b.data(),
                       static_cast<int>(b.shape(0)), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("dadt_elm").noconvert(), nb::arg("g").noconvert(), nb::arg("neg_vc").noconvert(),
        nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(), nb::arg("b").noconvert(),
        nb::arg("stream"),
        "Deterministic node-centric RHS assembly; writes into caller-allocated b.");

    m.def(
        "tet_lowest_node",
        [](const i32_1d& inverse, const Tets& tet_nodes, const i32_1d& lowest, uintptr_t stream) {
            require(lowest.shape(0) == tet_nodes.shape(0),
                    "tet_lowest_node: lowest must have one entry per tet");
            launch_tet_lowest_node(inverse.data(), tet_nodes.data(), lowest.data(),
                                   static_cast<int>(tet_nodes.shape(0)), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("inverse").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("lowest").noconvert(), nb::arg("stream"),
        "Smallest permuted node index of each tetrahedron, the sort key of the Morton layout.");

    m.def(
        "count_incident_node_csr",
        [](const Tets& tet_nodes, const i32_1d& ptr, const i32_1d& idx, const i32_1d& out_ptr,
           uintptr_t stream) {
            require(ptr.shape(0) == out_ptr.shape(0),
                    "count_incident_node_csr: ptr and out_ptr must both be n_seg + 1");
            return count_incident_node_csr(tet_nodes.data(), ptr.data(), idx.data(),
                                           out_ptr.data(),
                                           static_cast<int>(out_ptr.shape(0)) - 1,
                                           as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("tet_nodes").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("out_ptr").noconvert(), nb::arg("stream"),
        "Row offsets of the distinct nodes of the tetrahedra in each segment of a corner CSR. "
        "Fills out_ptr and returns nnz, the length to allocate the column index at. "
        "Synchronises the stream.");

    m.def(
        "fill_incident_node_csr",
        [](const Tets& tet_nodes, const i32_1d& ptr, const i32_1d& idx, const i32_1d& out_ptr,
           const i32_1d& out_idx, uintptr_t stream) {
            require(ptr.shape(0) == out_ptr.shape(0),
                    "fill_incident_node_csr: ptr and out_ptr must both be n_seg + 1");
            fill_incident_node_csr(tet_nodes.data(), ptr.data(), idx.data(), out_ptr.data(),
                                   out_idx.data(), static_cast<int>(out_ptr.shape(0)) - 1,
                                   as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("tet_nodes").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("out_ptr").noconvert(), nb::arg("out_idx").noconvert(), nb::arg("stream"),
        "Fill the column index for the rows count_incident_node_csr sized.");

    m.def(
        "count_patch_csr",
        [](const i32_1d& r1_ptr, const i32_1d& r1_idx, const i32_1d& neighbour, int min_nodes,
           const i32_1d& out_ptr, uintptr_t stream) {
            require(neighbour.shape(0) == r1_idx.shape(0),
                    "count_patch_csr: neighbour must have one entry per first-ring node");
            require(out_ptr.shape(0) == r1_ptr.shape(0),
                    "count_patch_csr: r1_ptr and out_ptr must both be n_slots + 1");
            return count_patch_csr(r1_ptr.data(), r1_idx.data(), neighbour.data(), min_nodes,
                                   out_ptr.data(), static_cast<int>(r1_ptr.shape(0)) - 1,
                                   as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("r1_ptr").noconvert(), nb::arg("r1_idx").noconvert(),
        nb::arg("neighbour").noconvert(), nb::arg("min_nodes"), nb::arg("out_ptr").noconvert(),
        nb::arg("stream"),
        "Row offsets of the recovery patches: a slot reaching fewer than min_nodes grows to the "
        "union of its neighbours' rings. Same convention as count_incident_node_csr.");

    m.def(
        "fill_patch_csr",
        [](const i32_1d& r1_ptr, const i32_1d& r1_idx, const i32_1d& neighbour, int min_nodes,
           const i32_1d& out_ptr, const i32_1d& out_idx, uintptr_t stream) {
            require(neighbour.shape(0) == r1_idx.shape(0),
                    "fill_patch_csr: neighbour must have one entry per first-ring node");
            require(out_ptr.shape(0) == r1_ptr.shape(0),
                    "fill_patch_csr: r1_ptr and out_ptr must both be n_slots + 1");
            fill_patch_csr(r1_ptr.data(), r1_idx.data(), neighbour.data(), min_nodes,
                           out_ptr.data(), out_idx.data(),
                           static_cast<int>(r1_ptr.shape(0)) - 1, as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("r1_ptr").noconvert(), nb::arg("r1_idx").noconvert(),
        nb::arg("neighbour").noconvert(), nb::arg("min_nodes"), nb::arg("out_ptr").noconvert(),
        nb::arg("out_idx").noconvert(), nb::arg("stream"),
        "Fill the column index for the rows count_patch_csr sized.");

    m.def(
        "assemble_stiffness_values",
        [](const TetGradd& g, const f64_1d& scale, const Tets& tet_nodes, const i32_1d& ptr,
           const i32_1d& idx, const i32_1d& indptr, const i32_1d& indices, const f64_1d& data,
           uintptr_t stream) {
            require(data.shape(0) == indices.shape(0) && ptr.shape(0) == indptr.shape(0),
                    "assemble_stiffness_values: shape mismatch");
            launch_stiffness_rows(g.data(), scale.data(), tet_nodes.data(), ptr.data(), idx.data(),
                                  indptr.data(), indices.data(), data.data(),
                                  static_cast<int>(indptr.shape(0)) - 1, as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("g").noconvert(), nb::arg("scale").noconvert(), nb::arg("tet_nodes").noconvert(),
        nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(), nb::arg("indptr").noconvert(),
        nb::arg("indices").noconvert(), nb::arg("data").noconvert(), nb::arg("stream"),
        "Fill a CSR pattern's values with the stiffness matrix, one thread per row.");

    m.def(
        "l1_dinv",
        [](const i32_1d& indptr, const i32_1d& indices, const f32_1d& data, const f32_1d& dinv,
           uintptr_t stream) {
            const std::size_t n_rows = indptr.shape(0) - 1;
            require(dinv.shape(0) == n_rows, "dinv must have one entry per row");
            require(indices.shape(0) == data.shape(0),
                    "indices and data must have one entry per nonzero");
            launch_l1_dinv(indptr.data(), indices.data(), data.data(), dinv.data(),
                           static_cast<int>(n_rows), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("indptr").noconvert(), nb::arg("indices").noconvert(), nb::arg("data").noconvert(),
        nb::arg("dinv").noconvert(), nb::arg("stream"),
        "l1-Jacobi smoother scaling 1 / (sign(a_ii) * sum_j |a_ij|), one thread per row.");

    m.def(
        "reconstruct_e_nodal_block",
        [](const f64_2d& v_block, const Tets& tet_nodes, const TetGradf& g,
           const Interleaved& dadt_nodes, const std::vector<Rows3f>& e_out,
           const std::vector<f32_1d>& magn_out, uintptr_t stream) {
            const int k = block_width(v_block.shape(1), "reconstruct_e_nodal_block: bad width");
            const std::size_t kd = dadt_nodes.shape(1);
            require(dadt_nodes.shape(0) == v_block.shape(0) &&
                        (kd == static_cast<std::size_t>(k) || kd == 1),
                    "reconstruct_e_nodal_block: dadt_nodes must be (n_nodes, k or 1, 3)");
            const std::size_t n_tet = tet_nodes.shape(0);
            const char* bad = "reconstruct_e_nodal_block: e_out and magn_out must be k arrays";
            const auto eo = optional_pack(e_out, k, n_tet, bad);
            const auto mo = optional_pack(magn_out, k, n_tet, bad);
            launch_reconstruct_nodal_block(v_block.data(), tet_nodes.data(), g.data(),
                                           dadt_nodes.data(), static_cast<std::int64_t>(3 * kd),
                                           kd == 1 ? 0 : 3, eo.data(), mo.data(),
                                           static_cast<int>(n_tet), k, as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("v_block").noconvert(), nb::arg("tet_nodes").noconvert(), nb::arg("g").noconvert(),
        nb::arg("dadt_nodes").noconvert(), nb::arg("e_out").noconvert(),
        nb::arg("magn_out").noconvert(), nb::arg("stream"),
        "reconstruct_e_block from nodal dA/dt, averaged onto each element in the kernel. "
        "dadt_nodes is interleaved (n_nodes, k, 3), or (n_nodes, 1, 3) for one field every "
        "placement shares.");

    m.def(
        "reconstruct_e",
        [](const f64_1d& v, const Tets& tet_nodes, const TetGradf& g, const Rows3f& dadt_elm,
           const Rows3f& e_out, const f32_1d& magn_out, uintptr_t stream) {
            const std::size_t n_tet = tet_nodes.shape(0);
            require(dadt_elm.shape(0) == n_tet && e_out.shape(0) == n_tet &&
                        magn_out.shape(0) == n_tet,
                    "reconstruct_e: dadt_elm, e_out and magn_out must have one row per tet");
            launch_reconstruct(v.data(), tet_nodes.data(), g.data(), dadt_elm.data(), e_out.data(),
                               magn_out.data(), static_cast<int>(n_tet), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("v").noconvert(), nb::arg("tet_nodes").noconvert(), nb::arg("g").noconvert(),
        nb::arg("dadt_elm").noconvert(), nb::arg("e_out").noconvert(),
        nb::arg("magn_out").noconvert(), nb::arg("stream"),
        "Element-centric E/magnE reconstruction; writes into caller-allocated e_out/magn_out.");

    m.def(
        "mark_outer_boundary",
        [](const Tets& tet_nodes, const i32_1d& ptr, const i32_1d& idx, const i32_1d& is_boundary,
           uintptr_t stream) {
            const std::size_t n_tet = tet_nodes.shape(0);
            require(idx.shape(0) == n_tet * 4,
                    "mark_outer_boundary: idx must hold one entry per tet corner");
            require(ptr.shape(0) == is_boundary.shape(0) + 1,
                    "mark_outer_boundary: ptr must hold n_nodes + 1 entries");
            launch_mark_outer_boundary(tet_nodes.data(), ptr.data(), idx.data(),
                                       is_boundary.data(), static_cast<int>(n_tet),
                                       as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("tet_nodes").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("is_boundary").noconvert(), nb::arg("stream"),
        "Mark the nodes of every face owned by exactly one tet, counted over the node2corner "
        "incidence CSR. is_boundary is only ever set, so it arrives zeroed.");

    m.def(
        "spr_weights",
        [](const Rows3d& nodes_mm, const Tets& tet_nodes, const f32_1d& vols, const i32_1d& ptr,
           const i32_1d& idx, const std::optional<i32_1d>& slot_node,
           const std::optional<i32_1d>& is_boundary, const f32_1d& w,
           const std::optional<i32_1d>& n_fallback, uintptr_t stream) {
            const std::size_t n_slots = ptr.shape(0) - 1;
            require(w.shape(0) == idx.shape(0), "spr_weights: w must have one entry per corner");
            require(!slot_node || slot_node->shape(0) == n_slots,
                    "spr_weights: slot_node must have one entry per slot");
            launch_spr_weights(nodes_mm.data(), tet_nodes.data(), vols.data(), ptr.data(),
                               idx.data(), data_or_null(slot_node), data_or_null(is_boundary),
                               w.data(), data_or_null(n_fallback), static_cast<int>(n_slots),
                               as_stream(stream));
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
        [](const Rows3d& nodes_mm, const Tris& tris, const i32_1d& node_ptr, const i32_1d& node_idx,
           const Rows3d& out, uintptr_t stream) {
            const std::size_t n_tri = tris.shape(0);
            require(out.shape(0) == n_tri, "skin_normals: tris and out must be (n_tri, 3)");
            require(node_idx.shape(0) == 3 * n_tri,
                    "skin_normals: node_idx must hold every corner");
            launch_skin_normals(nodes_mm.data(), tris.data(), node_ptr.data(), node_idx.data(),
                                out.data(), static_cast<int>(n_tri),
                                static_cast<int>(node_ptr.shape(0)) - 1, as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("nodes_mm").noconvert(), nb::arg("tris").noconvert(),
        nb::arg("node_ptr").noconvert(), nb::arg("node_idx").noconvert(),
        nb::arg("out").noconvert(), nb::arg("stream"),
        "Smoothed outward skin-triangle normals. node_ptr/node_idx is a node-to-corner CSR over "
        "the triangles, and its order is the order every node's smoothing sums in.");

    m.def(
        "ray_first_hit",
        [](const Rows3d& origins, const Rows3d& directions, const TriVerts& verts,
           const f64_1d& out_t, double min_t, uintptr_t stream) {
            const std::size_t n_rays = origins.shape(0);
            require(directions.shape(0) == n_rays,
                    "ray_first_hit: origins and directions must both be (n_rays, 3)");
            require(out_t.shape(0) == n_rays, "ray_first_hit: out_t must have one entry per ray");
            launch_ray_first_hit(origins.data(), directions.data(), verts.data(), out_t.data(),
                                 static_cast<int>(n_rays), static_cast<int>(verts.shape(0)),
                                 min_t, as_stream(stream));
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
        [](const Rows3d& nodes_mm, const i32_1d& pptr, const i32_1d& pidx, const i32_1d& slot_node,
           const Rows3f& w, const i32_1d& status, uintptr_t stream,
           const std::optional<i32_1d>& active) {
            const std::size_t n_slots = pptr.shape(0) - 1;
            require(w.shape(0) == pidx.shape(0), "hpr_weights: w must be (nnz, 3)");
            require(slot_node.shape(0) == n_slots && status.shape(0) == n_slots,
                    "hpr_weights: slot_node/status must be per slot");
            launch_hpr_weights(nodes_mm.data(), pptr.data(), pidx.data(), slot_node.data(),
                               w.data(), status.data(), data_or_null(active),
                               static_cast<int>(active ? active->shape(0) : n_slots),
                               as_stream(stream));
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
        [](const f64_2d& v_block, const Rows3f& w, const i32_1d& pptr, const i32_1d& pidx,
           const i32_1d& slot_node, const Interleaved& dadt_nodes,
           const std::vector<Rows3f>& e_slots, const f32_1d& work, uintptr_t stream,
           const std::optional<i32_1d>& active) {
            const int k = block_width(e_slots.size(), "hpr_grad: bad list sizes");
            const int stride = static_cast<int>(v_block.shape(1));
            require(stride >= k, "hpr_grad: v_block has fewer columns than placements");
            const std::size_t n_nodes = v_block.shape(0);
            const std::size_t n_slots = pptr.shape(0) - 1;
            const std::size_t kd = dadt_nodes.shape(1);
            require(dadt_nodes.shape(0) == n_nodes &&
                        (kd == static_cast<std::size_t>(k) || kd == 1),
                    "hpr_grad: dadt_nodes must be (n_nodes, k or 1, 3)");
            require(work.shape(0) >= hpr_grad_work_floats(static_cast<int>(n_nodes), k),
                    "hpr_grad: work is smaller than hpr_grad_work_size");
            const auto out = pack<float*>(e_slots, k, n_slots,
                                          "hpr_grad: e_slots must be k (n_slots, 3)");
            launch_hpr_grad(v_block.data(), w.data(), pptr.data(), pidx.data(), slot_node.data(),
                            dadt_nodes.data(), static_cast<std::int64_t>(3 * kd), kd == 1 ? 0 : 3,
                            out.data(), static_cast<int>(n_slots), data_or_null(active),
                            static_cast<int>(active ? active->shape(0) : n_slots), k, stride,
                            static_cast<int>(n_nodes), work.data(), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("v_block").noconvert(), nb::arg("w").noconvert(), nb::arg("pptr").noconvert(),
        nb::arg("pidx").noconvert(), nb::arg("slot_node").noconvert(),
        nb::arg("dadt_nodes").noconvert(), nb::arg("e_slots").noconvert(),
        nb::arg("work").noconvert(), nb::arg("stream"), nb::arg("active").noconvert() = nb::none(),
        "Recovered E on slots from the nodal potential: E = -grad_v - dA/dt, with dA/dt taken "
        "exactly at the node rather than averaged over an element. dadt_nodes is interleaved "
        "(n_nodes, k, 3), or (n_nodes, 1, 3) for one field every placement shares; work holds "
        "hpr_grad_work_size float32s.");

    m.def(
        "hpr_grad_work_size",
        [](int n_nodes, int k) { return hpr_grad_work_floats(n_nodes, k); },
        nb::arg("n_nodes"), nb::arg("k"), "float32s of work hpr_grad needs.");

    m.def(
        "recover_nodes",
        [](const std::vector<Rows3f>& e_in, const f32_1d& w, const i32_1d& ptr, const i32_1d& idx,
           const std::vector<Rows3f>& e_slots, uintptr_t stream) {
            const int k = block_width(e_in.size(), "recover_nodes: bad list sizes");
            const std::size_t n_slots = ptr.shape(0) - 1;
            const auto in = pack<const float*>(e_in, k, e_in[0].shape(0),
                                               "recover_nodes: e_in must be k (n_tet, 3)");
            const auto out = pack<float*>(e_slots, k, n_slots,
                                          "recover_nodes: e_slots must be k (n_slots, 3)");
            launch_recover_nodes(in.data(), w.data(), ptr.data(), idx.data(), out.data(),
                                 static_cast<int>(n_slots), k, as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("e_in").noconvert(), nb::arg("w").noconvert(), nb::arg("ptr").noconvert(),
        nb::arg("idx").noconvert(), nb::arg("e_slots").noconvert(), nb::arg("stream"),
        "Patch gather E*[s] = sum_c w[c] E[c>>2] for <=8 placements; the operator is read once "
        "for the whole block and a column's result does not depend on k.");

    m.def(
        "recover_elements",
        [](const std::vector<Rows3f>& e_slots, const i32_1d& slot_of_corner,
           const std::vector<Rows3f>& e_out, const std::vector<f32_1d>& magn_out, uintptr_t stream,
           const std::optional<i32_1d>& active) {
            const int k = block_width(e_slots.size(), "recover_elements: bad list sizes");
            const std::size_t n_tet = slot_of_corner.shape(0) / 4;
            const char* bad = "recover_elements: e_out and magn_out must be k arrays of n_tet";
            const auto in = pack<const float*>(e_slots, k, e_slots[0].shape(0),
                                               "recover_elements: e_slots must be k (n_slots, 3)");
            const auto eo = optional_pack(e_out, k, n_tet, bad);
            const auto mo = optional_pack(magn_out, k, n_tet, bad);
            launch_recover_elements(in.data(), slot_of_corner.data(), eo.data(), mo.data(),
                                    static_cast<int>(n_tet), data_or_null(active),
                                    static_cast<int>(active ? active->shape(0) : n_tet), k,
                                    as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("e_slots").noconvert(), nb::arg("slot_of_corner").noconvert(),
        nb::arg("e_out").noconvert(), nb::arg("magn_out").noconvert(), nb::arg("stream"),
        nb::arg("active").noconvert() = nb::none(),
        "Sample the recovered field back at the barycentres and take its magnitude. An empty "
        "e_out or magn_out is not written.");

    m.def(
        "element_weight",
        [](const f64_1d& values, const Tets& tet_nodes, const TetGradf& g, const f32_1d& neg_vc,
           const Rows3d& w_e, uintptr_t stream) {
            require(w_e.shape(0) == tet_nodes.shape(0),
                    "element_weight: w_e must have one row per tet");
            launch_element_weight(values.data(), tet_nodes.data(), g.data(), neg_vc.data(),
                                  w_e.data(), static_cast<int>(tet_nodes.shape(0)),
                                  as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("values").noconvert(), nb::arg("tet_nodes").noconvert(), nb::arg("g").noconvert(),
        nb::arg("neg_vc").noconvert(), nb::arg("w_e").noconvert(), nb::arg("stream"),
        "Per-element reciprocity weight w_e = vol*sigma*(G_e values); into caller-allocated w_e.");

    m.def(
        "node_gather",
        [](const f64_1d& corner, const i32_1d& ptr, const i32_1d& idx, const f64_1d& out,
           uintptr_t stream) {
            const std::size_t n_nodes = ptr.shape(0) - 1;
            require(out.shape(0) == n_nodes, "node_gather: out must have one entry per node");
            launch_node_gather(corner.data(), ptr.data(), idx.data(), out.data(),
                               static_cast<int>(n_nodes), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("corner").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("out").noconvert(), nb::arg("stream"),
        "Node-centric scalar corner gather over the node2corner CSR; into caller-allocated out.");

    m.def(
        "node_gather3",
        [](const Rows3d& w_e, const i32_1d& ptr, const i32_1d& idx, const Rows3d& node_w,
           uintptr_t stream) {
            const std::size_t n_nodes = ptr.shape(0) - 1;
            require(node_w.shape(0) == n_nodes,
                    "node_gather3: node_w must have one row per node");
            launch_node_gather3(w_e.data(), ptr.data(), idx.data(), node_w.data(),
                                 static_cast<int>(n_nodes), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("w_e").noconvert(), nb::arg("ptr").noconvert(), nb::arg("idx").noconvert(),
        nb::arg("node_w").noconvert(), nb::arg("stream"),
        "Node-centric 3-vector corner gather with 1/4 weight; into caller-allocated node_w.");

    m.def(
        "adm_evaluate",
        [](const Frames& frames, const Rows3f& coil_pos, const Rows3f& coil_mom, const f32_1d& q,
           const u8_1d& cell_ok, const i32_1d& cell_slot, const f32_1d& q_near,
           const i32_1d& list_ptr, const f32_1d& entries, double mu0_4pi,
           const std::vector<double>& origin, const std::vector<double>& spacing,
           const std::vector<int>& shape, int n_dir, double didt, const f64_2d& out,
           const i32_1d& fault, uintptr_t stream) {
            const GridArgs g = isotropic_grid_args(origin, spacing, shape, "adm_evaluate");
            const std::size_t n_frames = frames.shape(0);
            const std::int64_t n_cells =
                static_cast<std::int64_t>(shape[0] - 1) * (shape[1] - 1) * (shape[2] - 1);
            require(coil_mom.shape(0) == coil_pos.shape(0) &&
                        static_cast<std::int64_t>(q.shape(0)) == g.points() * n_dir * 3 &&
                        static_cast<std::int64_t>(cell_ok.shape(0)) == n_cells &&
                        static_cast<std::int64_t>(cell_slot.shape(0)) == n_cells &&
                        out.shape(0) == n_frames && static_cast<int>(out.shape(1)) == n_dir &&
                        fault.shape(0) == 2,
                    "adm_evaluate: shape mismatch");
            require_slots(list_ptr, list_ptr.shape(0) - 1, q_near, entries, n_dir,
                          "adm_evaluate");
            launch_adm_evaluate(frames.data(), static_cast<int>(n_frames), coil_pos.data(),
                                coil_mom.data(), static_cast<int>(coil_pos.shape(0)), q.data(),
                                cell_ok.data(), cell_slot.data(), q_near.data(), list_ptr.data(),
                                entries.data(), mu0_4pi, g.origin, g.spacing, g.shape, n_dir,
                                didt, out.data(), fault.data(), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("frames").noconvert(), nb::arg("coil_pos").noconvert(),
        nb::arg("coil_mom").noconvert(), nb::arg("q").noconvert(), nb::arg("cell_ok").noconvert(),
        nb::arg("cell_slot").noconvert(), nb::arg("q_near").noconvert(),
        nb::arg("list_ptr").noconvert(), nb::arg("entries").noconvert(), nb::arg("mu0_4pi"),
        nb::arg("origin"), nb::arg("spacing"), nb::arg("shape"), nb::arg("n_dir"),
        nb::arg("didt"), nb::arg("out").noconvert(), nb::arg("fault").noconvert(),
        nb::arg("stream"),
        "Target E-vectors (n_frames, n_dir) of coil frames against a sampled reciprocity field. "
        "A cell with a slot (cell_slot >= 0) interpolates its own corners q_near and adds the "
        "sources list_ptr delimits in entries (adm_entry_floats(n_dir) floats each: position in "
        "grid index units, then the weights) directly. fault (2 ints the caller sets to "
        "n_frames) receives the lowest candidate off the grid and the lowest reading an "
        "unsampled cell.");

    m.def(
        "adm_sample",
        [](const Rows3f& src, const SetRows3f& src_w, const Dev<int32_t, -1, 6>& tile_box,
           const i64_1d& points, double mu0_4pi, const std::vector<double>& origin,
           const std::vector<double>& spacing, const std::vector<int>& shape, const f32_1d& q,
           uintptr_t stream) {
            const GridArgs g = isotropic_grid_args(origin, spacing, shape, "adm_sample");
            const int n_dir = static_cast<int>(src_w.shape(0));
            require(static_cast<std::int64_t>(q.shape(0)) == g.points() * n_dir * 3,
                    "adm_sample: q must be (nx * ny * nz * n_dir * 3,)");
            require_sources(src, src_w, n_dir, "adm_sample");
            const std::size_t tile = static_cast<std::size_t>(adm_sample_tile());
            require(tile_box.shape(0) == (src.shape(0) + tile - 1) / tile,
                    "adm_sample: tile_box must have one row per source tile");
            launch_adm_sample(src.data(), src_w.data(), static_cast<int>(src.shape(0)), n_dir,
                              tile_box.data(), points.data(), static_cast<int>(points.shape(0)),
                              mu0_4pi, g.origin, g.spacing, g.shape, q.data(), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("src").noconvert(), nb::arg("src_w").noconvert(),
        nb::arg("tile_box").noconvert(), nb::arg("points").noconvert(), nb::arg("mu0_4pi"),
        nb::arg("origin"), nb::arg("spacing"), nb::arg("shape"), nb::arg("q").noconvert(),
        nb::arg("stream"),
        "Samples the reciprocity field at the sorted flat grid indices points into q, each point "
        "from every source (grid index units, sorted by cell) outside its eight cells. tile_box "
        "holds each adm_sample_tile()-source tile's floor(src) box, lo xyz then hi xyz.");

    m.def("adm_sample_tile", &adm_sample_tile, "Sources per adm_sample tile.");

    m.def(
        "adm_q_points",
        [](const Rows3f& src, const SetRows3f& src_w, const Rows3f& points, double scale,
           const f32_2d& out, uintptr_t stream) {
            const int n_dir = static_cast<int>(src_w.shape(0));
            require_sources(src, src_w, n_dir, "adm_q_points");
            require(out.shape(0) == points.shape(0) &&
                        static_cast<int>(out.shape(1)) == 3 * n_dir,
                    "adm_q_points: out must be (n_points, 3 * n_dir)");
            launch_adm_q_points(src.data(), src_w.data(), static_cast<int>(src.shape(0)), n_dir,
                                points.data(), static_cast<int>(points.shape(0)), scale,
                                out.data(), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("src").noconvert(), nb::arg("src_w").noconvert(), nb::arg("points").noconvert(),
        nb::arg("scale"), nb::arg("out").noconvert(), nb::arg("stream"),
        "out (n_points, 3 * n_dir) = scale * sum over every source of W x (p - s) / |p - s|^3, "
        "with src, src_w (n_dir, n_src, 3) and points in one frame and unit.");

    m.def(
        "adm_slot_corners",
        [](const i64_1d& slot_cells, int first_slot, const i32_1d& list_ptr,
           const f32_1d& entries, int n_dir, const f32_1d& q, double mu0_4pi,
           const std::vector<double>& origin, const std::vector<double>& spacing,
           const std::vector<int>& shape, const f32_1d& q_near, uintptr_t stream) {
            const GridArgs g = isotropic_grid_args(origin, spacing, shape, "adm_slot_corners");
            const std::size_t n_slots = slot_cells.shape(0);
            require(static_cast<std::int64_t>(q.shape(0)) == g.points() * n_dir * 3 &&
                        first_slot >= 0 && static_cast<std::size_t>(first_slot) <= n_slots,
                    "adm_slot_corners: shape mismatch");
            require_slots(list_ptr, n_slots, q_near, entries, n_dir, "adm_slot_corners");
            launch_adm_slot_corners(slot_cells.data(), first_slot, static_cast<int>(n_slots),
                                    list_ptr.data(), entries.data(), n_dir, q.data(), mu0_4pi,
                                    g.origin, g.spacing, g.shape, q_near.data(),
                                    as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("slot_cells").noconvert(), nb::arg("first_slot"), nb::arg("list_ptr").noconvert(),
        nb::arg("entries").noconvert(), nb::arg("n_dir"), nb::arg("q").noconvert(),
        nb::arg("mu0_4pi"), nb::arg("origin"), nb::arg("spacing"), nb::arg("shape"),
        nb::arg("q_near").noconvert(), nb::arg("stream"),
        "Fills q_near (n_slots, 8, n_dir, 3) from first_slot on: each slot cell's sampled corners "
        "less its entries' sources that the corner's own sample kept.");

    m.def("adm_entry_floats", &adm_entry_floats, nb::arg("n_dir"),
          "Floats per packed ADM slot entry for n_dir weight sets.");

    m.def(
        "adm_mark",
        [](const Frames& frames, const Rows3f& coil_pos, const std::vector<double>& origin,
           const std::vector<double>& spacing, const std::vector<int>& shape, const u8_1d& need,
           uintptr_t stream) {
            const GridArgs g = grid_args(origin, spacing, shape);
            require(static_cast<std::int64_t>(need.shape(0)) == g.points(),
                    "adm_mark: need must have one byte per grid point");
            launch_adm_mark(frames.data(), static_cast<int>(frames.shape(0)), coil_pos.data(),
                            static_cast<int>(coil_pos.shape(0)), g.origin, g.spacing, g.shape,
                            need.data(), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("frames").noconvert(), nb::arg("coil_pos").noconvert(), nb::arg("origin"),
        nb::arg("spacing"), nb::arg("shape"), nb::arg("need").noconvert(), nb::arg("stream"),
        "Sets need to 1 at every grid point a frame's dipoles interpolate from.");

    m.def(
        "adm_reach",
        [](const std::vector<double>& origin, const std::vector<double>& spacing,
           const std::vector<int>& shape, const Rows3d& origins, const Rows3d& axes,
           const u8_2d& profile, double h0, double cell, const u8_1d& mask, uintptr_t stream) {
            const GridArgs g = grid_args(origin, spacing, shape);
            require(origins.shape(0) == axes.shape(0) &&
                        static_cast<std::int64_t>(mask.shape(0)) == g.points(),
                    "adm_reach: shape mismatch");
            launch_adm_reach(g.origin, g.spacing, g.shape, origins.data(), axes.data(),
                             static_cast<int>(origins.shape(0)), profile.data(),
                             static_cast<int>(profile.shape(0)), static_cast<int>(profile.shape(1)),
                             h0, cell, mask.data(), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("origin"), nb::arg("spacing"), nb::arg("shape"), nb::arg("origins").noconvert(),
        nb::arg("axes").noconvert(), nb::arg("profile").noconvert(), nb::arg("h0"),
        nb::arg("cell"), nb::arg("mask").noconvert(), nb::arg("stream"),
        "Marks the grid points a coil's swept (rho, h) profile reaches about any centre's axis.");

    m.def(
        "clear_cuda_error", [] { (void)cudaGetLastError(); },
        nb::call_guard<nb::gil_scoped_release>(),
        "Reset the calling thread's last CUDA error, which a failed allocation leaves set.");
    m.def(
        "host_copy_async",
        [](uintptr_t src, uintptr_t dst, std::size_t bytes, uintptr_t stream) {
            auto copy = std::make_unique<HostCopy>(
                HostCopy{reinterpret_cast<const void*>(src), reinterpret_cast<void*>(dst), bytes});
            check_cuda(cudaLaunchHostFunc(as_stream(stream), run_host_copy, copy.get()),
                       "host_copy_async", "launch host func");
            (void)copy.release();
        },
        nb::call_guard<nb::gil_scoped_release>(), nb::arg("src"), nb::arg("dst"),
        nb::arg("bytes"), nb::arg("stream"),
        "Copy bytes from src to dst (host pointers) once the stream reaches this point. The "
        "caller keeps both buffers alive until then.");

    m.attr("SUMMARY_LAYOUT") = nb::make_tuple(kSummaryPeak, kSummaryPeakLocation,
                                              kSummaryCenterOfGravity, kSummaryMean, kSummaryStd,
                                              kSummaryQuantiles);
    m.def(
        "region_summary_work_size",
        [](int n, int k, bool f64) { return region_summary_work_size(n, k, f64); },
        nb::call_guard<nb::gil_scoped_release>(), nb::arg("n"), nb::arg("k"), nb::arg("f64"),
        "Bytes of workspace region_summary needs.");
    m.def(
        "region_summary",
        [](const std::vector<nb::ndarray<nb::ndim<1>, nb::c_contig, nb::device::cuda>>& fields,
           const i32_1d& index, const f64_1d& w, const Rows3d& bary, const std::vector<double>& qs,
           int anchor, const std::vector<double>& fracs, const u8_1d& work, const f64_2d& out,
           uintptr_t stream) {
            const int k = block_width(fields.size(), "region_summary: between 1 and 8 fields");
            const std::size_t n = index.shape(0);
            const bool f64 = fields[0].dtype() == nb::dtype<double>();
            const auto ptrs = pack<const void*>(
                fields, k, fields[0].shape(0),
                "region_summary: fields must share one length and be float32 or float64");
            for (const auto& f : fields) {
                require(f.dtype() == (f64 ? nb::dtype<double>() : nb::dtype<float>()),
                        "region_summary: fields must share one length and be float32 or float64");
            }
            const int n_q = static_cast<int>(qs.size());
            const int n_f = static_cast<int>(fracs.size());
            require(w.shape(0) == n && bary.shape(0) == n && static_cast<int>(out.shape(0)) == k &&
                        static_cast<int>(out.shape(1)) == kSummaryQuantiles + n_q + n_f,
                    "region_summary: shape mismatch");
            launch_region_summary(ptrs.data(), f64, k, static_cast<int>(n), index.data(),
                                  w.data(), bary.data(), qs.data(), n_q, anchor, fracs.data(),
                                  n_f, work.data(), work.shape(0), out.data(), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(), nb::arg("fields").noconvert(),
        nb::arg("index").noconvert(), nb::arg("w").noconvert(), nb::arg("bary").noconvert(),
        nb::arg("qs"), nb::arg("anchor"), nb::arg("fracs"), nb::arg("work").noconvert(),
        nb::arg("out").noconvert(), nb::arg("stream"),
        "Volume-weighted metrics of up to 8 fields over one region; see fem/fem.hpp for the "
        "layout of out.");

    m.def(
        "accumulate_moments_block",
        [](const std::vector<f32_1d>& magn, const f64_1d& sum_e, const f64_1d& sumsq_e,
           uintptr_t stream) {
            const int k = block_width(magn.size(), "accumulate_moments_block: bad sizes");
            const std::size_t n = sum_e.shape(0);
            require(sumsq_e.shape(0) == n, "accumulate_moments_block: bad sizes");
            const auto ptrs = pack<const float*>(magn, k, n, "accumulate_moments_block: bad sizes");
            launch_accumulate_moments_block(ptrs.data(), k, sum_e.data(), sumsq_e.data(),
                                            static_cast<int>(n), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("magn").noconvert(), nb::arg("sum_e").noconvert(), nb::arg("sumsq_e").noconvert(),
        nb::arg("stream"),
        "Streaming |E| moments for up to 8 draws, folded in the order given: sum_e += magn and "
        "sumsq_e += magn^2, in place.");

    m.def(
        "row_dot",
        [](const f64_2d& a, const f64_2d& s, const f64_2d& out, uintptr_t stream) {
            const std::size_t n = a.shape(0);
            const std::size_t p = a.shape(1);
            const std::size_t k = s.shape(0);
            require(s.shape(1) == p && out.shape(0) == n && out.shape(1) == k,
                    "row_dot: a must be (n, p), s (k, p) and out (n, k)");
            launch_row_dot(a.data(), s.data(), out.data(), static_cast<int>(n),
                           static_cast<int>(p), static_cast<int>(k), as_stream(stream));
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
        [](const f32_2d& q, const f64_2d& b, const f64_1d& work, const f64_2d& x0,
           uintptr_t stream) {
            const int n = static_cast<int>(q.shape(0));
            const int m_basis = static_cast<int>(q.shape(1));
            const int k = static_cast<int>(b.shape(1));
            require(static_cast<int>(b.shape(0)) == n && static_cast<int>(x0.shape(0)) == n &&
                        static_cast<int>(x0.shape(1)) == k &&
                        static_cast<std::int64_t>(work.shape(0)) >=
                            sweep_seed_work_size(n, m_basis, k),
                    "sweep_seed: q (n, m), b and x0 (n, k), work sweep_seed_work_size(n, m, k)");
            launch_sweep_seed(q.data(), b.data(), n, m_basis, k, work.data(), x0.data(),
                              as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("q").noconvert(), nb::arg("b").noconvert(), nb::arg("work").noconvert(),
        nb::arg("x0").noconvert(), nb::arg("stream"),
        "x0 = q (q^T b) per column for an A-orthonormal fp32 basis; a column's result does not "
        "depend on k.");

    m.def(
        "place_transforms",
        [](const Rows3d& centers, const std::optional<Rows3d>& handles, const f64_1d& dists,
           const TriVerts& verts, const Rows3d& tnorm, const Frames& out,
           const std::optional<i32_1d>& degenerate, uintptr_t stream) {
            const std::size_t n_pl = centers.shape(0);
            const std::size_t n_tri = verts.shape(0);
            require((!handles || handles->shape(0) == n_pl) && dists.shape(0) == n_pl &&
                        out.shape(0) == n_pl && (!degenerate || degenerate->shape(0) == n_pl),
                    "place_transforms: handles, dists, out and degenerate must be per placement");
            require(tnorm.shape(0) == n_tri,
                    "place_transforms: verts and tnorm must be per triangle");
            launch_place(centers.data(), data_or_null(handles), dists.data(), verts.data(),
                         tnorm.data(), out.data(), data_or_null(degenerate),
                         static_cast<int>(n_pl), static_cast<int>(n_tri), as_stream(stream));
        },
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("centers").noconvert(), nb::arg("handles").noconvert(),
        nb::arg("dists").noconvert(), nb::arg("verts").noconvert(),
        nb::arg("tnorm").noconvert(), nb::arg("out").noconvert(),
        nb::arg("degenerate").noconvert(), nb::arg("stream"),
        "Batched closest-point scalp projection + coil frame; writes (P, 4, 4) row-major out "
        "and a (P,) flag marking placements whose handle left the in-plane axis undefined, or "
        "whose centre was not finite. Pass handles=None for the projection and normal alone, "
        "with degenerate=None.");
}
