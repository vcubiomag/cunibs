#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <concepts>
#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

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

// An argument the call only reads. Its data is const, so it cannot be handed to a pointer a
// kernel writes through.
template <typename A>
struct ReadOnly;
template <typename T, typename... Rest>
struct ReadOnly<nb::ndarray<T, Rest...>> {
    using type = nb::ndarray<const T, Rest...>;
};
template <typename A>
using In = typename ReadOnly<A>::type;

using f64_1d = Dev<double, -1>;
using f64_2d = Dev<double, -1, -1>;
using f32_1d = Dev<float, -1>;
using f32_2d = Dev<float, -1, -1>;
using i32_1d = Dev<int32_t, -1>;
using i64_1d = Dev<int64_t, -1>;
using u8_1d = Dev<uint8_t, -1>;
using u8_2d = Dev<uint8_t, -1, -1>;
using Rows3d = Dev<double, -1, 3>;
using Rows3f = Dev<float, -1, 3>;
using Rows4f = Dev<float, -1, 4>;
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
template <int Ndim>
using Host = nb::ndarray<const double, nb::ndim<Ndim>, nb::c_contig, nb::device::cpu>;

// The caller's stream, passed from Python as its integer handle.
struct Stream {
    cudaStream_t handle;
    operator cudaStream_t() const { return handle; }
};

namespace nanobind::detail {
template <>
struct type_caster<Stream> {
    NB_TYPE_CASTER(Stream, const_name("int"))

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        make_caster<std::uintptr_t> ptr;
        if (!ptr.from_python(src, flags, cleanup)) return false;
        value.handle = reinterpret_cast<cudaStream_t>(ptr.value);
        return true;
    }
};
}  // namespace nanobind::detail

namespace {

// The message is only built for a check that fails.
template <typename... Parts>
void require(bool ok, const Parts&... what) {
    if (!ok) throw std::invalid_argument((std::string{} + ... + what));
}

// Every kernel indexes in int, so a count past its range is refused rather than wrapped.
template <std::integral T>
int narrow(T n) {
    require(std::in_range<int>(n), "an array dimension exceeds the int range the kernels index in");
    return static_cast<int>(n);
}

template <typename A>
int extent(const A& a, std::size_t axis = 0) {
    return narrow(a.shape(axis));
}

template <typename T>
auto data_or_null(const std::optional<T>& a) -> decltype(a->data()) {
    return a ? a->data() : nullptr;
}

// A failed launch reports itself only through the thread's last error: no synchronise returns it.
// So a native call reads it on the way out, before the caller can use what the launch was to write.
// It starts cleared, so that an error left by CuPy or the caller, such as a failed allocation, is
// not reported as this call's.
class NativeCall {
public:
    NativeCall() : exceptions_(std::uncaught_exceptions()) { (void)cudaGetLastError(); }
    ~NativeCall() noexcept(false) {
        if (std::uncaught_exceptions() == exceptions_) {
            check_cuda(cudaGetLastError(), "native", "launch");
        }
    }

private:
    nb::gil_scoped_release release_;
    int exceptions_;
};

int block_width(std::size_t k, const char* what) {
    require(k >= 1 && k <= static_cast<std::size_t>(kMaxStageBlock), what);
    return static_cast<int>(k);
}

constexpr nb::call_guard<NativeCall> native_call;

// A device array argument: never converted, since a copy would not be what the kernel writes to.
auto dev(const char* name) { return nb::arg(name).noconvert(); }

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

// NaN compares false, so a NaN tolerance would otherwise run every solve to its iteration limit.
void require_tolerance(double tolerance) {
    require(tolerance >= 0.0, "tolerance must be non-negative");
}

// The same for an output a caller may skip: an empty list leaves every pointer null.
template <typename Array>
std::array<float*, kMaxStageBlock> optional_pack(const std::vector<Array>& arrays, int k,
                                                 std::size_t rows, const char* what) {
    if (arrays.empty()) return {};
    return pack<float*>(arrays, k, rows, what);
}

using GridVec = std::array<double, 3>;
using GridShape = std::array<int, 3>;

GridGeom grid_args(const GridVec& origin, const GridVec& spacing, const GridShape& shape) {
    GridGeom g;
    for (int i = 0; i < 3; ++i) {
        require(shape[i] >= 2 && spacing[i] > 0.0,
                "grid needs at least 2 samples and a positive spacing");
        g.origin[i] = origin[i];
        g.spacing[i] = spacing[i];
        g.n[i] = shape[i];
    }
    return g;
}

// The ADM sampling and near-source kernels work in grid index units and scale by 1 / spacing^2,
// which needs the same spacing on every axis.
GridGeom isotropic_grid_args(const GridVec& origin, const GridVec& spacing,
                             const GridShape& shape, const char* what) {
    const GridGeom g = grid_args(origin, spacing, shape);
    require(g.spacing[1] == g.spacing[0] && g.spacing[2] == g.spacing[0], what,
            ": the grid spacing must be equal on every axis");
    return g;
}

void require_sources(const auto& src, const auto& src_w, int n_dir, const char* what) {
    require(extent(src_w) == n_dir && src_w.shape(1) == src.shape(0), what,
            ": src_w must be (n_dir, n_src, 3)");
}

void require_slots(const auto& list_ptr, std::size_t n_slots, const auto& q_near,
                   const auto& entries, int n_dir, const char* what) {
    const std::size_t width = static_cast<std::size_t>(adm_entry_floats(n_dir));
    require(list_ptr.shape(0) == n_slots + 1 && q_near.shape(0) == n_slots * 8 * n_dir * 3 &&
                entries.shape(0) % width == 0 &&
                reinterpret_cast<std::uintptr_t>(entries.data()) % 16 == 0,
            what, ": list_ptr, q_near and entries must match the slots");
}

// cuda.core is imported on the first failure rather than with the extension, since it adds tens of
// milliseconds to the import of a package that only a multi-chunk sweep otherwise needs.
void translate_out_of_memory(const std::exception_ptr& p, void*) {
    try {
        std::rethrow_exception(p);
    } catch (const CudaOutOfMemory& e) {
        const nb::object type = nb::module_::import_("cuda.core").attr("CUDAOutOfMemoryError");
        PyErr_SetString(type.ptr(), e.what());
    }
}

using BlockResult = std::tuple<int, std::vector<double>, std::vector<int>>;

BlockResult as_tuple(PcgBlockResult r) {
    return {r.iterations, std::move(r.relative_residual), std::move(r.column_iterations)};
}

}  // namespace

NB_MODULE(_solver_ext, m) {
    nb::register_exception_translator(translate_out_of_memory);
    // Compiled-in limits, exported so the Python layer never restates them.
    m.attr("MAX_STAGE_BLOCK") = kMaxStageBlock;
    m.attr("BLOCK_SIZES") =
        std::apply([](auto... w) { return nb::make_tuple(w...); }, kBlockWidths);

    nb::class_<NativeVCycle>(m, "NativeVCycle")
        .def(nb::init<>())
        .def("set_smoother", &NativeVCycle::set_smoother, nb::arg("degree"), nb::arg("lower_ratio"),
             "Set the Chebyshev degree and the lower end of its interval, as a divisor of the "
             "smoother's spectral upper bound. Must precede the first add_level.")
        .def(
            "add_level",
            [](NativeVCycle& self, const In<i32_1d>& row_ptr, const In<i32_1d>& col_idx,
               const In<f32_1d>& values, const In<f32_1d>& dinv, const In<i32_1d>& p_row_ptr,
               const In<i32_1d>& p_col_idx, const In<f32_1d>& p_values, const In<i32_1d>& r_row_ptr,
               const In<i32_1d>& r_col_idx, const In<f32_1d>& r_values, Stream stream) {
                const int n_rows = extent(row_ptr) - 1;
                require(col_idx.shape(0) == values.shape(0) &&
                            p_col_idx.shape(0) == p_values.shape(0) &&
                            r_col_idx.shape(0) == r_values.shape(0),
                        "each CSR must have one column per value");
                require(dinv.shape(0) + 1 == row_ptr.shape(0), "dinv must have one entry per row");
                require(r_values.shape(0) == p_values.shape(0), "R must be the transpose of P");
                require(p_row_ptr.shape(0) == row_ptr.shape(0),
                        "P must have one row per fine row");
                self.add_level(n_rows, extent(values), extent(r_row_ptr) - 1, extent(p_values),
                               row_ptr.data(), col_idx.data(), values.data(), dinv.data(),
                               p_row_ptr.data(), p_col_idx.data(), p_values.data(),
                               r_row_ptr.data(), r_col_idx.data(), r_values.data(), stream);
            },
            native_call, dev("row_ptr"), dev("col_idx"), dev("values"), dev("dinv"),
            dev("p_row_ptr"), dev("p_col_idx"), dev("p_values"), dev("r_row_ptr"), dev("r_col_idx"),
            dev("r_values"), nb::arg("stream"),
            "Append one non-coarsest level (fp32 CSR, 1/guard(d), prolongator CSR and its "
            "transpose); device arrays are copied into solver-owned buffers.")
        .def(
            "set_coarse",
            [](NativeVCycle& self, const In<f32_2d>& ainv, Stream stream) {
                require(ainv.shape(0) == ainv.shape(1), "coarse inverse must be square");
                self.set_coarse(extent(ainv), ainv.data(), stream);
            },
            native_call, dev("ainv"), nb::arg("stream"),
            "Set the dense (row-major) inverse of the coarsest matrix.")
        .def("finalize", &NativeVCycle::finalize,
             "Validate level chain consistency; required before a solve.")
        .def("n_levels", &NativeVCycle::n_levels,
             "Number of coarsening levels, excluding the coarsest (dense-inverse) one.");

    m.def(
        "select_size4",
        [](const In<i32_1d>& row_ptr, const In<i32_1d>& col_idx, const In<f32_1d>& values,
           const i32_1d& aggregates, Stream stream) {
            const int n_rows = extent(row_ptr) - 1;
            require(extent(aggregates) == n_rows, "aggregates must have one entry per row");
            require(col_idx.shape(0) == values.shape(0),
                    "col_idx and values must have one entry per nonzero");
            return select_size4(n_rows, extent(values), row_ptr.data(), col_idx.data(),
                                values.data(), aggregates.data(), stream);
        },
        native_call, dev("row_ptr"), dev("col_idx"), dev("values"), dev("aggregates"),
        nb::arg("stream"),
        "Unsmoothed pairwise aggregation (AMGx SIZE_4). Fills `aggregates` with a surjective "
        "row -> aggregate map and returns the aggregate count. Synchronises the stream.");

    nb::class_<PcgAmgSolver>(m, "PcgAmgSolver")
        .def(
            "__init__",
            [](PcgAmgSolver* self, const In<i32_1d>& row_ptr, const In<i32_1d>& col_idx,
               const In<f64_1d>& values, Stream stream) {
                require(col_idx.shape(0) == values.shape(0), "CSR must have one column per value");
                new (self) PcgAmgSolver(extent(row_ptr) - 1, extent(values), row_ptr.data(),
                                        col_idx.data(), values.data(), stream);
            },
            native_call, dev("row_ptr"), dev("col_idx"), dev("values"), nb::arg("stream"))
        .def_prop_ro("n", &PcgAmgSolver::n, "Rows of the matrix.")
        .def_prop_ro("nnz", &PcgAmgSolver::nnz, "Nonzeros of the matrix's entry CSR.")
        .def(
            "export_csr",
            [](PcgAmgSolver& self, const i32_1d& row_ptr, const std::optional<i32_1d>& col_idx,
               const std::optional<f64_1d>& values, Stream stream) {
                require(extent(row_ptr) == self.n() + 1 &&
                            (!col_idx || extent(*col_idx) == self.nnz()) &&
                            (!values || extent(*values) == self.nnz()),
                        "export_csr: row_ptr must be (n + 1), col_idx and values (nnz)");
                self.export_csr(row_ptr.data(), data_or_null(col_idx), data_or_null(values),
                                stream);
            },
            native_call, dev("row_ptr"), dev("col_idx"), dev("values"), nb::arg("stream"),
            "Write the matrix's entry CSR into caller arrays; col_idx or values may be None.")
        .def(
            "solve_mixed",
            [](PcgAmgSolver& self, NativeVCycle& preconditioner, const In<f64_1d>& b,
               const f64_1d& x, double tolerance, int max_iters, Stream stream,
               const std::optional<In<f64_1d>>& x0) {
                require_tolerance(tolerance);
                const std::size_t n = self.n();
                require(b.shape(0) == n && x.shape(0) == n && (!x0 || x0->shape(0) == n),
                        "b, x and x0 must have one entry per row");
                const PcgResult r = self.solve_mixed(preconditioner, b.data(), x.data(), tolerance,
                                                     max_iters, stream, data_or_null(x0));
                return std::make_tuple(r.iterations, r.relative_residual);
            },
            native_call, nb::arg("preconditioner"), dev("b"), dev("x"), nb::arg("tolerance"),
            nb::arg("max_iters"), nb::arg("stream"), dev("x0") = nb::none())
        .def(
            "solve_mixed_block",
            [](PcgAmgSolver& self, NativeVCycle& preconditioner, const In<f64_2d>& B,
               const f64_2d& X, double tolerance, int max_iters, Stream stream,
               const std::optional<In<f64_2d>>& X0) {
                require_tolerance(tolerance);
                const std::size_t n = self.n();
                const std::size_t k = B.shape(1);
                require(B.shape(0) == n && X.shape(0) == n && X.shape(1) == k &&
                            (!X0 || (X0->shape(0) == n && X0->shape(1) == k)),
                        "B, X and X0 must all be (n, k)");
                return as_tuple(self.solve_mixed_block(preconditioner, B.data(), X.data(),
                                                       narrow(k), tolerance, max_iters, stream,
                                                       data_or_null(X0)));
            },
            native_call, nb::arg("preconditioner"), dev("B"), dev("X"), nb::arg("tolerance"),
            nb::arg("max_iters"), nb::arg("stream"), dev("X0") = nb::none(),
            "Lockstep k-RHS mixed-precision PCG over row-major (n, k) operands; returns "
            "(block iterations, per-column relative residuals, per-column iteration counts). A "
            "column freezes once its own residual meets tolerance, so it stops where it would "
            "have solved alone. k in {1, 2, 4, 8}.")
        .def(
            "solve_affine_block",
            [](PcgAmgSolver& self, NativeVCycle& preconditioner, AffineTerms& terms,
               const In<f64_2d>& B, const f64_2d& X, const Host<2>& sigma, double tolerance,
               int max_iters, Stream stream, const std::optional<In<f64_2d>>& X0) {
                require_tolerance(tolerance);
                const std::size_t n = self.n();
                const std::size_t k = B.shape(1);
                require(B.shape(0) == n && X.shape(0) == n && X.shape(1) == k &&
                            sigma.shape(0) == k && extent(sigma, 1) == terms.n_terms() &&
                            (!X0 || (X0->shape(0) == n && X0->shape(1) == k)),
                        "B, X and X0 must be (n, k) and sigma (k, n_terms)");
                return as_tuple(self.solve_affine_block(preconditioner, terms, B.data(), X.data(),
                                                        narrow(k), sigma.data(), tolerance,
                                                        max_iters, stream, data_or_null(X0)));
            },
            native_call, nb::arg("preconditioner"), nb::arg("terms"), dev("B"), dev("X"),
            dev("sigma"), nb::arg("tolerance"), nb::arg("max_iters"), nb::arg("stream"),
            dev("X0") = nb::none(),
            "The block solve with column c's operator A(sigma[c]) from `terms`: every column its "
            "own matrix, sharing the parts' reads and the preconditioner. sigma is a host "
            "(k, n_terms) float64 array. X0 may be X itself. Returns what solve_mixed_block "
            "does.");

    nb::class_<AffineTerms>(m, "AffineTerms")
        .def(
            "__init__",
            [](AffineTerms* self, const PcgAmgSolver& solver, const In<f64_2d>& terms,
               Stream stream) {
                require(extent(terms, 1) == solver.nnz(), "affine terms must be (n_terms, nnz)");
                new (self) AffineTerms(solver, extent(terms), terms.data(), stream);
            },
            native_call, nb::keep_alive<1, 2>(), nb::arg("solver"), dev("terms"), nb::arg("stream"),
            "An operator affine in per-column coefficients over solver's pattern, "
            "A(sigma) = sum_t sigma_t A_t, from its parts' values in entry order (n_terms, nnz). "
            "Keeps solver alive.")
        .def_prop_ro("n_terms", &AffineTerms::n_terms)
        .def(
            "values",
            [](const AffineTerms& self, const Host<1>& sigma, const f64_1d& out, Stream stream) {
                require(extent(sigma) == self.n_terms(),
                        "values: sigma must have one entry per term");
                require(extent(out) == self.nnz(), "values: out must have one entry per nonzero");
                self.values(sigma.data(), out.data(), stream);
            },
            native_call, dev("sigma"), dev("out"), nb::arg("stream"),
            "out = sum_t sigma[t] A_t in the solver's entry order, for host sigma (n_terms,).")
        .def(
            "matmul",
            [](const AffineTerms& self, int term, const In<f64_2d>& x, const f64_2d& y,
               Stream stream) {
                require(
                    extent(x) == self.n() && y.shape(0) == x.shape(0) && y.shape(1) == x.shape(1),
                    "matmul: x and y must both be (n, m)");
                self.matmul(term, x.data(), y.data(), extent(x, 1), stream);
            },
            native_call, nb::arg("term"), dev("x"), dev("y"), nb::arg("stream"),
            "y = A_term x for row-major (n, m) float64 x, each row summed over the part's own "
            "nonzeros in column order with one fused multiply-add per entry.");

    m.def(
        "p1_gradients",
        [](const In<Rows3d>& nodes_mm, const In<Tets>& tet_nodes, const TetGradd& g,
           const f64_1d& vols, Stream stream) {
            const std::size_t n_tet = tet_nodes.shape(0);
            require(g.shape(0) == n_tet && vols.shape(0) == n_tet,
                    "p1_gradients: g and vols must have one entry per tet");
            launch_p1_gradients(nodes_mm.data(), tet_nodes.data(), g.data(), vols.data(),
                                narrow(n_tet), stream);
        },
        native_call, dev("nodes_mm"), dev("tet_nodes"), dev("g"), dev("vols"), nb::arg("stream"),
        "P1 basis-function gradients (n_tet, 4, 3) in 1/m and element volumes in m^3, from node "
        "coordinates in millimetres; writes into caller-allocated g/vols.");

    m.def(
        "dadt_block",
        [](const In<Rows3d>& pos_m, const In<Rows3d>& mom, const In<Frames>& transforms,
           const In<Rows3d>& nodes_mm, const f32_2d& out, float didt, Stream stream, int column) {
            const std::size_t n_nodes = nodes_mm.shape(0);
            const int k = extent(transforms);
            const int stride = extent(out, 1);
            require(mom.shape(0) == pos_m.shape(0),
                    "dadt_block: pos_m and mom must have one row per dipole");
            require(out.shape(0) == n_nodes && stride % 3 == 0 && column >= 0 &&
                        3 * (static_cast<std::int64_t>(column) + k) <= stride,
                    "dadt_block: out must be (n_nodes, 3m) with room for k columns from column");
            launch_dadt_block(pos_m.data(), mom.data(), transforms.data(), extent(pos_m), k,
                              nodes_mm.data(), narrow(n_nodes), out.data() + 3 * column, stride,
                              didt, stream);
        },
        native_call, dev("pos_m"), dev("mom"), dev("transforms"), dev("nodes_mm"), dev("out"),
        nb::arg("didt"), nb::arg("stream"), nb::arg("column") = 0,
        "dA/dt at nodes (mm) for k placements of one coil: coil-frame dipoles (m) and moments, "
        "and (k, 4, 4) coil-to-head affines in mm. out is (n_nodes, 3m), an interleaved "
        "(n_nodes, m, 3) block, and placement i fills column column + i.");

    m.def(
        "rhs_operator_fill",
        [](const In<TetGradd>& g, const In<f64_1d>& scale, const In<Tets>& tet_nodes,
           const In<i32_1d>& ptr, const In<i32_1d>& idx, const In<i32_1d>& indptr,
           const In<i32_1d>& indices, const Rows4f& op, Stream stream) {
            require(op.shape(0) == indices.shape(0) && ptr.shape(0) == indptr.shape(0) &&
                        scale.shape(0) == g.shape(0) && tet_nodes.shape(0) == g.shape(0),
                    "rhs_operator_fill: shape mismatch");
            launch_rhs_operator_fill(g.data(), scale.data(), tet_nodes.data(), ptr.data(),
                                     idx.data(), indptr.data(), indices.data(), op.data(),
                                     extent(indptr) - 1, stream);
        },
        native_call, dev("g"), dev("scale"), dev("tet_nodes"), dev("ptr"), dev("idx"),
        dev("indptr"), dev("indices"), dev("op"), nb::arg("stream"),
        "Build the forward RHS operator over the stiffness pattern: op is (nnz, 4), each row a "
        "packed (Bx, By, Bz, column); scale is neg_vc / 4 in float64.");

    m.def(
        "rhs_operator_apply",
        [](const In<i32_1d>& indptr, const In<Rows4f>& op, const In<Interleaved>& a,
           const f32_2d& b, Stream stream) {
            const std::size_t n_nodes = indptr.shape(0) - 1;
            const int k = block_width(a.shape(1), "rhs_operator_apply: bad block size");
            require(a.shape(0) == n_nodes && b.shape(0) == n_nodes && extent(b, 1) == k,
                    "rhs_operator_apply: shape mismatch");
            launch_rhs_operator(indptr.data(), op.data(), a.data(), b.data(), narrow(n_nodes), k,
                                stream);
        },
        native_call, dev("indptr"), dev("op"), dev("a"), dev("b"), nb::arg("stream"),
        "b = B a for an interleaved (n_nodes, k, 3) nodal dA/dt; writes row-major (n_nodes, k) "
        "float32. Each column's sum order is independent of k.");

    m.def(
        "dadt_node_to_element",
        [](const In<Rows3f>& dadt_nodes, const In<Tets>& tet_nodes, const Rows3f& out,
           Stream stream) {
            require(out.shape(0) == tet_nodes.shape(0),
                    "dadt_node_to_element: out must have one row per tet");
            launch_dadt_element_average(dadt_nodes.data(), tet_nodes.data(), out.data(),
                                        extent(tet_nodes), stream);
        },
        native_call, dev("dadt_nodes"), dev("tet_nodes"), dev("out"), nb::arg("stream"),
        "Average nodal dA/dt onto tetrahedra; writes into caller-allocated out.");

    m.def(
        "rhs_assemble",
        [](const In<Rows3f>& dadt_elm, const In<TetGradf>& g, const In<f32_1d>& neg_vc,
           const In<i32_1d>& ptr, const In<i32_1d>& idx, const f32_1d& b, Stream stream) {
            require(ptr.shape(0) == b.shape(0) + 1, "rhs_assemble: ptr must be n_nodes + 1");
            require(dadt_elm.shape(0) == g.shape(0) && neg_vc.shape(0) == g.shape(0),
                    "rhs_assemble: dadt_elm, g and neg_vc must be per tetrahedron");
            launch_rhs(dadt_elm.data(), g.data(), neg_vc.data(), ptr.data(), idx.data(), b.data(),
                       extent(b), stream);
        },
        native_call, dev("dadt_elm"), dev("g"), dev("neg_vc"), dev("ptr"), dev("idx"), dev("b"),
        nb::arg("stream"),
        "Deterministic node-centric RHS assembly; writes into caller-allocated b.");

    m.def(
        "tet_lowest_node",
        [](const In<i32_1d>& inverse, const In<Tets>& tet_nodes, const i32_1d& lowest,
           Stream stream) {
            require(lowest.shape(0) == tet_nodes.shape(0),
                    "tet_lowest_node: lowest must have one entry per tet");
            launch_tet_lowest_node(inverse.data(), tet_nodes.data(), lowest.data(),
                                   extent(tet_nodes), stream);
        },
        native_call, dev("inverse"), dev("tet_nodes"), dev("lowest"), nb::arg("stream"),
        "Smallest permuted node index of each tetrahedron, the sort key of the Morton layout.");

    m.def(
        "count_incident_node_csr",
        [](const In<Tets>& tet_nodes, const In<i32_1d>& ptr, const In<i32_1d>& idx,
           const i32_1d& out_ptr, Stream stream) {
            require(ptr.shape(0) == out_ptr.shape(0),
                    "count_incident_node_csr: ptr and out_ptr must both be n_seg + 1");
            return count_incident_node_csr(tet_nodes.data(), ptr.data(), idx.data(), out_ptr.data(),
                                           extent(out_ptr) - 1, stream);
        },
        native_call, dev("tet_nodes"), dev("ptr"), dev("idx"), dev("out_ptr"), nb::arg("stream"),
        "Row offsets of the distinct nodes of the tetrahedra in each segment of a corner CSR. "
        "Fills out_ptr and returns nnz, the length to allocate the column index at. "
        "Synchronises the stream.");

    m.def(
        "fill_incident_node_csr",
        [](const In<Tets>& tet_nodes, const In<i32_1d>& ptr, const In<i32_1d>& idx,
           const In<i32_1d>& out_ptr, const i32_1d& out_idx, Stream stream) {
            require(ptr.shape(0) == out_ptr.shape(0),
                    "fill_incident_node_csr: ptr and out_ptr must both be n_seg + 1");
            fill_incident_node_csr(tet_nodes.data(), ptr.data(), idx.data(), out_ptr.data(),
                                   out_idx.data(), extent(out_ptr) - 1, stream);
        },
        native_call, dev("tet_nodes"), dev("ptr"), dev("idx"), dev("out_ptr"), dev("out_idx"),
        nb::arg("stream"), "Fill the column index for the rows count_incident_node_csr sized.");

    m.def(
        "count_patch_csr",
        [](const In<i32_1d>& r1_ptr, const In<i32_1d>& r1_idx, const In<i32_1d>& neighbour,
           int min_nodes, const i32_1d& out_ptr, Stream stream) {
            require(neighbour.shape(0) == r1_idx.shape(0),
                    "count_patch_csr: neighbour must have one entry per first-ring node");
            require(out_ptr.shape(0) == r1_ptr.shape(0),
                    "count_patch_csr: r1_ptr and out_ptr must both be n_slots + 1");
            return count_patch_csr(r1_ptr.data(), r1_idx.data(), neighbour.data(), min_nodes,
                                   out_ptr.data(), extent(r1_ptr) - 1, stream);
        },
        native_call, dev("r1_ptr"), dev("r1_idx"), dev("neighbour"), nb::arg("min_nodes"),
        dev("out_ptr"), nb::arg("stream"),
        "Row offsets of the recovery patches: a slot reaching fewer than min_nodes grows to the "
        "union of its neighbours' rings. Same convention as count_incident_node_csr.");

    m.def(
        "fill_patch_csr",
        [](const In<i32_1d>& r1_ptr, const In<i32_1d>& r1_idx, const In<i32_1d>& neighbour,
           int min_nodes, const In<i32_1d>& out_ptr, const i32_1d& out_idx, Stream stream) {
            require(neighbour.shape(0) == r1_idx.shape(0),
                    "fill_patch_csr: neighbour must have one entry per first-ring node");
            require(out_ptr.shape(0) == r1_ptr.shape(0),
                    "fill_patch_csr: r1_ptr and out_ptr must both be n_slots + 1");
            fill_patch_csr(r1_ptr.data(), r1_idx.data(), neighbour.data(), min_nodes,
                           out_ptr.data(), out_idx.data(), extent(r1_ptr) - 1, stream);
        },
        native_call, dev("r1_ptr"), dev("r1_idx"), dev("neighbour"), nb::arg("min_nodes"),
        dev("out_ptr"), dev("out_idx"), nb::arg("stream"),
        "Fill the column index for the rows count_patch_csr sized.");

    m.def(
        "assemble_stiffness_values",
        [](const In<TetGradd>& g, const In<f64_1d>& scale, const In<Tets>& tet_nodes,
           const In<i32_1d>& ptr, const In<i32_1d>& idx, const In<i32_1d>& indptr,
           const In<i32_1d>& indices, const f64_1d& data, Stream stream) {
            require(data.shape(0) == indices.shape(0) && ptr.shape(0) == indptr.shape(0),
                    "assemble_stiffness_values: shape mismatch");
            launch_stiffness_rows(g.data(), scale.data(), tet_nodes.data(), ptr.data(), idx.data(),
                                  indptr.data(), indices.data(), data.data(), extent(indptr) - 1,
                                  stream);
        },
        native_call, dev("g"), dev("scale"), dev("tet_nodes"), dev("ptr"), dev("idx"),
        dev("indptr"), dev("indices"), dev("data"), nb::arg("stream"),
        "Fill a CSR pattern's values with the stiffness matrix, one thread per row.");

    m.def(
        "l1_dinv",
        [](const In<i32_1d>& indptr, const In<i32_1d>& indices, const In<f32_1d>& data,
           const f32_1d& dinv, Stream stream) {
            const std::size_t n_rows = indptr.shape(0) - 1;
            require(dinv.shape(0) == n_rows, "dinv must have one entry per row");
            require(indices.shape(0) == data.shape(0),
                    "indices and data must have one entry per nonzero");
            launch_l1_dinv(indptr.data(), indices.data(), data.data(), dinv.data(), narrow(n_rows),
                           stream);
        },
        native_call, dev("indptr"), dev("indices"), dev("data"), dev("dinv"), nb::arg("stream"),
        "l1-Jacobi smoother scaling 1 / (sign(a_ii) * sum_j |a_ij|), one thread per row.");

    m.def(
        "reconstruct_e_nodal_block",
        [](const In<f64_2d>& v_block, const In<Tets>& tet_nodes, const In<TetGradf>& g,
           const In<Interleaved>& dadt_nodes, const std::vector<Rows3f>& e_out,
           const std::vector<f32_1d>& magn_out, Stream stream) {
            const int k = block_width(v_block.shape(1), "reconstruct_e_nodal_block: bad width");
            const std::size_t kd = dadt_nodes.shape(1);
            require(dadt_nodes.shape(0) == v_block.shape(0) &&
                        (kd == static_cast<std::size_t>(k) || kd == 1),
                    "reconstruct_e_nodal_block: dadt_nodes must be (n_nodes, k or 1, 3)");
            const std::size_t n_tet = tet_nodes.shape(0);
            require(g.shape(0) == n_tet, "reconstruct_e_nodal_block: g must have one row per tet");
            const char* bad = "reconstruct_e_nodal_block: e_out and magn_out must be k arrays";
            const auto eo = optional_pack(e_out, k, n_tet, bad);
            const auto mo = optional_pack(magn_out, k, n_tet, bad);
            launch_reconstruct_nodal_block(v_block.data(), tet_nodes.data(), g.data(),
                                           dadt_nodes.data(), kd == 1, eo.data(), mo.data(),
                                           narrow(n_tet), k, stream);
        },
        native_call, dev("v_block"), dev("tet_nodes"), dev("g"), dev("dadt_nodes"), dev("e_out"),
        dev("magn_out"), nb::arg("stream"),
        "E and its magnitude per tetrahedron from k potentials and nodal dA/dt, averaged onto each "
        "element in the kernel. An empty e_out or magn_out is not written. dadt_nodes is "
        "interleaved (n_nodes, k, 3), or (n_nodes, 1, 3) for one field every placement shares.");

    m.def(
        "reconstruct_e",
        [](const In<f64_1d>& v, const In<Tets>& tet_nodes, const In<TetGradf>& g,
           const In<Rows3f>& dadt_elm, const Rows3f& e_out, const f32_1d& magn_out, Stream stream) {
            const std::size_t n_tet = tet_nodes.shape(0);
            require(g.shape(0) == n_tet && dadt_elm.shape(0) == n_tet && e_out.shape(0) == n_tet &&
                        magn_out.shape(0) == n_tet,
                    "reconstruct_e: g, dadt_elm, e_out and magn_out must have one row per tet");
            launch_reconstruct(v.data(), tet_nodes.data(), g.data(), dadt_elm.data(), e_out.data(),
                               magn_out.data(), narrow(n_tet), stream);
        },
        native_call, dev("v"), dev("tet_nodes"), dev("g"), dev("dadt_elm"), dev("e_out"),
        dev("magn_out"), nb::arg("stream"),
        "Element-centric E/magnE reconstruction; writes into caller-allocated e_out/magn_out.");

    m.def(
        "mark_outer_boundary",
        [](const In<Tets>& tet_nodes, const In<i32_1d>& ptr, const In<i32_1d>& idx,
           const i32_1d& is_boundary, Stream stream) {
            const std::size_t n_tet = tet_nodes.shape(0);
            require(idx.shape(0) == n_tet * 4,
                    "mark_outer_boundary: idx must hold one entry per tet corner");
            require(ptr.shape(0) == is_boundary.shape(0) + 1,
                    "mark_outer_boundary: ptr must hold n_nodes + 1 entries");
            launch_mark_outer_boundary(tet_nodes.data(), ptr.data(), idx.data(), is_boundary.data(),
                                       narrow(n_tet), stream);
        },
        native_call, dev("tet_nodes"), dev("ptr"), dev("idx"), dev("is_boundary"),
        nb::arg("stream"),
        "Mark the nodes of every face owned by exactly one tet, counted over the node2corner "
        "incidence CSR. is_boundary is only ever set, so it arrives zeroed.");

    m.def(
        "spr_weights",
        [](const In<Rows3d>& nodes_mm, const In<Tets>& tet_nodes, const In<f32_1d>& vols,
           const In<i32_1d>& ptr, const In<i32_1d>& idx, const std::optional<In<i32_1d>>& slot_node,
           const In<i32_1d>& is_boundary, const f32_1d& w, const i32_1d& n_fallback,
           Stream stream) {
            const std::size_t n_slots = ptr.shape(0) - 1;
            require(w.shape(0) == idx.shape(0), "spr_weights: w must have one entry per corner");
            require(!slot_node || slot_node->shape(0) == n_slots,
                    "spr_weights: slot_node must have one entry per slot");
            require(is_boundary.shape(0) == nodes_mm.shape(0),
                    "spr_weights: is_boundary must have one entry per node");
            require(n_fallback.shape(0) == 1, "spr_weights: n_fallback must hold one counter");
            launch_spr_weights(nodes_mm.data(), tet_nodes.data(), vols.data(), ptr.data(),
                               idx.data(), data_or_null(slot_node), is_boundary.data(), w.data(),
                               n_fallback.data(), narrow(n_slots), stream);
        },
        native_call, dev("nodes_mm"), dev("tet_nodes"), dev("vols"), dev("ptr"), dev("idx"),
        dev("slot_node"), dev("is_boundary"), dev("w"), dev("n_fallback"), nb::arg("stream"),
        "One scalar patch weight per corner. Pass slot_node=None when a slot is a node. "
        "is_boundary is one int per node, zero to fit it; n_fallback is one counter the kernel "
        "adds to.");

    m.def(
        "skin_normals",
        [](const In<Rows3d>& nodes_mm, const In<Tris>& tris, const In<i32_1d>& node_ptr,
           const In<i32_1d>& node_idx, const Rows3d& out, Stream stream) {
            const std::size_t n_tri = tris.shape(0);
            require(out.shape(0) == n_tri, "skin_normals: tris and out must be (n_tri, 3)");
            require(node_idx.shape(0) == 3 * n_tri,
                    "skin_normals: node_idx must hold every corner");
            launch_skin_normals(nodes_mm.data(), tris.data(), node_ptr.data(), node_idx.data(),
                                out.data(), narrow(n_tri), extent(node_ptr) - 1, stream);
        },
        native_call, dev("nodes_mm"), dev("tris"), dev("node_ptr"), dev("node_idx"), dev("out"),
        nb::arg("stream"),
        "Smoothed outward skin-triangle normals. node_ptr/node_idx is a node-to-corner CSR over "
        "the triangles, and its order is the order every node's smoothing sums in.");

    m.def(
        "ray_first_hit",
        [](const In<Rows3d>& origins, const In<Rows3d>& directions, const In<TriVerts>& verts,
           const f64_1d& out_t, double min_t, Stream stream) {
            const std::size_t n_rays = origins.shape(0);
            require(directions.shape(0) == n_rays,
                    "ray_first_hit: origins and directions must both be (n_rays, 3)");
            require(out_t.shape(0) == n_rays, "ray_first_hit: out_t must have one entry per ray");
            launch_ray_first_hit(origins.data(), directions.data(), verts.data(), out_t.data(),
                                 narrow(n_rays), extent(verts), min_t, stream);
        },
        native_call, dev("origins"), dev("directions"), dev("verts"), dev("out_t"),
        nb::arg("min_t"), nb::arg("stream"),
        "Moller-Trumbore first-hit distance per ray against a triangle surface; +inf where the "
        "ray misses. Hits nearer than min_t are skipped, so a ray leaving one surface does not "
        "hit the one it started on.");

    m.def(
        "hpr_weights",
        [](const In<Rows3d>& nodes_m, const In<i32_1d>& pptr, const In<i32_1d>& pidx,
           const In<i32_1d>& slot_node, const Rows3f& w, const i32_1d& status, Stream stream,
           const std::optional<In<i32_1d>>& active) {
            const std::size_t n_slots = pptr.shape(0) - 1;
            require(w.shape(0) == pidx.shape(0), "hpr_weights: w must be (nnz, 3)");
            require(slot_node.shape(0) == n_slots && status.shape(0) == n_slots,
                    "hpr_weights: slot_node/status must be per slot");
            launch_hpr_weights(nodes_m.data(), pptr.data(), pidx.data(), slot_node.data(), w.data(),
                               status.data(), data_or_null(active),
                               active ? extent(*active) : narrow(n_slots), stream);
        },
        native_call, dev("nodes_m"), dev("pptr"), dev("pidx"), dev("slot_node"), dev("w"),
        dev("status"), nb::arg("stream"), dev("active") = nb::none(),
        "Harmonic-constrained potential-recovery weights: grad_v[s] = sum_m w[s,m] v[m], with "
        "nodes_m in metres so that the weights carry 1/m. "
        "status is the rung the fit took, counted down from the widest basis: 0 harmonic cubic, "
        "1 harmonic quadratic, 2 linear, 3 no gradient determined.");

    m.def(
        "hpr_grad",
        [](const In<f64_2d>& v_block, const In<Rows3f>& w, const In<i32_1d>& pptr,
           const In<i32_1d>& pidx, const In<i32_1d>& slot_node, const In<Interleaved>& dadt_nodes,
           const std::vector<Rows3f>& e_slots, const f32_1d& work, Stream stream,
           const std::optional<In<i32_1d>>& active) {
            const int k = block_width(e_slots.size(), "hpr_grad: bad list sizes");
            const int stride = extent(v_block, 1);
            require(stride >= k, "hpr_grad: v_block has fewer columns than placements");
            const std::size_t n_nodes = v_block.shape(0);
            const std::size_t n_slots = pptr.shape(0) - 1;
            const std::size_t kd = dadt_nodes.shape(1);
            require(w.shape(0) == pidx.shape(0), "hpr_grad: w must be (nnz, 3)");
            require(slot_node.shape(0) == n_slots, "hpr_grad: slot_node must be per slot");
            require(dadt_nodes.shape(0) == n_nodes &&
                        (kd == static_cast<std::size_t>(k) || kd == 1),
                    "hpr_grad: dadt_nodes must be (n_nodes, k or 1, 3)");
            require(work.shape(0) >= hpr_grad_work_floats(narrow(n_nodes), k),
                    "hpr_grad: work is smaller than hpr_grad_work_size");
            const auto out = pack<float*>(e_slots, k, n_slots,
                                          "hpr_grad: e_slots must be k (n_slots, 3)");
            launch_hpr_grad(v_block.data(), w.data(), pptr.data(), pidx.data(), slot_node.data(),
                            dadt_nodes.data(), static_cast<std::int64_t>(3 * kd), kd == 1 ? 0 : 3,
                            out.data(), narrow(n_slots), data_or_null(active),
                            active ? extent(*active) : narrow(n_slots), k, stride, narrow(n_nodes),
                            work.data(), stream);
        },
        native_call, dev("v_block"), dev("w"), dev("pptr"), dev("pidx"), dev("slot_node"),
        dev("dadt_nodes"), dev("e_slots"), dev("work"), nb::arg("stream"),
        dev("active") = nb::none(),
        "Recovered E on slots from the nodal potential: E = -grad_v - dA/dt, with dA/dt taken "
        "exactly at the node. dadt_nodes is interleaved "
        "(n_nodes, k, 3), or (n_nodes, 1, 3) for one field every placement shares; work holds "
        "hpr_grad_work_size float32s.");

    m.def("hpr_grad_work_size", &hpr_grad_work_floats, nb::arg("n_nodes"), nb::arg("k"),
          "float32s of work hpr_grad needs.");

    m.def(
        "recover_nodes",
        [](const std::vector<In<Rows3f>>& e_in, const In<f32_1d>& w, const In<i32_1d>& ptr,
           const In<i32_1d>& idx, const std::vector<Rows3f>& e_slots, Stream stream) {
            const int k = block_width(e_in.size(), "recover_nodes: bad list sizes");
            const std::size_t n_slots = ptr.shape(0) - 1;
            require(w.shape(0) == idx.shape(0), "recover_nodes: w must match idx");
            const auto in = pack<const float*>(e_in, k, e_in[0].shape(0),
                                               "recover_nodes: e_in must be k (n_tet, 3)");
            const auto out = pack<float*>(e_slots, k, n_slots,
                                          "recover_nodes: e_slots must be k (n_slots, 3)");
            launch_recover_nodes(in.data(), w.data(), ptr.data(), idx.data(), out.data(),
                                 narrow(n_slots), k, stream);
        },
        native_call, dev("e_in"), dev("w"), dev("ptr"), dev("idx"), dev("e_slots"),
        nb::arg("stream"),
        "Patch gather E*[s] = sum_c w[c] E[c>>2] for <=8 placements; the operator is read once "
        "for the whole block and a column's result does not depend on k.");

    m.def(
        "recover_elements",
        [](const std::vector<In<Rows3f>>& e_slots, const In<i32_1d>& slot_of_corner,
           const std::vector<Rows3f>& e_out, const std::vector<f32_1d>& magn_out, Stream stream,
           const std::optional<In<i32_1d>>& active) {
            const int k = block_width(e_slots.size(), "recover_elements: bad list sizes");
            const std::size_t n_tet = slot_of_corner.shape(0) / 4;
            require(slot_of_corner.shape(0) % 4 == 0,
                    "recover_elements: slot_of_corner must be (4 * n_tet,)");
            require(reinterpret_cast<std::uintptr_t>(slot_of_corner.data()) % 16 == 0,
                    "recover_elements: slot_of_corner must start 16-byte aligned");
            const char* bad = "recover_elements: e_out and magn_out must be k arrays of n_tet";
            const auto in = pack<const float*>(e_slots, k, e_slots[0].shape(0),
                                               "recover_elements: e_slots must be k (n_slots, 3)");
            const auto eo = optional_pack(e_out, k, n_tet, bad);
            const auto mo = optional_pack(magn_out, k, n_tet, bad);
            launch_recover_elements(in.data(), slot_of_corner.data(), eo.data(), mo.data(),
                                    narrow(n_tet), data_or_null(active),
                                    active ? extent(*active) : narrow(n_tet), k, stream);
        },
        native_call, dev("e_slots"), dev("slot_of_corner"), dev("e_out"), dev("magn_out"),
        nb::arg("stream"), dev("active") = nb::none(),
        "Sample the recovered field back at the barycentres and take its magnitude. An empty "
        "e_out or magn_out is not written.");

    m.def(
        "node_gather",
        [](const In<f64_1d>& corner, const In<i32_1d>& ptr, const In<i32_1d>& idx,
           const f64_1d& out, Stream stream) {
            const std::size_t n_nodes = ptr.shape(0) - 1;
            require(out.shape(0) == n_nodes, "node_gather: out must have one entry per node");
            launch_node_gather(corner.data(), ptr.data(), idx.data(), out.data(), narrow(n_nodes),
                               stream);
        },
        native_call, dev("corner"), dev("ptr"), dev("idx"), dev("out"), nb::arg("stream"),
        "Node-centric scalar corner gather over the node2corner CSR; into caller-allocated out.");

    m.def(
        "node_weight",
        [](const In<f64_1d>& lam, const In<Tets>& tet_nodes, const In<TetGradf>& g,
           const In<f32_1d>& neg_vc, const In<i32_1d>& ptr, const In<i32_1d>& idx,
           const In<i32_1d>& roi_elems, const In<f64_1d>& roi_alpha,
           const std::array<double, 3>& direction, const Rows3d& node_w, Stream stream) {
            const std::size_t n_nodes = ptr.shape(0) - 1;
            require(node_w.shape(0) == n_nodes && lam.shape(0) == n_nodes,
                    "node_weight: lam and node_w must have one row per node");
            require(g.shape(0) == tet_nodes.shape(0) && neg_vc.shape(0) == tet_nodes.shape(0),
                    "node_weight: g and neg_vc must have one row per tet");
            require(roi_alpha.shape(0) == roi_elems.shape(0),
                    "node_weight: roi_alpha must have one entry per ROI element");
            launch_node_weight(lam.data(), tet_nodes.data(), g.data(), neg_vc.data(),
                               extent(tet_nodes), ptr.data(), idx.data(), roi_elems.data(),
                               roi_alpha.data(), extent(roi_elems), direction.data(), node_w.data(),
                               narrow(n_nodes), stream);
        },
        native_call, dev("lam"), dev("tet_nodes"), dev("g"), dev("neg_vc"), dev("ptr"), dev("idx"),
        dev("roi_elems"), dev("roi_alpha"), nb::arg("direction"), dev("node_w"), nb::arg("stream"),
        "Nodal reciprocity weights, 1/4 the sum over each node's corners of "
        "vol*sigma*(G_e lam) - alpha_e direction; roi_elems ascending; into caller-allocated "
        "node_w.");

    m.def(
        "adm_evaluate",
        [](const In<Frames>& frames, const In<Rows3f>& coil_pos, const In<Rows3f>& coil_mom,
           const In<f32_1d>& q, const In<u8_1d>& cell_ok, const In<i32_1d>& cell_slot,
           const In<f32_1d>& q_near, const In<i32_1d>& list_ptr, const In<f32_1d>& entries,
           const GridVec& origin, const GridVec& spacing, const GridShape& shape, int n_dir,
           double didt, const f64_2d& out, const i32_1d& fault, Stream stream) {
            const GridGeom g = isotropic_grid_args(origin, spacing, shape, "adm_evaluate");
            const std::size_t n_frames = frames.shape(0);
            const std::int64_t n_cells =
                static_cast<std::int64_t>(shape[0] - 1) * (shape[1] - 1) * (shape[2] - 1);
            require(coil_mom.shape(0) == coil_pos.shape(0) &&
                        static_cast<std::int64_t>(q.shape(0)) == g.points() * n_dir * 3 &&
                        static_cast<std::int64_t>(cell_ok.shape(0)) == n_cells &&
                        static_cast<std::int64_t>(cell_slot.shape(0)) == n_cells &&
                        out.shape(0) == n_frames && extent(out, 1) == n_dir && fault.shape(0) == 2,
                    "adm_evaluate: shape mismatch");
            require_slots(list_ptr, list_ptr.shape(0) - 1, q_near, entries, n_dir,
                          "adm_evaluate");
            launch_adm_evaluate(frames.data(), narrow(n_frames), coil_pos.data(), coil_mom.data(),
                                extent(coil_pos), q.data(), cell_ok.data(), cell_slot.data(),
                                q_near.data(), list_ptr.data(), entries.data(), g, n_dir, didt,
                                out.data(), fault.data(), stream);
        },
        native_call, dev("frames"), dev("coil_pos"), dev("coil_mom"), dev("q"), dev("cell_ok"),
        dev("cell_slot"), dev("q_near"), dev("list_ptr"), dev("entries"), nb::arg("origin"),
        nb::arg("spacing"), nb::arg("shape"), nb::arg("n_dir"), nb::arg("didt"), dev("out"),
        dev("fault"), nb::arg("stream"),
        "Target E-vectors (n_frames, n_dir) of coil frames against a sampled reciprocity field. "
        "A cell with a slot (cell_slot >= 0) interpolates its own corners q_near and adds the "
        "sources list_ptr delimits in entries (adm_entry_floats(n_dir) floats each: position in "
        "grid index units, then the weights) directly. fault (2 ints the caller sets to "
        "n_frames) receives the lowest candidate off the grid and the lowest reading an "
        "unsampled cell.");

    m.def(
        "adm_sample",
        [](const In<Rows3f>& src, const In<SetRows3f>& src_w, const Dev<int32_t, -1, 6>& tile_box,
           const In<i64_1d>& points, const GridVec& origin, const GridVec& spacing,
           const GridShape& shape, const f32_1d& q, Stream stream) {
            const GridGeom g = isotropic_grid_args(origin, spacing, shape, "adm_sample");
            const int n_dir = extent(src_w);
            require(static_cast<std::int64_t>(q.shape(0)) == g.points() * n_dir * 3,
                    "adm_sample: q must be (nx * ny * nz * n_dir * 3,)");
            require_sources(src, src_w, n_dir, "adm_sample");
            const std::size_t tile = static_cast<std::size_t>(adm_sample_tile());
            require(tile_box.shape(0) == (src.shape(0) + tile - 1) / tile,
                    "adm_sample: tile_box must have one row per source tile");
            launch_adm_sample(src.data(), src_w.data(), extent(src), n_dir, tile_box.data(),
                              points.data(), extent(points), g, q.data(), stream);
        },
        native_call, dev("src"), dev("src_w"), dev("tile_box"), dev("points"), nb::arg("origin"),
        nb::arg("spacing"), nb::arg("shape"), dev("q"), nb::arg("stream"),
        "Samples the reciprocity field at the sorted flat grid indices points into q, each point "
        "from every source (grid index units, sorted by cell) outside its eight cells. tile_box "
        "holds each adm_sample_tile()-source tile's floor(src) box, lo xyz then hi xyz.");

    m.def("adm_sample_tile", &adm_sample_tile, "Sources per adm_sample tile.");

    m.def(
        "adm_q_points",
        [](const In<Rows3f>& src, const In<SetRows3f>& src_w, const In<Rows3f>& points,
           double scale, const f32_2d& out, Stream stream) {
            const int n_dir = extent(src_w);
            require_sources(src, src_w, n_dir, "adm_q_points");
            require(out.shape(0) == points.shape(0) && extent(out, 1) == 3 * n_dir,
                    "adm_q_points: out must be (n_points, 3 * n_dir)");
            launch_adm_q_points(src.data(), src_w.data(), extent(src), n_dir, points.data(),
                                extent(points), scale, out.data(), stream);
        },
        native_call, dev("src"), dev("src_w"), dev("points"), nb::arg("scale"), dev("out"),
        nb::arg("stream"),
        "out (n_points, 3 * n_dir) = scale * sum over every source of W x (p - s) / |p - s|^3, "
        "with src, src_w (n_dir, n_src, 3) and points in one frame and unit.");

    m.def(
        "adm_slot_corners",
        [](const In<i64_1d>& slot_cells, int first_slot, const In<i32_1d>& list_ptr,
           const In<f32_1d>& entries, int n_dir, const In<f32_1d>& q, const GridVec& origin,
           const GridVec& spacing, const GridShape& shape, const f32_1d& q_near, Stream stream) {
            const GridGeom g = isotropic_grid_args(origin, spacing, shape, "adm_slot_corners");
            const std::size_t n_slots = slot_cells.shape(0);
            require(static_cast<std::int64_t>(q.shape(0)) == g.points() * n_dir * 3 &&
                        first_slot >= 0 && static_cast<std::size_t>(first_slot) <= n_slots,
                    "adm_slot_corners: shape mismatch");
            require_slots(list_ptr, n_slots, q_near, entries, n_dir, "adm_slot_corners");
            launch_adm_slot_corners(slot_cells.data(), first_slot, narrow(n_slots), list_ptr.data(),
                                    entries.data(), n_dir, q.data(), g, q_near.data(), stream);
        },
        native_call, dev("slot_cells"), nb::arg("first_slot"), dev("list_ptr"), dev("entries"),
        nb::arg("n_dir"), dev("q"), nb::arg("origin"), nb::arg("spacing"), nb::arg("shape"),
        dev("q_near"), nb::arg("stream"),
        "Fills q_near (n_slots, 8, n_dir, 3) from first_slot on: each slot cell's sampled corners "
        "less its entries' sources that the corner's own sample kept.");

    m.def("adm_entry_floats", &adm_entry_floats, nb::arg("n_dir"),
          "Floats per packed ADM slot entry for n_dir weight sets.");

    m.def(
        "adm_mark",
        [](const In<Frames>& frames, const In<Rows3f>& coil_pos, const GridVec& origin,
           const GridVec& spacing, const GridShape& shape, const u8_1d& need, Stream stream) {
            const GridGeom g = grid_args(origin, spacing, shape);
            require(static_cast<std::int64_t>(need.shape(0)) == g.points(),
                    "adm_mark: need must have one byte per grid point");
            launch_adm_mark(frames.data(), extent(frames), coil_pos.data(), extent(coil_pos), g,
                            need.data(), stream);
        },
        native_call, dev("frames"), dev("coil_pos"), nb::arg("origin"), nb::arg("spacing"),
        nb::arg("shape"), dev("need"), nb::arg("stream"),
        "Sets need to 1 at every grid point a frame's dipoles interpolate from.");

    m.def(
        "adm_reach",
        [](const GridVec& origin, const GridVec& spacing, const GridShape& shape,
           const In<Rows3d>& origins, const In<Rows3d>& axes, const In<u8_2d>& profile, double h0,
           double cell, const u8_1d& mask, Stream stream) {
            const GridGeom g = grid_args(origin, spacing, shape);
            require(origins.shape(0) == axes.shape(0) &&
                        static_cast<std::int64_t>(mask.shape(0)) == g.points(),
                    "adm_reach: shape mismatch");
            launch_adm_reach(g, origins.data(), axes.data(), extent(origins), profile.data(),
                             extent(profile), extent(profile, 1), h0, cell, mask.data(), stream);
        },
        native_call, nb::arg("origin"), nb::arg("spacing"), nb::arg("shape"), dev("origins"),
        dev("axes"), dev("profile"), nb::arg("h0"), nb::arg("cell"), dev("mask"), nb::arg("stream"),
        "Marks the grid points a coil's swept (rho, h) profile reaches about any centre's axis.");

    m.attr("SUMMARY_LAYOUT") = nb::make_tuple(kSummaryPeak, kSummaryPeakLocation,
                                              kSummaryCenterOfGravity, kSummaryMean, kSummaryStd,
                                              kSummaryQuantiles);
    m.def("region_summary_work_size", &region_summary_work_size, nb::arg("n"), nb::arg("k"),
          nb::arg("f64"), "Bytes of workspace region_summary needs.");
    m.def(
        "region_summary",
        [](const std::vector<nb::ndarray<nb::ro, nb::ndim<1>, nb::c_contig, nb::device::cuda>>&
               fields,
           const In<i32_1d>& index, const In<f64_1d>& w, const In<Rows3d>& bary,
           const std::vector<double>& qs, int anchor, const std::vector<double>& fracs,
           const u8_1d& work, const f64_2d& out, Stream stream) {
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
            const int n_q = narrow(qs.size());
            const int n_f = narrow(fracs.size());
            require(w.shape(0) == n && bary.shape(0) == n && extent(out) == k &&
                        extent(out, 1) == kSummaryQuantiles + n_q + n_f,
                    "region_summary: shape mismatch");
            launch_region_summary(ptrs.data(), f64, k, narrow(n), index.data(), w.data(),
                                  bary.data(), qs.data(), n_q, anchor, fracs.data(), n_f,
                                  work.data(), work.shape(0), out.data(), stream);
        },
        native_call, dev("fields"), dev("index"), dev("w"), dev("bary"), nb::arg("qs"),
        nb::arg("anchor"), nb::arg("fracs"), dev("work"), dev("out"), nb::arg("stream"),
        "Volume-weighted metrics of up to 8 float32 or float64 fields over one region. out is "
        "(k, SUMMARY_LAYOUT[-1] + len(qs) + len(fracs)): the entries SUMMARY_LAYOUT gives offsets "
        "for, then the quantiles qs, then the volume at or above fracs[f] times quantile "
        "qs[anchor].");

    m.def(
        "accumulate_moments_block",
        [](const std::vector<In<f32_1d>>& magn, const f64_1d& sum_e, const f64_1d& sumsq_e,
           Stream stream) {
            const int k = block_width(magn.size(), "accumulate_moments_block: bad sizes");
            const std::size_t n = sum_e.shape(0);
            require(sumsq_e.shape(0) == n, "accumulate_moments_block: bad sizes");
            const auto ptrs = pack<const float*>(magn, k, n, "accumulate_moments_block: bad sizes");
            launch_accumulate_moments_block(ptrs.data(), k, sum_e.data(), sumsq_e.data(), narrow(n),
                                            stream);
        },
        native_call, dev("magn"), dev("sum_e"), dev("sumsq_e"), nb::arg("stream"),
        "Streaming |E| moments for up to 8 draws, folded in the order given: sum_e += magn and "
        "sumsq_e += magn^2, in place.");

    m.def(
        "row_dot",
        [](const In<f64_2d>& a, const In<f64_2d>& s, const f64_2d& out, Stream stream) {
            const std::size_t n = a.shape(0);
            const std::size_t p = a.shape(1);
            const std::size_t k = s.shape(0);
            require(s.shape(1) == p && out.shape(0) == n && out.shape(1) == k,
                    "row_dot: a must be (n, p), s (k, p) and out (n, k)");
            launch_row_dot(a.data(), s.data(), out.data(), narrow(n), narrow(p), narrow(k), stream);
        },
        native_call, dev("a"), dev("s"), dev("out"), nb::arg("stream"),
        "out = a @ s.T for a tall, narrow a, each entry summed in term order. A column's result "
        "does not depend on k.");

    m.def("sweep_seed_work_size", &sweep_seed_work_size, nb::arg("n"), nb::arg("m"), nb::arg("k"),
          "Doubles of scratch sweep_seed needs for an (n, m) basis and k columns.");

    m.def(
        "sweep_seed",
        [](const In<f32_2d>& q, const In<f64_2d>& b, const f64_1d& work, const f64_2d& x0,
           Stream stream) {
            const int n = extent(q);
            const int m_basis = extent(q, 1);
            const int k = extent(b, 1);
            require(
                extent(b) == n && extent(x0) == n && extent(x0, 1) == k &&
                    static_cast<std::int64_t>(work.shape(0)) >= sweep_seed_work_size(n, m_basis, k),
                "sweep_seed: q (n, m), b and x0 (n, k), work sweep_seed_work_size(n, m, k)");
            launch_sweep_seed(q.data(), b.data(), n, m_basis, k, work.data(), x0.data(), stream);
        },
        native_call, dev("q"), dev("b"), dev("work"), dev("x0"), nb::arg("stream"),
        "x0 = q (q^T b) per column for an A-orthonormal fp32 basis; a column's result does not "
        "depend on k.");

    m.def(
        "place_transforms",
        [](const In<Rows3d>& centers, const std::optional<In<Rows3d>>& handles,
           const In<f64_1d>& dists, const In<TriVerts>& verts, const In<Rows3d>& tnorm,
           const Frames& out, const std::optional<i32_1d>& degenerate, Stream stream) {
            const std::size_t n_pl = centers.shape(0);
            const std::size_t n_tri = verts.shape(0);
            require((!handles || handles->shape(0) == n_pl) && dists.shape(0) == n_pl &&
                        out.shape(0) == n_pl && (!degenerate || degenerate->shape(0) == n_pl),
                    "place_transforms: handles, dists, out and degenerate must be per placement");
            require(tnorm.shape(0) == n_tri,
                    "place_transforms: verts and tnorm must be per triangle");
            launch_place(centers.data(), data_or_null(handles), dists.data(), verts.data(),
                         tnorm.data(), out.data(), data_or_null(degenerate), narrow(n_pl),
                         narrow(n_tri), stream);
        },
        native_call, dev("centers"), dev("handles"), dev("dists"), dev("verts"), dev("tnorm"),
        dev("out"), dev("degenerate"), nb::arg("stream"),
        "Batched closest-point scalp projection + coil frame; writes (P, 4, 4) row-major out "
        "and a (P,) flag marking placements whose handle left the in-plane axis undefined, or "
        "whose centre was not finite. Pass handles=None for the projection and normal alone, "
        "with degenerate=None.");
}
