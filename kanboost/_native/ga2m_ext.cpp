// Optional C++ (pybind11) accelerator for kanboost.train.ga2m's
// predict-time forward pass. NOT required for kanboost to work -- this
// extension is built on a best-effort basis (see setup.py) and
// kanboost.train.ga2m falls back to the pure Python/numba path
// automatically whenever it is not present.
//
// Measured on INSPIRE (Large scale, 54 rounds, 26267 test rows):
// ~2.26x faster than the numba-accelerated Python path for the same
// forward pass, verified numerically identical (differences at
// floating-point noise, ~1e-15/1e-16) -- see
// docs/inspire_kanboost_evaluation.md and AI_REVIEW_LOOP.md (CC-21)
// for the full investigation and evidence.
//
// Two functions, mirroring GA2M's two-layer forward pass exactly:
//
//   layer0_forward_shared: computes z (n x n_hidden) for GA2M's sparse
//     layer0 structure (n_in main-effect units, each fed by exactly one
//     feature, plus n_pairs interaction units, each fed by two
//     features). Each DISTINCT feature's B-spline basis is computed
//     exactly ONCE and reused via matmul for every hidden unit that
//     needs it -- mirroring KANLayer.forward's own
//     `for i in range(n_in): B = basis(x[:,i]); out += B @ coef[i].T`
//     structure. An earlier, naive version that recomputed the basis
//     once per (feature, hidden-unit) edge -- redundant whenever a
//     feature appears in more than one pair -- was ~2x slower.
//
//   layer1_forward_sum: for each hidden unit's own knots1/coefs, computes
//     its output-layer B-spline basis against z[:,h], multiplies by that
//     unit's coefficient block, sums across all hidden units, and scales
//     by gamma -- the round's contribution to F.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>
#include <cstring>

namespace py = pybind11;

namespace {

// Dense Cox-de-Boor recursion -- same algorithm as
// kanboost.core.kan.bspline's numba kernel. Writes into scratchA/scratchB
// (reused across calls by the caller to avoid repeated heap allocation);
// on return, out_ptr/out_bases describe the final (n, K) basis matrix.
void b_basis_dense(const double* x, int n, const double* knots, int n_knots, int k,
                    std::vector<double>& scratchA, std::vector<double>& scratchB,
                    double*& out_ptr, int& out_bases) {
    int n_bases0 = n_knots - 1;
    scratchA.assign((size_t)n * n_bases0, 0.0);
    double* cur = scratchA.data();
    for (int idx = 0; idx < n; ++idx) {
        double xv = x[idx];
        double* row = cur + (size_t)idx * n_bases0;
        for (int i = 0; i < n_bases0; ++i)
            if (knots[i] <= xv && xv < knots[i + 1]) row[i] = 1.0;
        if (xv >= knots[n_knots - 2]) row[n_bases0 - 1] = 1.0;
    }
    int cur_bases = n_bases0;
    for (int kk = 1; kk <= k; ++kk) {
        int n_bases = n_knots - kk - 1;
        scratchB.assign((size_t)n * n_bases, 0.0);
        double* nxt = scratchB.data();
        for (int idx = 0; idx < n; ++idx) {
            double xv = x[idx];
            const double* crow = cur + (size_t)idx * cur_bases;
            double* nrow = nxt + (size_t)idx * n_bases;
            for (int i = 0; i < n_bases; ++i) {
                double d1 = knots[i + kk] - knots[i];
                double d2 = knots[i + kk + 1] - knots[i + 1];
                double term1 = (d1 > 1e-14) ? (xv - knots[i]) / d1 * crow[i] : 0.0;
                double term2 = (d2 > 1e-14) ? (knots[i + kk + 1] - xv) / d2 * crow[i + 1] : 0.0;
                nrow[i] = term1 + term2;
            }
        }
        scratchA.swap(scratchB);
        cur = scratchA.data();
        cur_bases = n_bases;
    }
    out_ptr = scratchA.data();
    out_bases = cur_bases;
}

}  // namespace

static py::array_t<double> layer0_forward_shared(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> knots0,
    int k0,
    py::array_t<double, py::array::c_style | py::array::forcecast> coef0_main,   // (n_in, K0)
    py::array_t<int, py::array::c_style | py::array::forcecast> pair_a,
    py::array_t<int, py::array::c_style | py::array::forcecast> pair_b,
    py::array_t<double, py::array::c_style | py::array::forcecast> coef0_pair_a, // (n_pairs, K0)
    py::array_t<double, py::array::c_style | py::array::forcecast> coef0_pair_b  // (n_pairs, K0)
) {
    auto Xb = X.request();
    int n = (int)Xb.shape[0];
    int n_in = (int)Xb.shape[1];
    const double* Xp = static_cast<const double*>(Xb.ptr);

    auto knotsb = knots0.request();
    const double* knots = static_cast<const double*>(knotsb.ptr);
    int n_knots = (int)knotsb.shape[0];
    int K0 = n_knots - k0 - 1;
    double lo = knots[k0], hi = knots[n_knots - k0 - 1];

    auto mainb = coef0_main.request();
    const double* coef_main = static_cast<const double*>(mainb.ptr);

    auto pab = pair_a.request(); const int* pa = static_cast<const int*>(pab.ptr);
    auto pbb = pair_b.request(); const int* pb = static_cast<const int*>(pbb.ptr);
    int n_pairs = (int)pab.shape[0];
    auto cab = coef0_pair_a.request(); const double* ca = static_cast<const double*>(cab.ptr);
    auto cbb = coef0_pair_b.request(); const double* cb = static_cast<const double*>(cbb.ptr);

    int n_hidden = n_in + n_pairs;
    py::array_t<double> z_arr({n, n_hidden});
    double* z = z_arr.mutable_data();
    std::memset(z, 0, sizeof(double) * (size_t)n * n_hidden);

    // Group every (coef_vec, hidden_unit_idx) edge by its feature index --
    // each distinct feature's basis gets computed exactly once below.
    std::unordered_map<int, std::vector<std::pair<const double*, int>>> edges_by_feature;
    for (int j = 0; j < n_in; ++j)
        edges_by_feature[j].push_back({coef_main + (size_t)j * K0, j});
    for (int idx = 0; idx < n_pairs; ++idx) {
        int h = n_in + idx;
        edges_by_feature[pa[idx]].push_back({ca + (size_t)idx * K0, h});
        edges_by_feature[pb[idx]].push_back({cb + (size_t)idx * K0, h});
    }

    std::vector<double> col(n), scratchA, scratchB;
    for (auto& kv : edges_by_feature) {
        int feat = kv.first;
        for (int idx = 0; idx < n; ++idx) {
            double v = Xp[(size_t)idx * n_in + feat];
            col[idx] = (v < lo) ? lo : (v > hi ? hi : v);
        }
        double* basis_ptr; int bases;
        b_basis_dense(col.data(), n, knots, n_knots, k0, scratchA, scratchB, basis_ptr, bases);

        for (auto& edge : kv.second) {
            const double* coef_vec = edge.first;
            int hidden_idx = edge.second;
            for (int idx = 0; idx < n; ++idx) {
                const double* row = basis_ptr + (size_t)idx * bases;
                double acc = 0.0;
                for (int i = 0; i < bases; ++i) acc += row[i] * coef_vec[i];
                z[(size_t)idx * n_hidden + hidden_idx] += acc;
            }
        }
    }
    return z_arr;
}

static py::array_t<double> layer1_forward_sum(
    py::array_t<double, py::array::c_style | py::array::forcecast> Z,       // (n, n_hidden)
    py::array_t<double, py::array::c_style | py::array::forcecast> knots1,  // (n_hidden, n_knots1)
    int k1,
    py::array_t<double, py::array::c_style | py::array::forcecast> coefs,   // (n_hidden * K1,)
    double gamma
) {
    auto Zb = Z.request();
    int n = (int)Zb.shape[0];
    int n_hidden = (int)Zb.shape[1];
    const double* Zp = static_cast<const double*>(Zb.ptr);

    auto knotsb = knots1.request();
    const double* knots_all = static_cast<const double*>(knotsb.ptr);
    int n_knots1 = (int)knotsb.shape[1];
    int K1 = n_knots1 - k1 - 1;

    auto coefb = coefs.request();
    const double* c = static_cast<const double*>(coefb.ptr);

    py::array_t<double> out_arr(n);
    double* out = out_arr.mutable_data();
    std::memset(out, 0, sizeof(double) * n);

    std::vector<double> col(n), scratchA, scratchB;
    for (int h = 0; h < n_hidden; ++h) {
        const double* knots = knots_all + (size_t)h * n_knots1;
        double lo = knots[k1], hi = knots[n_knots1 - k1 - 1];
        for (int idx = 0; idx < n; ++idx) {
            double v = Zp[(size_t)idx * n_hidden + h];
            col[idx] = (v < lo) ? lo : (v > hi ? hi : v);
        }
        double* basis_ptr; int bases;
        b_basis_dense(col.data(), n, knots, n_knots1, k1, scratchA, scratchB, basis_ptr, bases);

        const double* c_h = c + (size_t)h * K1;
        for (int idx = 0; idx < n; ++idx) {
            const double* row = basis_ptr + (size_t)idx * bases;
            double acc = 0.0;
            for (int i = 0; i < bases; ++i) acc += row[i] * c_h[i];
            out[idx] += acc;
        }
    }
    for (int idx = 0; idx < n; ++idx) out[idx] *= gamma;
    return out_arr;
}

PYBIND11_MODULE(_ga2m_cpp, m) {
    m.doc() = "Optional C++ accelerator for kanboost.train.ga2m's predict-time forward pass.";
    m.def("layer0_forward_shared", &layer0_forward_shared,
          "GA2M layer0 forward (shared per-feature basis computation)");
    m.def("layer1_forward_sum", &layer1_forward_sum,
          "GA2M layer1 forward + weighted sum across hidden units");
}
