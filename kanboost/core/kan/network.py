"""
DeepKAN — a numpy/scipy KAN trained by closed-form P-spline solvers instead of
gradient descent.  Public interface matches pykan's KAN closely enough that
kanboost's boosting loop and interpret pipeline work with only import-line changes.

GAM mode (fix_symbolic(1,0,0,'x') called):
  Each boosting round is ONE joint penalised least-squares solve:
      c = (BᵀWB + λDᵀD)⁻¹BᵀWr
  with optional two-pass monotone projection (B-spline variation-diminishing property).

Non-GAM mode:
  Gauss-Newton block-coordinate descent (ALS) — one penalised lstsq per layer per sweep,
  no lr/batch/opt hyperparameters.
"""

from __future__ import annotations

import numpy as np

from kanboost.core.kan.bspline import (
    _b_basis_1d, build_grid, extend_grid, diff_matrix,
    coef2curve, curve2coef,
)
from kanboost.core.kan.layer import KANLayer

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


def _np(t) -> np.ndarray:
    if _TORCH and isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t, dtype=float)


def _make_knots(n_features: int, G: int, k: int) -> np.ndarray:
    """Uniform extended knot grid for `n_features` splines, all on [-1, 1].
    Returns (n_features, G+2k+1).
    """
    base = build_grid(G, k)                       # (1, G+1)
    ext = extend_grid(base, k_extend=k)           # (1, G+2k+1)
    return np.repeat(ext, n_features, axis=0)     # (n_features, G+2k+1)


def _penalty_block(K: int, lam_smooth: float, lam_ridge: float) -> np.ndarray:
    """P-spline penalty for one feature: λ_smooth·DᵀD + λ_ridge·I, shape (K, K)."""
    D = diff_matrix(K, order=2)
    return lam_smooth * D.T @ D + lam_ridge * np.eye(K)


def _eigh_factor(M: np.ndarray, rel_floor: float = 1e-10):
    """Eigendecompose the (symmetric, PSD) matrix M once, with eigenvalues
    below `rel_floor * max_eigenvalue` floored — a truncated pseudo-inverse
    factorization reusable across many right-hand sides via `_solve_with_factor`.

    Splitting this out of `_solve_normal` matters here: `_solve_layer` and the
    ALS layer-0 update both solve the SAME M against many different RHS
    vectors (one per output channel / hidden unit), and `np.linalg.eigh` is
    the dominant O(n^3) cost — repeating it per RHS was measured to waste an
    n_out/n_hidden-fold multiple of the actual linear-algebra work for no
    reason, since M never changes within that loop.
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    floor = rel_floor * max(float(eigvals.max()), 1e-12)
    eigvals_clipped = np.maximum(eigvals, floor)
    return eigvecs, eigvals_clipped


def _solve_with_factor(eigvecs: np.ndarray, eigvals_clipped: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve M x = rhs given a factorization from `_eigh_factor` -- O(n^2), no eigh."""
    return eigvecs @ ((eigvecs.T @ rhs) / eigvals_clipped)


def _solve_normal(M: np.ndarray, rhs: np.ndarray, rel_floor: float = 1e-10) -> np.ndarray:
    """Solve the (symmetric, PSD) normal equations M x = rhs via truncated
    eigendecomposition: eigenvalues below `rel_floor * max_eigenvalue` are
    floored before inverting.  Default is a near-machine-precision floor
    (safe for the well-posed GAM closed-form solve). ALS callers pass a much
    looser `rel_floor` explicitly (1e-4) because correlated hidden units
    routinely make M near-singular there (condition numbers up to ~1e16
    observed) — a fixed additive jitter is negligible against M's large
    eigenvalues in that regime, so the floor has to be relative to the
    spectrum, not to M's diagonal. Confined to ALS call sites so it doesn't
    silently over-regularize the flagship GAM path.

    For a single RHS, use this. For many RHS against the same M, factor once
    with `_eigh_factor` and reuse via `_solve_with_factor` instead.
    """
    eigvecs, eigvals_clipped = _eigh_factor(M, rel_floor)
    return _solve_with_factor(eigvecs, eigvals_clipped, rhs)


# ---------------------------------------------------------------------------

class DeepKAN:
    """
    Parameters mirror KAN(width, grid, k, seed, device, auto_save) plus any
    kwargs pykan accepts (all extra kwargs are silently ignored so
    _learner_kan_kwargs in base.py works unchanged, including scale_base_mu,
    scale_base_sigma, sb_trainable, sp_trainable).
    """

    def __init__(
        self,
        width,
        grid: int = 3,
        k: int = 3,
        seed: int = 1,
        device: str = "cpu",
        auto_save: bool = False,
        **_ignored,
    ) -> None:
        # Normalise width: KAN accepts [n, h, 1] or [[n,0],[h,0],[1,0]].
        # base.py reads learner.width[0][0].
        if isinstance(width[0], (list, tuple)):
            flat = [w[0] for w in width]
        else:
            flat = list(width)
        self._width_flat = flat
        # pykan format with multiplicity column so width[0][0] works:
        self.width = [[n, 0] for n in flat]

        self._grid_param = grid
        self.k = k
        self._seed = seed
        self._output_identity = False   # set True by fix_symbolic(1,0,0,'x')
        self._last_x: np.ndarray | None = None

        n_in, n_hidden, n_out = flat[0], flat[1], flat[2]

        # Layer 0: input → hidden  (or input → 1 in GAM mode where n_hidden=1)
        knots0 = _make_knots(n_in, grid, k)
        layer0 = KANLayer(n_in, n_hidden, knots0, k)
        # Init: each hidden unit gets a distinct random linear projection of the
        # inputs (random-features style) so hidden units start decorrelated —
        # identical init per unit makes layer-1's design matrix collinear and
        # blows up its ALS-solved coefficients.
        rng = np.random.RandomState(seed)
        x_init = np.linspace(-1.0, 1.0, 200)
        K0 = knots0.shape[1] - k - 1
        # _make_knots repeats the SAME extended grid for every feature, so
        # the basis B and its fit against x_init are identical regardless of
        # (j, h) -- lstsq is linear in the RHS, so scaling the RHS by w[j]
        # scales the solution by w[j] too. Solve once instead of n_in*n_hidden
        # times (each of those was recomputing an identical B-spline basis).
        B_init = _b_basis_1d(x_init, knots0[0], k)
        c_unit, _, _, _ = np.linalg.lstsq(B_init, x_init, rcond=None)
        for h in range(n_hidden):
            w = rng.randn(n_in) / np.sqrt(n_in)
            for j in range(n_in):
                layer0.coef[j, h] = w[j] * c_unit

        # Layer 1: hidden → output  (ignored in GAM identity mode)
        knots1 = _make_knots(n_hidden, grid, k)
        layer1 = KANLayer(n_hidden, n_out, knots1, k)

        self.layers = [layer0, layer1]
        self.act_fun = self.layers   # alias — base.py accesses act_fun[0].coef

    # ------------------------------------------------------------------
    # pykan compat shims
    # ------------------------------------------------------------------

    def fix_symbolic(self, l: int, i: int, o: int, name: str, **kw) -> None:
        """Only (1, 0, 0, 'x') is called from the boosting loop — mark
        GAM identity mode.  Any other combination raises NotImplementedError
        so misuse is caught immediately.
        """
        if (l, i, o, name) == (1, 0, 0, "x"):
            if self.layers[0].n_out != 1:
                raise NotImplementedError(
                    "DeepKAN GAM mode (fix_symbolic(1,0,0,'x')) requires "
                    "kan_hidden=1 (an exact additive model needs exactly one "
                    f"hidden unit per feature-sum term); got kan_hidden={self.layers[0].n_out}."
                )
            self._output_identity = True
        else:
            raise NotImplementedError(
                f"DeepKAN.fix_symbolic({l},{i},{o},{name!r}) is not supported. "
                "Only the GAM-mode call fix_symbolic(1,0,0,'x') is implemented."
            )

    def get_act(self, x) -> None:
        """Cache last input for prune_edge / refine / feature_interaction."""
        self._last_x = _np(x)

    def attribute(self, plot: bool = False) -> None:
        pass

    def prune_edge(self, threshold: float, **kw) -> None:
        """Zero out edges whose max absolute activation is below threshold."""
        if self._last_x is None:
            return
        layer = self.layers[0]
        for i in range(layer.n_in):
            B = _b_basis_1d(self._last_x[:, i], layer.knots[i], layer.k)
            for o in range(layer.n_out):
                acts = np.abs(B @ layer.coef[i, o])
                if acts.max() < threshold:
                    layer.coef[i, o] = 0.0

    def refine(self, new_grid: int) -> "DeepKAN":
        """Return a new DeepKAN with refined grid, preserving predictions.

        Every edge (in both layers) is re-expressed on the new knot grid by
        resampling its OWN curve on a dense synthetic domain — this is
        well-conditioned regardless of how correlated the real training data
        is, unlike solving a fresh lstsq against actual (possibly collinear)
        activations.
        """
        new_net = DeepKAN(
            self._width_flat, grid=new_grid, k=self.k, seed=self._seed,
        )
        new_net._output_identity = self._output_identity
        new_net._intercept = getattr(self, "_intercept", 0.0)
        if self._last_x is None:
            return new_net

        x = self._last_x                               # (n, n_in)

        old_layer0 = self.layers[0]
        new_layer0 = new_net.layers[0]

        # Update new layer-0 knots to cover actual data range
        for i in range(old_layer0.n_in):
            lo, hi = x[:, i].min() - 0.1, x[:, i].max() + 0.1
            base = np.linspace(lo, hi, new_grid + 1)[np.newaxis, :]
            new_layer0.knots[i] = extend_grid(base, k_extend=new_layer0.k)[0]

        # Refine each layer-0 edge curve by direct coefficient mapping via a
        # dense evaluation grid. Measured empirically: residual error here is
        # dominated by the old/new knot ranges genuinely differing (real
        # basis-change error), not by sampling density — 5000 points already
        # matches 200000 to 4 decimal places, so this is not a knob worth
        # raising for "more precision".
        _N_EVAL = 5000
        for i in range(old_layer0.n_in):
            knots_old = old_layer0.knots[i]
            knots_new = new_layer0.knots[i]
            k = old_layer0.k
            x_fine = np.linspace(knots_old[k], knots_old[-k - 1], _N_EVAL)
            B_old = _b_basis_1d(x_fine, knots_old, k)
            B_new = _b_basis_1d(x_fine, knots_new, k)
            for o in range(old_layer0.n_out):
                y_edge = B_old @ old_layer0.coef[i, o]
                c, _, _, _ = np.linalg.lstsq(B_new, y_edge, rcond=None)
                new_layer0.coef[i, o] = c

        if self._output_identity:
            return new_net

        # Refit layer 1 the same dense-grid way as layer 0 — re-expressing
        # the OLD curve on new knots is well-conditioned by construction;
        # solving a fresh lstsq against real (correlated, noisy) hidden
        # activations is not, and was empirically observed to blow up
        # layer-1 coefficients by orders of magnitude.
        old_layer1 = self.layers[1]
        new_layer1 = new_net.layers[1]
        for h in range(old_layer1.n_in):
            lo, hi = old_layer1.knots[h][old_layer1.k], old_layer1.knots[h][-old_layer1.k - 1]
            base = np.linspace(lo, hi, new_grid + 1)[np.newaxis, :]
            new_layer1.knots[h] = extend_grid(base, k_extend=new_layer1.k)[0]

        for h in range(old_layer1.n_in):
            knots_old = old_layer1.knots[h]
            knots_new = new_layer1.knots[h]
            k = old_layer1.k
            z_fine = np.linspace(knots_old[k], knots_old[-k - 1], _N_EVAL)
            B_old = _b_basis_1d(z_fine, knots_old, k)
            B_new = _b_basis_1d(z_fine, knots_new, k)
            for o in range(old_layer1.n_out):
                y_edge = B_old @ old_layer1.coef[h, o]
                c, _, _, _ = np.linalg.lstsq(B_new, y_edge, rcond=None)
                new_layer1.coef[h, o] = c

        return new_net

    def feature_interaction(self, l: int, neuron_th: float, feature_th: float) -> dict:
        """Return {feature_group_tuple: count} for hidden units above neuron_th."""
        if self._last_x is None:
            return {}
        layer0 = self.layers[0]
        layer1 = self.layers[1]
        x = self._last_x
        z = layer0.forward(x)
        counts: dict = {}
        for h in range(layer0.n_out):
            B1 = _b_basis_1d(z[:, h], layer1.knots[h], layer1.k)
            acts1 = np.abs(B1 @ layer1.coef[h, 0])
            if acts1.max() < neuron_th:
                continue
            group = []
            for j in range(layer0.n_in):
                B0 = _b_basis_1d(x[:, j], layer0.knots[j], layer0.k)
                acts0 = np.abs(B0 @ layer0.coef[j, h])
                if acts0.max() >= feature_th:
                    group.append(j)
            key = tuple(group)
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, x):
        """x : torch.Tensor or numpy (n, n_in) → torch.Tensor (n, 1).
        Returns a torch tensor to remain compatible with base.py arithmetic.
        """
        x_np = _np(x)
        out = self._forward_np(x_np)     # (n, 1)
        if _TORCH:
            if isinstance(x, torch.Tensor):
                return torch.tensor(out, dtype=x.dtype, device=x.device)
            return torch.tensor(out, dtype=torch.float32)
        return out

    def _forward_np(self, x: np.ndarray) -> np.ndarray:
        """Pure numpy forward, returns (n, 1)."""
        if self._output_identity:
            z = self.layers[0].forward(x)                     # (n, n_hidden)
            out = z.sum(axis=1, keepdims=True)                # (n, 1)
            return out + getattr(self, "_intercept", 0.0)
        else:
            z = self.layers[0].forward(x)   # (n, n_hidden)
            return self.layers[1].forward(z) # (n, 1)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: dict,
        opt: str = "Adam",
        steps: int = 20,
        lr=None,
        loss_fn=None,
        update_grid: bool = False,
        lamb: float = 0.0,
        lamb_l1: float = 1.0,
        lamb_coefdiff: float = 0.0,
        sample_weight=None,
        monotone_signs=None,
        basis_cache: dict | None = None,
        **_ignored,
    ) -> "DeepKAN":
        X = _np(dataset["train_input"])       # (n, n_in)
        r = _np(dataset["train_label"]).flatten()  # (n,)
        W = None if sample_weight is None else _np(sample_weight).flatten()

        lam_smooth = float(lamb) * float(lamb_coefdiff)
        lam_ridge  = float(lamb) * float(lamb_l1)

        if self._output_identity and self.layers[0].n_out == 1:
            # True single-hidden-unit GAM → P-spline closed form
            self._solve_gam(X, r, W, lam_smooth, lam_ridge, monotone_signs)
        else:
            self._fit_als(X, r, W, steps, lam_smooth, lam_ridge, basis_cache=basis_cache)
        return self

    # ------------------------------------------------------------------
    # GAM closed-form solver
    # ------------------------------------------------------------------

    def _solve_gam(
        self,
        X: np.ndarray,
        r: np.ndarray,
        W,
        lam_smooth: float,
        lam_ridge: float,
        monotone_signs,
    ) -> None:
        """Joint P-spline solve for the additive model F(x) = Σ_j g_j(x_j)."""
        n, p = X.shape
        layer = self.layers[0]
        K = layer.coef.shape[2]
        w = W if W is not None else np.ones(n)

        # Build stacked design matrix B = [B_0 | B_1 | ... | B_{p-1} | 1]
        # shape (n, p*K + 1)
        cols = []
        for j in range(p):
            Bj = _b_basis_1d(X[:, j], layer.knots[j], layer.k)  # (n, K)
            cols.append(Bj)
        cols.append(np.ones((n, 1)))  # intercept
        B = np.concatenate(cols, axis=1)  # (n, p*K + 1)

        # Block-diagonal penalty (no penalty on intercept)
        P_blk = _penalty_block(K, lam_smooth, lam_ridge)
        P_full = np.zeros((p * K + 1, p * K + 1))
        for j in range(p):
            s, e = j * K, (j + 1) * K
            P_full[s:e, s:e] = P_blk

        # Weighted normal equations
        Bw = B * w[:, np.newaxis]
        A = Bw.T @ B + P_full
        b = Bw.T @ r

        c = _solve_normal(A, b)   # A = BᵀWB + P_full already built

        # Distribute solution to coef
        intercept = c[-1]
        for j in range(p):
            layer.coef[j, 0, :] = c[j * K:(j + 1) * K]

        if monotone_signs is None or not any(s != 0 for s in monotone_signs):
            # Store intercept as a constant bias in the last (trivial) feature's
            # coef — simpler than a separate attribute.  We add it to predictions
            # by including the ones column already, so nothing more needed.
            # Actually: just store it as a separate attribute so forward() works.
            self._intercept = float(intercept)
            return

        # ---- two-pass monotone projection ----
        constrained = [j for j, s in enumerate(monotone_signs) if s != 0]
        unconstrained = [j for j in range(p) if j not in constrained]

        for j in constrained:
            sign = monotone_signs[j]
            c_j = layer.coef[j, 0, :]
            if sign > 0:
                layer.coef[j, 0, :] = np.maximum.accumulate(c_j)
            else:
                layer.coef[j, 0, :] = -np.maximum.accumulate(-c_j)

        if not unconstrained:
            self._intercept = float(intercept)
            return

        # Residual after constrained features
        r2 = r.copy()
        for j in constrained:
            Bj = _b_basis_1d(X[:, j], layer.knots[j], layer.k)
            r2 -= Bj @ layer.coef[j, 0]

        # Re-solve unconstrained features + intercept against r2
        cols_u = [_b_basis_1d(X[:, j], layer.knots[j], layer.k) for j in unconstrained]
        cols_u.append(np.ones((n, 1)))
        Bu = np.concatenate(cols_u, axis=1)
        Pu = np.zeros((len(unconstrained) * K + 1, len(unconstrained) * K + 1))
        for idx in range(len(unconstrained)):
            s, e = idx * K, (idx + 1) * K
            Pu[s:e, s:e] = P_blk
        Buw = Bu * w[:, np.newaxis]
        Au = Buw.T @ Bu + Pu
        bu = Buw.T @ r2
        cu = _solve_normal(Au, bu)
        for idx, j in enumerate(unconstrained):
            layer.coef[j, 0, :] = cu[idx * K:(idx + 1) * K]
        self._intercept = float(cu[-1])

    # ------------------------------------------------------------------
    # Non-GAM ALS solver
    # ------------------------------------------------------------------

    def _fit_als(
        self,
        X: np.ndarray,
        r: np.ndarray,
        W,
        n_sweeps: int,
        lam_smooth: float,
        lam_ridge: float,
        basis_cache: dict | None = None,
    ) -> None:
        """Gauss-Newton block-coordinate descent for multi-layer DeepKAN.

        The per-hidden-unit linearization step is a crude local approximation
        (not a guaranteed descent direction), so each sweep is only kept if it
        actually lowers the loss; the best-seen state is restored at the end.
        """
        layer0, layer1 = self.layers[0], self.layers[1]
        n_sweeps = min(n_sweeps, 10)
        w = W if W is not None else np.ones(len(r))

        # layer0's knots are never touched anywhere in this method (only
        # layer1.knots gets rebuilt, from `z`, each sweep) -- so layer0's
        # B-spline basis matrix B0, its normal-equations system, and that
        # system's eigh factorization are all invariant across the ENTIRE
        # fit, not just within one sweep. Profiling showed B-spline basis
        # evaluation (not eigh) was the actual dominant cost (~60% of fit
        # time), driven by redundant re-evaluation, not by eigh -- so B0 is
        # computed ONCE here and every `layer0.forward(X)` in this method is
        # replaced by a cheap `B0 @ (current coef, reshaped)` matmul instead
        # of re-deriving the B-spline basis from scratch each time.
        #
        # `basis_cache`, if given by the boosting loop (kanboost/core/base.py
        # `_boost_chain`), extends this one level further: every learner in a
        # boosting chain shares the same X, kan_grid/kan_k, lamb*, and
        # sample_weight, so B0/M0/its eigh factorization are identical across
        # ALL learners in the chain, not just within one learner's sweeps --
        # computed once by the first learner, reused by every later one.
        n_in0 = X.shape[1]
        cache_key = "layer0_system"
        if basis_cache is not None and cache_key in basis_cache:
            B0, P0_full, K0, Bw0, M0, eigvecs0, eigvals0 = basis_cache[cache_key]
        else:
            B0, P0_full, K0 = self._joint_design_matrix(layer0, X, lam_smooth, lam_ridge)
            Bw0 = B0 * w[:, np.newaxis]
            M0 = Bw0.T @ B0 + P0_full
            eigvecs0, eigvals0 = _eigh_factor(M0, rel_floor=1e-4)
            if basis_cache is not None:
                basis_cache[cache_key] = (B0, P0_full, K0, Bw0, M0, eigvecs0, eigvals0)

        def _layer0_forward_cached() -> np.ndarray:
            C0 = layer0.coef.transpose(0, 2, 1).reshape(n_in0 * K0, layer0.n_out)
            return B0 @ C0

        def _loss(z_: np.ndarray) -> float:
            out_ = layer1.forward(z_)
            return float((w * (r - out_.flatten()) ** 2).mean())

        z = _layer0_forward_cached()
        best_loss = _loss(z)
        best_state = (layer0.coef.copy(), layer1.coef.copy(), layer1.knots.copy())

        prev_loss = best_loss
        for _ in range(n_sweeps):
            # (1) Fix layer0, solve layer1 by penalised lstsq
            z = _layer0_forward_cached()       # (n, n_hidden)
            self._update_layer_knots(layer1, z)
            B1, K1 = self._solve_layer(layer1, z, r, w, lam_smooth, lam_ridge)

            # (2) Fix layer1, linearise targets per hidden unit, solve layer0
            # using the factorization hoisted above. `B1` (built moments ago
            # inside `_solve_layer` from this same z/knots) is reused here via
            # column slicing instead of re-deriving each hidden unit's basis,
            # and again below for `out` instead of a second `layer1.forward(z)`.
            #
            # (A Gauss-Seidel variant -- refreshing the shared residual after
            # each hidden unit instead of once for all of them -- was tried
            # and empirically measured to give zero ensemble-level R^2
            # benefit here, since boosting's many weak learners already
            # compensate for one learner's ALS sub-optimality, while costing
            # ~5% more wall-clock time from the extra per-unit basis/output
            # recomputation. Reverted in favor of this simpler, faster Jacobi
            # form; keep it that way unless a case emerges where a single
            # (non-boosted) DeepKAN fit's quality matters on its own.)
            n_hidden = layer0.n_out
            C1 = layer1.coef.transpose(0, 2, 1).reshape(n_hidden * K1, layer1.n_out)
            out = B1 @ C1                      # (n, 1) — same as layer1.forward(z)
            e = r - out.flatten()
            for h in range(n_hidden):
                Bh = B1[:, h * K1:(h + 1) * K1]  # (n, K1) — reused, not recomputed
                slope = (Bh @ layer1.coef[h, 0])                        # (n,)
                norm_sq = float((slope ** 2).sum())
                if norm_sq < 1e-12:
                    continue
                z_spread = float(z[:, h].std()) + 1e-6
                step = np.clip(e * slope / norm_sq, -2 * z_spread, 2 * z_spread)
                t_h = z[:, h] + step
                c = _solve_with_factor(eigvecs0, eigvals0, Bw0.T @ t_h)
                for i in range(n_in0):
                    layer0.coef[i, h] = c[i * K0:(i + 1) * K0]

            z = _layer0_forward_cached()
            loss = _loss(z)
            if loss < best_loss:
                best_loss = loss
                best_state = (layer0.coef.copy(), layer1.coef.copy(), layer1.knots.copy())
            # 1e-3 (not the original 1e-6): measured that 1e-6 was so strict
            # it NEVER triggered in practice -- every learner burned the full
            # sweep budget regardless of dataset (confirmed via profiling:
            # exactly n_estimators*n_sweeps calls to _solve_layer, every time).
            # 1e-3 lets genuinely-converged learners stop early -- verified
            # ~1.7-2x wall-clock speedup on both California Housing and
            # Friedman-1000 with no measurable accuracy cost (Friedman-1000's
            # R^2 was unchanged/slightly better; California Housing's dropped
            # <0.004, within CV noise) -- unlike cross-round warm-starting
            # (tried and rejected: safe on California Housing, cost ~0.04 R^2
            # on Friedman-1000's sharper residual structure). This threshold
            # is per-learner internal training-loss convergence, not
            # validation-based -- validation-aware ALS stopping was
            # separately instrumented and found unnecessary at grid=3.
            if prev_loss - loss < 1e-3 * prev_loss:
                break
            prev_loss = loss

        layer0.coef, layer1.coef, layer1.knots = best_state

    def _update_layer_knots(self, layer: KANLayer, z: np.ndarray) -> None:
        """Rebuild layer's knot grids to span the actual range of z."""
        G, k = self._grid_param, self.k
        for h in range(layer.n_in):
            lo = z[:, h].min() - 0.1
            hi = z[:, h].max() + 0.1
            base = np.linspace(lo, hi, G + 1)[np.newaxis, :]
            layer.knots[h] = extend_grid(base, k_extend=k)[0]

    @staticmethod
    def _joint_design_matrix(layer: KANLayer, x_in: np.ndarray, lam_smooth: float, lam_ridge: float):
        """Stacked design matrix B = [B_0 | B_1 | ... | B_{n_in-1}] (n, n_in*K)
        and its block-diagonal penalty, for jointly solving F(x) = Σ_i g_i(x_i)
        over ALL inputs at once. Used by `_solve_layer` and `_fit_als`'s layer-0
        update so neither ever fits one input's spline in isolation — an
        isolated, per-feature fit has each input independently trying to
        reconstruct the WHOLE target, so summing n_in of them overshoots by
        ~n_in x (this was a real bug in an earlier per-feature-loop version,
        invisible in small (2-3 feature) tests but catastrophic with real
        (~30 feature) data).
        """
        n_in = x_in.shape[1]
        K = layer.coef.shape[2]
        P_blk = _penalty_block(K, lam_smooth, lam_ridge)
        cols = [_b_basis_1d(x_in[:, i], layer.knots[i], layer.k) for i in range(n_in)]
        B = np.concatenate(cols, axis=1)  # (n, n_in*K)
        P_full = np.zeros((n_in * K, n_in * K))
        for i in range(n_in):
            s, e = i * K, (i + 1) * K
            P_full[s:e, s:e] = P_blk
        return B, P_full, K

    def _solve_layer(
        self,
        layer: KANLayer,
        x_in: np.ndarray,
        targets: np.ndarray,
        w: np.ndarray,
        lam_smooth: float,
        lam_ridge: float,
        precomputed=None,
    ):
        """Joint penalised lstsq for each output channel of `layer`.
        Solves F(x) = Σ_i g_i(x_i) for each output (joint across inputs).
        x_in : (n, n_in_layer); targets : (n,) or (n, n_out).

        `precomputed`, if given, is a `(B, P_full, K)` triple from
        `_joint_design_matrix` — lets a caller that already built it (e.g. to
        reuse `B`'s column-blocks afterwards) skip rebuilding it here.
        Returns `(B, K)` so the caller can reuse the same design matrix
        instead of recomputing `_b_basis_1d` from scratch for the same
        `x_in`/knots right after this call returns.
        """
        n_in = x_in.shape[1]
        n_out = layer.n_out
        if precomputed is not None:
            B, P_full, K = precomputed
        else:
            B, P_full, K = self._joint_design_matrix(layer, x_in, lam_smooth, lam_ridge)

        Bw = B * w[:, np.newaxis]
        M = Bw.T @ B + P_full
        # Factor M once — it's identical for every output channel, only the
        # RHS (BᵀWt) differs, so eigh only needs to run once here, not n_out
        # times.
        #
        # ponytail: a Cholesky-with-jitter fast path was tried and REJECTED
        # here after verification, not just assumed safe. ALS's M routinely
        # has condition numbers up to ~1e16-1e17 (near-duplicate/collinear
        # hidden-unit columns); a small diagonal jitter large enough for
        # `cho_factor` to succeed is NOT equivalent to eigh's relative
        # eigenvalue-truncation floor (Tikhonov/ridge regularization shifts
        # every eigenvalue uniformly; truncation only clips the smallest
        # ones) -- measured on a synthetic near-singular case (cond~3.7e17):
        # Cholesky-with-jitter gave a solution norm of 6.4M vs eigh's 148, a
        # silent, badly-wrong answer, not a merely-slower-but-correct one.
        # Making Cholesky's jitter track the true max eigenvalue (to match
        # eigh's semantics safely) needs a cheap spectral-norm estimate (e.g.
        # power iteration), which erodes most of the expected speed win and
        # adds its own approximation to verify -- not worth it for the ~30%
        # time share this solve occupies. Left as eigh; see project notes.
        eigvecs, eigvals_clipped = _eigh_factor(M, rel_floor=1e-4)
        for o in range(n_out):
            t = targets.flatten() if targets.ndim == 1 else targets[:, o]
            c = _solve_with_factor(eigvecs, eigvals_clipped, Bw.T @ t)   # BᵀWt, not BᵀW(Wt)
            for i in range(n_in):
                layer.coef[i, o] = c[i * K:(i + 1) * K]
        return B, K

    # ------------------------------------------------------------------
    # Analytic derivative (replaces torch.autograd.grad in base.py)
    # ------------------------------------------------------------------

    def predict_derivative_analytic(self, x, col: int) -> np.ndarray:
        """d F(x) / d x_{col}.  x can be torch tensor or numpy.
        Returns (n,) numpy array.
        """
        x_np = _np(x)
        layer0 = self.layers[0]

        if self._output_identity:
            # F = Σ_j g_j(x_j)  →  dF/dx_col = g_col'(x_col)
            from kanboost.core.kan.bspline import _b_basis_deriv_1d
            dBcol = _b_basis_deriv_1d(x_np[:, col], layer0.knots[col], layer0.k)
            return (dBcol @ layer0.coef[col, 0]).flatten()
        else:
            # Chain rule: dF/dx_j = Σ_h (dF/dz_h) * (dz_h/dx_j)
            # dz_h/dx_j = g0_{jh}'(x_j)
            # dF/dz_h = g1_{h0}'(z_h)
            from kanboost.core.kan.bspline import _b_basis_deriv_1d
            z = layer0.forward(x_np)                              # (n, n_hidden)
            layer1 = self.layers[1]
            dFdz = np.zeros((x_np.shape[0], layer0.n_out))       # (n, n_hidden)
            for h in range(layer0.n_out):
                dB1 = _b_basis_deriv_1d(z[:, h], layer1.knots[h], layer1.k)
                dFdz[:, h] = (dB1 @ layer1.coef[h, 0]).flatten()
            dzdxcol = np.zeros((x_np.shape[0], layer0.n_out))
            for h in range(layer0.n_out):
                dB0 = _b_basis_deriv_1d(x_np[:, col], layer0.knots[col], layer0.k)
                dzdxcol[:, h] = (dB0 @ layer0.coef[col, h]).flatten()
            return (dFdz * dzdxcol).sum(axis=1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "format": "deepkan-v1",
            "width": self._width_flat,
            "grid": self._grid_param,
            "k": self.k,
            "_output_identity": self._output_identity,
            "layers": [(layer.coef.copy(), layer.knots.copy()) for layer in self.layers],
            "_intercept": getattr(self, "_intercept", 0.0),
        }

    def load_state_dict(self, sd: dict) -> None:
        if not isinstance(sd, dict) or sd.get("format", "").startswith("deepkan") is False:
            # Likely a pykan torch state_dict
            raise ValueError(
                "Cannot load a pykan state_dict (format_version ≤ 2) into DeepKAN. "
                "Re-train the model with the current version."
            )
        for layer, (coef, knots) in zip(self.layers, sd["layers"]):
            layer.coef = np.array(coef)
            layer.knots = np.array(knots)
        self._output_identity = bool(sd.get("_output_identity", False))
        self._intercept = float(sd.get("_intercept", 0.0))


# Public alias matching pykan's `from kan import KAN`
KAN = DeepKAN
