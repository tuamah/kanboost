# Checking the additive assumption: `kanboost.interpret.interactions`

`gam=True` fits an exact additive model, `F(x) = c + sum_j g_j(x_j)` —
one shape function per feature, with no term for how two features
might interact. This is what makes [symbolic export](symbolic-export.md)
and [editable models](editing-dashboard.md) possible, but it's also a
real structural assumption: if the true relationship in your data
genuinely needs a cross-feature term (say, risk depends on `age * bmi`,
not `age` and `bmi` separately), `gam=True` cannot represent that,
no matter how well it's tuned.

`kanboost.interpret.interactions` answers *is that assumption actually true for
my data* with a real number, instead of leaving it as an unverified
assumption.

## Why you can't just check the `gam=True` model itself

Running an interaction statistic on a `gam=True` model directly always
comes back near zero — not because the data lacks interactions, but
because the architecture is *forced* additive and literally cannot
express one. Verified during development: a `gam=True` model on the
Breast Cancer dataset showed H-statistics of 0.01–0.05 across several
feature pairs — that's the noise floor, not evidence of "no
interactions in the data".

To see what's actually being left out, you need to measure interaction
strength on a **flexible** model fit to the *same* data instead
(`gam=False`) — that's what `check_additive_sufficiency()` does for
you automatically.

## One-call check: `check_additive_sufficiency()`

```python
from kanboost import KANBoostClassifier
from kanboost.interpret.interactions import check_additive_sufficiency

model = KANBoostClassifier(gam=True, kan_hidden=1, random_state=0)
model.fit(X_train, y_train)

result = check_additive_sufficiency(model, X_train, y_train, top_n=6, threshold=0.1)

print(result["verdict"])  # "additive_sufficient" or "interactions_detected"
for row in result["pairwise"]:
    print(row["feature_j"], row["feature_k"], row["h_gam"], row["h_flexible"], row["exceeds_threshold"])
```

It refits a `gam=False` counterpart internally (same hyperparameters,
same data), runs `friedman_h()` (below) on both for the `top_n` most
important features, and returns a verdict — `"interactions_detected"`
if any pair's `h_flexible` exceeds `threshold`, `"additive_sufficient"`
otherwise. `h_gam` is included for context (the noise floor), not
compared against the threshold itself.

Only meaningful on a `gam=True` model — raises `ValueError` on
`gam=False` (there's nothing being "left out" to check if the model
wasn't additive to begin with).

**On `threshold=0.1`**: calibrated against what was actually measured
during development, not picked arbitrarily. On Breast Cancer, `gam=True`
sits at 0.01–0.05 (noise), a real-but-modest interaction there reached
0.06–0.10, and a synthetic sanity check (a hand-written function with
an exact multiplicative interaction, see below) reached 0.9996. `0.1`
sits just above Breast Cancer's own modest signal — adjust it for your
own data's scale if your results look different.

## The underlying statistic: `friedman_h()`

Friedman's H-statistic (Friedman & Popescu, 2008) for a pair of
features `(j, k)`, from partial dependence:

```
H_jk^2 = sum_i [PD_jk(x_j^i, x_k^i) - PD_j(x_j^i) - PD_k(x_k^i)]^2
         / sum_i PD_jk(x_j^i, x_k^i)^2
```

(all three partial-dependence functions centered to mean zero first).
`H` close to 0 means the pair's joint effect on the model's output is
additive; close to 1 means it's almost entirely interaction.

```python
from kanboost.interpret.interactions import friedman_h

result = friedman_h(flexible_model, X, features=["age", "bmi"])
result["pairwise"]   # {(f_j, f_k): H_jk}
result["ranked"]      # [(f_j, f_k, H_jk), ...] sorted descending
```

Model-agnostic by design — works on **any** fitted estimator with
`predict_proba` or `predict`, not just KANBoost, via
`sklearn.inspection.partial_dependence`.

**Verified against a known-exact function** (`a*b + 0.1*c` — a genuine
interaction between `a`/`b`, none with `c`): `H(a, b) = 0.9996`,
`H(a, c) = H(b, c) ≈ 0.02` — confirming the formula and interpolation
are correct, independent of any real model's own fitting noise.

**Caveat, also verified during development**: tree ensembles
(RandomForest/XGBoost) can show a *spuriously* elevated H (0.5+) for
feature pairs with no true interaction, purely from partial-dependence
estimation noise on a moderate sample. Read a real model's H values
with that in mind, especially with few samples or a small ensemble.
