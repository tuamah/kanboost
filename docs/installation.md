# Installation

```bash
pip install kanboost
```

Or from source:

```bash
git clone https://github.com/tuamah/kanboost.git
cd kanboost
pip install -e .
```

## Optional extras

KANBoost's core install is deliberately minimal. Two optional features
each have their own extra so you never pull in a dependency you don't
use:

```bash
pip install kanboost[api]         # FastAPI serving layer
pip install kanboost[dashboard]   # interactive Streamlit dashboard
```

Hyperparameter tuning lives in a separate sibling package, not an
extra, so it stays usable for tuning other model types too:

```bash
pip install kantun
```

See [Serving & observability](guide/serving.md), [Editable models &
dashboard](guide/editing-dashboard.md), and [Tuning with
kantun](guide/tuning-with-kantun.md) for what each of these unlocks.
