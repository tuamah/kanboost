"""kanboost.core.kan — numpy/scipy KAN, no torch/pykan dependency.

Drop-in for `from kan import KAN` and `from kan.utils import fit_params, SYMBOLIC_LIB`
and `from kan.spline import coef2curve, curve2coef, extend_grid`.
"""

from kanboost.core.kan.network import KAN, DeepKAN
from kanboost.core.kan.bspline import fit_params, SYMBOLIC_LIB, coef2curve, curve2coef, extend_grid

__all__ = [
    "KAN",
    "DeepKAN",
    "fit_params",
    "SYMBOLIC_LIB",
    "coef2curve",
    "curve2coef",
    "extend_grid",
]
