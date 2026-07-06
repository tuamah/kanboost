"""
kanboost.observability -- timing, memory, GPU, and per-round metrics.

This module never imports anything from `_base`/`classifier`/`regressor`
and never edits them. Everything here works from the *outside*: either
by calling a model's already-public methods (`.predict()`, `.device_`),
or by temporarily monkeypatching one instance's bound `_fit_learner`
method for the duration of a `with capture_boosting_rounds(model):`
block (restored in a `finally`, even if `fit()` raises). If a future
kanboost release renames `_fit_learner`, only this module's per-round
capture breaks -- prediction timing, memory, and GPU flags keep working
regardless, since they don't depend on any private method name.

Zero new dependencies for the timing/memory/GPU functions (stdlib +
torch, already required by kanboost). `memory_snapshot()` uses `psutil`
for accurate current RSS if it's installed, falling back to the
stdlib's `resource.getrusage` (peak RSS, Unix-only) otherwise.
"""

from __future__ import annotations

import contextlib
import io
import re
import time
from dataclasses import dataclass

import torch


@dataclass
class PredictionMetrics:
    elapsed_seconds: float
    n_samples: int
    samples_per_second: float
    device: str


def time_predict(model, X, method: str = "predict") -> tuple:
    """Call `model.<method>(X)` and time it.

    Returns `(result, PredictionMetrics)`. `method` is typically
    "predict" or "predict_proba".
    """
    fn = getattr(model, method)
    t0 = time.perf_counter()
    result = fn(X)
    elapsed = time.perf_counter() - t0
    n = len(X)
    return result, PredictionMetrics(
        elapsed_seconds=elapsed,
        n_samples=n,
        samples_per_second=(n / elapsed) if elapsed > 0 else float("inf"),
        device=str(getattr(model, "device_", "unknown")),
    )


@dataclass
class MemorySnapshot:
    rss_mb: float | None
    rss_is_peak: bool  # True when using a peak-memory fallback (Unix `resource`)
    gpu_allocated_mb: float | None
    gpu_reserved_mb: float | None


def _current_rss_mb() -> tuple[float | None, bool]:
    """Best-effort current process RSS, in this order: psutil (accurate,
    cross-platform) -> Windows ctypes/psapi -> Unix `resource` (peak, not
    current) -> None if nothing is available."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2), False
    except ImportError:
        pass

    try:
        import ctypes
        import ctypes.wintypes as wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        # Explicit argtypes/restype: ctypes defaults to c_int for both on
        # a bare windll call, which silently mis-marshals the pointer/DWORD
        # args on 64-bit Windows and makes the call fail every time.
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD,
        ]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 ** 2), False
    except (ImportError, AttributeError, OSError):
        pass

    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        return (raw / 1024 if raw > 1024 ** 2 else raw / (1024 ** 2)), True
    except ImportError:
        pass

    return None, False


def memory_snapshot() -> MemorySnapshot:
    """Process RSS memory (see `_current_rss_mb` for the fallback chain;
    `rss_mb` is `None` if no method worked) plus current CUDA allocator
    stats."""
    rss_mb, is_peak = _current_rss_mb()

    gpu_alloc = gpu_reserved = None
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 2)

    return MemorySnapshot(
        rss_mb=rss_mb, rss_is_peak=is_peak,
        gpu_allocated_mb=gpu_alloc, gpu_reserved_mb=gpu_reserved,
    )


def gpu_utilization_flag(model=None) -> dict:
    """Whether CUDA is available/visible, and (if a fitted model is
    passed) whether that model is actually running on it."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
    if model is not None:
        model_device = str(getattr(model, "device_", "unknown"))
        info["model_device"] = model_device
        info["model_on_gpu"] = model_device.startswith("cuda")
    return info


@dataclass
class RoundMetric:
    round: int
    elapsed_seconds: float
    gpu_allocated_mb: float | None
    loss: float | None  # val_loss if eval_set/validation_fraction is set, else None


_ROUND_LOG_RE = re.compile(r"val_loss=([\d.eE+-]+)")


@contextlib.contextmanager
def capture_boosting_rounds(model):
    """`with capture_boosting_rounds(model) as rounds: model.fit(X, y)`

    Yields a list that fills up with one `RoundMetric` per weak learner
    fit during any `.fit()` call inside the block (across every
    one-vs-rest chain for a multiclass classifier, in fit order).

    Implementation: temporarily wraps this *instance's* `_fit_learner`
    bound method to time each call and snapshot GPU memory, and
    temporarily forces `model.verbose = True` while capturing stdout to
    recover each round's `val_loss` from kanboost's existing verbose
    logging (parsed, not reprinted) -- no core file is modified, and
    both are restored in `finally` even if `fit()` raises.
    """
    if not hasattr(model, "_fit_learner"):
        raise AttributeError(
            "capture_boosting_rounds expects a KANBoostClassifier/Regressor-like "
            "estimator with a _fit_learner method; got "
            f"{type(model).__name__} with no such attribute."
        )

    records: list[RoundMetric] = []
    counter = {"i": 0}
    original_fit_learner = model._fit_learner
    original_verbose = getattr(model, "verbose", False)

    def instrumented(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_fit_learner(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        gpu_mb = (
            torch.cuda.memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available() else None
        )
        records.append(RoundMetric(
            round=counter["i"], elapsed_seconds=elapsed,
            gpu_allocated_mb=gpu_mb, loss=None,
        ))
        counter["i"] += 1
        return result

    model._fit_learner = instrumented
    model.verbose = True
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            yield records
    finally:
        model._fit_learner = original_fit_learner
        model.verbose = original_verbose

        losses = [float(m.group(1)) for m in _ROUND_LOG_RE.finditer(captured.getvalue())]
        for i, loss in enumerate(losses):
            if i < len(records):
                records[i].loss = loss
