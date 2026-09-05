"""
Measurement primitives for honest GF(2) benchmarking.

Methodology rules enforced here (see benchmarks/README.md for the rationale):

1. Time and memory are measured in SEPARATE passes. ``tracemalloc`` inflates
   wall time by 14-28x on allocation-heavy Python code, and it does so
   unevenly across implementations, so a timing taken while it is active is
   not a timing of the algorithm.
2. Every timing has a warmup call, then ``reps`` measured calls. We report
   ``min`` (the least noise-contaminated estimate of the true cost) alongside
   ``median`` and the spread, so an unstable measurement is visible instead of
   hidden inside a mean.
3. Contenders are checked for equivalent output before they are compared. A
   routine that returns one nullspace vector is not comparable to one that
   returns a full basis.
4. Setup (building the library's own matrix type from shared input data) is
   timed separately from the operation, and reported. Folding setup into the
   operation is legitimate only if every contender pays it.
"""

from __future__ import annotations

import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class Timing:
    """Wall-clock result for one contender, in milliseconds."""

    min_ms: float
    median_ms: float
    p90_ms: float
    reps: int
    samples: list[float] = field(repr=False, default_factory=list)


@dataclass
class Memory:
    """Peak *allocation* during one call, in MiB, plus resident data size."""

    peak_mib: float
    resident_mib: float | None = None


def time_call(fn: Callable[..., Any], *args, reps: int = 7, warmup: int = 1) -> Timing:
    """Time ``fn(*args)`` with no profiler attached.

    ``tracemalloc`` must not be running; see the module docstring.
    """
    if tracemalloc.is_tracing():
        raise RuntimeError("time_call must not run under tracemalloc (see module docstring)")

    for _ in range(warmup):
        fn(*args)

    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    return Timing(
        min_ms=samples[0],
        median_ms=statistics.median(samples),
        p90_ms=samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        reps=reps,
        samples=samples,
    )


def peak_alloc(fn: Callable[..., Any], *args) -> Memory:
    """Peak bytes allocated during a single ``fn(*args)``, measured alone.

    This is an *allocation* figure, not a steady-state footprint: it answers
    "how much scratch does this call need", which is the question that matters
    for a large elimination. For the size of the stored matrix itself use
    ``resident_bytes``.
    """
    fn(*args)  # warm caches / imports so they are not charged to the measurement
    tracemalloc.start()
    try:
        fn(*args)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return Memory(peak_mib=peak / (1024 * 1024))


def resident_bytes(obj: Any) -> int | None:
    """Best-effort steady-state size of a matrix object's data buffers."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    for attr in ("memory_usage",):
        if hasattr(obj, attr):
            try:
                return int(getattr(obj, attr)().memory_bytes)
            except Exception:
                pass
    if isinstance(getattr(obj, "data", None), np.ndarray):
        return int(obj.data.nbytes)
    return None


@dataclass
class Contender:
    """One implementation of one operation, plus how to build its input."""

    name: str
    setup: Callable[[Any], Any]  # shared input -> library-native object
    run: Callable[[Any], Any]  # library-native object -> result
    normalize: Callable[[Any], Any]  # result -> comparable canonical form
    available: bool = True
    note: str = ""


def run_operation(
    label: str,
    payload: Any,
    contenders: list[Contender],
    reps: int = 7,
    reference: str | None = None,
) -> dict:
    """Measure every available contender on one shared ``payload``.

    Returns a plain dict so results can be serialised and the report
    regenerated from data rather than transcribed by hand.
    """
    out: dict[str, Any] = {"operation": label, "results": {}, "equivalence": {}}
    canon: dict[str, Any] = {}

    for c in contenders:
        if not c.available:
            out["results"][c.name] = {"skipped": "unavailable", "note": c.note}
            continue
        try:
            native = c.setup(payload)
            setup_t = time_call(c.setup, payload, reps=max(3, reps // 2))
            run_t = time_call(c.run, native, reps=reps)
            mem = peak_alloc(c.run, native)
            mem.resident_mib = (
                resident_bytes(native) / (1024 * 1024) if resident_bytes(native) is not None else None
            )
            canon[c.name] = c.normalize(c.run(native))
            out["results"][c.name] = {
                "setup": asdict(setup_t) | {"samples": None},
                "run": asdict(run_t) | {"samples": None},
                "memory": asdict(mem),
                "note": c.note,
            }
        except Exception as exc:  # a contender that errors is reported, never dropped
            out["results"][c.name] = {"error": f"{type(exc).__name__}: {exc}", "note": c.note}

    ref = reference if reference in canon else (next(iter(canon), None))
    out["reference"] = ref
    if ref is not None:
        for name, value in canon.items():
            out["equivalence"][name] = "match" if value == canon[ref] else "DIFFERS"
    return out


def format_table(op: dict) -> str:
    """Render one operation's results as a fixed-width table."""
    lines = [f"\n{op['operation']}", "-" * len(op["operation"])]
    hdr = (
        f"{'implementation':<26} {'run min':>10} {'median':>10} "
        f"{'setup min':>10} {'peak MiB':>9}  {'=?':<7} note"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    base = op["results"].get(op.get("reference") or "", {})
    base_min = base.get("run", {}).get("min_ms")

    for name, r in op["results"].items():
        if "skipped" in r:
            lines.append(f"{name:<26} {'skipped':>10}   {r.get('note', '')}")
            continue
        if "error" in r:
            lines.append(f"{name:<26} {'ERROR':>10}   {r['error']}")
            continue
        eq = op["equivalence"].get(name, "?")
        rel = ""
        if base_min:
            rel = f"  ({r['run']['min_ms'] / base_min:.2f}x)"
        lines.append(
            f"{name:<26} {r['run']['min_ms']:>10.3f} {r['run']['median_ms']:>10.3f} "
            f"{r['setup']['min_ms']:>10.3f} {r['memory']['peak_mib']:>9.3f}  {eq:<7} {r['note']}{rel}"
        )
    return "\n".join(lines)
