"""CUDA / device helpers.

Small, dependency-free utilities so that the rest of the package never has to
write ``if cuda: ...`` twice:

* :func:`resolve_device` — pick the best available device for a request;
* :func:`device_info` — name, capability, memory for logging;
* :func:`amp_context` — FP16 autocast for inference (training stays FP32:
  deterministic-policy updates are sensitive to half-precision critic gradients);
* :func:`maybe_compile` — optional ``torch.compile`` with a safe fallback;
* :func:`synchronize` / :func:`memory_report` — honest timing and footprint.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, Optional

import torch


def cuda_available() -> bool:
    return torch.cuda.is_available()


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """Return a usable device: honour ``cuda`` when present, otherwise CPU."""
    req = (requested or "cuda").lower()
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if ":" in req:
            return torch.device(req)
        return torch.device("cuda")
    return torch.device(req)


def device_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        idx = device.index or torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "memory_gb": round(props.total_memory / 1024 ** 3, 2),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        })
    else:
        info.update({"name": "cpu", "torch": torch.__version__, "cuda": None})
    return info


@contextlib.contextmanager
def amp_context(device: torch.device, enabled: bool = True):
    """FP16 autocast on CUDA, a no-op elsewhere or when disabled."""
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def maybe_compile(module: torch.nn.Module, enabled: bool = True,
                  mode: str = "reduce-overhead") -> torch.nn.Module:
    """``torch.compile`` with a fallback: a failed compile must never break a run."""
    if not enabled or not hasattr(torch, "compile"):
        return module
    try:
        return torch.compile(module, mode=mode)
    except Exception as exc:                                   # pragma: no cover
        print(f"[cuda] torch.compile unavailable ({exc}); falling back to eager")
        return module


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device: torch.device, iters: int = 50, warmup: int = 5) -> Dict[str, float]:
    """Wall-clock timing with warm-up, synchronisation and p50/p95 percentiles.

    Returns ``mean_ms``, ``p50_ms``, ``p95_ms`` and ``throughput`` (calls/second).
    """
    for _ in range(warmup):
        fn()
    synchronize(device)

    samples = []
    t0 = time.perf_counter()
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - t) * 1e3)
    total = time.perf_counter() - t0

    samples.sort()
    pick = lambda q: samples[min(len(samples) - 1, int(q * len(samples)))]
    return {"mean_ms": sum(samples) / len(samples), "p50_ms": pick(0.50),
            "p95_ms": pick(0.95), "throughput": iters / total}


def memory_report(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    return {"allocated_mb": torch.cuda.memory_allocated(device) / 1024 ** 2,
            "reserved_mb": torch.cuda.memory_reserved(device) / 1024 ** 2,
            "peak_mb": torch.cuda.max_memory_allocated(device) / 1024 ** 2}


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


__all__ = ["cuda_available", "resolve_device", "device_info", "amp_context",
           "maybe_compile", "synchronize", "timed", "memory_report",
           "reset_peak_memory"]
