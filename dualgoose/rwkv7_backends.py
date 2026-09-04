"""Backend selection for RWKV-style recurrent scans."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

from .rwkv7_ref import rwkv7_scan

ScanBackend = Literal["reference", "compiled", "triton", "triton_recompute"]
ScanFn = Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]]


def make_scan_fn(backend: str) -> ScanFn:
    """Return a scan callable for the requested backend."""

    if backend == "reference":
        return rwkv7_scan
    if backend == "compiled":
        if not hasattr(torch, "compile"):
            raise RuntimeError("scan_backend=compiled requires torch.compile")
        return torch.compile(rwkv7_scan, mode="reduce-overhead")
    if backend == "triton":
        from .rwkv7_triton import rwkv7_scan_triton

        return rwkv7_scan_triton
    if backend == "triton_recompute":
        from .rwkv7_triton import rwkv7_scan_triton_recompute

        return rwkv7_scan_triton_recompute
    raise ValueError(
        "scan_backend must be reference, compiled, triton, or triton_recompute"
    )
