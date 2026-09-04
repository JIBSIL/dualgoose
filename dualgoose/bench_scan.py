"""Benchmark the Python reference RWKV-style scan."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from .rwkv7_backends import make_scan_fn
from .train_toy import git_metadata


def _triton_config(backend: str, d_model: int) -> dict[str, object]:
    if backend not in {"triton", "triton_recompute"}:
        return {
            "triton_block_dk": "",
            "triton_num_warps": "",
            "triton_num_stages": "",
            "triton_backward_block_dv": "",
            "triton_backward_mode": "",
            "triton_recompute_checkpoint_interval": "",
        }
    from .rwkv7_triton import triton_kernel_config

    config = triton_kernel_config(d_model)
    return {
        "triton_block_dk": config["block_dk"],
        "triton_num_warps": config["num_warps"],
        "triton_num_stages": config["num_stages"],
        "triton_backward_block_dv": config["backward_block_dv"],
        "triton_backward_mode": config["backward_mode"],
        "triton_recompute_checkpoint_interval": config[
            "recompute_checkpoint_interval"
        ],
    }


def make_inputs(
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    scale = d_model**-0.5
    v = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.tanh(torch.randn(batch_size, seq_len, d_model, device=device)) * scale
    r = torch.tanh(torch.randn(batch_size, seq_len, d_model, device=device)) * scale
    w = torch.sigmoid(torch.randn(batch_size, seq_len, d_model, device=device))
    kappa = torch.tanh(torch.randn(batch_size, seq_len, d_model, device=device)) * scale
    a = torch.sigmoid(torch.randn(batch_size, seq_len, d_model, device=device)) * w
    return v, k, r, w, kappa, a


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_case(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
    use_rank: bool,
    warmup: int,
    iters: int,
    compile_scan: bool = False,
    scan_backend: str | None = None,
    include_backward: bool = False,
) -> dict[str, object]:
    inputs = make_inputs(batch_size, seq_len, d_model, device)
    if include_backward:
        inputs = tuple(tensor.requires_grad_(True) for tensor in inputs)
    backend = scan_backend or ("compiled" if compile_scan else "reference")
    scan_fn = make_scan_fn(backend)

    def run_once() -> tuple[torch.Tensor, torch.Tensor | None]:
        for tensor in inputs:
            tensor.grad = None
        out = scan_fn(*inputs, use_rank=use_rank)
        if not include_backward:
            return out, None
        loss = out.square().mean()
        loss.backward()
        return out, loss

    for _ in range(warmup):
        run_once()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iters):
        out, loss = run_once()
    synchronize(device)
    elapsed = time.perf_counter() - start
    tokens = batch_size * seq_len * iters
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else None
    )
    grad_abs_mean = (
        float(inputs[0].grad.detach().abs().mean().cpu())
        if include_backward and inputs[0].grad is not None
        else None
    )
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "device": str(device),
        "use_rank": use_rank,
        "scan_backend": backend,
        **_triton_config(backend, d_model),
        "compile_scan": compile_scan,
        "include_backward": include_backward,
        "iters": iters,
        "elapsed_s": elapsed,
        "ms_per_iter": 1000.0 * elapsed / iters,
        "tokens_per_s": tokens / elapsed,
        "peak_memory_mb": peak_memory_mb,
        "output_abs_mean": float(out.detach().abs().mean().cpu()),
        "loss": (
            float(loss.detach().cpu())
            if include_backward and loss is not None
            else None
        ),
        "grad_abs_mean": grad_abs_mean,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-models", type=int, nargs="+", default=None)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compile-scan", action="store_true")
    parser.add_argument("--include-backward", action="store_true")
    parser.add_argument(
        "--scan-backend",
        choices=["reference", "compiled", "triton", "triton_recompute"],
        default=None,
    )
    parser.add_argument("--output-json", default="results/rwkv7_ref_bench.json")
    parser.add_argument("--output-csv", default="results/rwkv7_ref_bench.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rows = []
    d_models = args.d_models if args.d_models is not None else [args.d_model]
    for d_model in d_models:
        for seq_len in args.seq_lens:
            for use_rank in [True, False]:
                rows.append(
                    benchmark_case(
                        batch_size=args.batch_size,
                        seq_len=seq_len,
                        d_model=d_model,
                        device=device,
                        use_rank=use_rank,
                        warmup=args.warmup,
                        iters=args.iters,
                        compile_scan=args.compile_scan,
                        scan_backend=args.scan_backend,
                        include_backward=args.include_backward,
                    )
                )
    result = {
        "benchmark": "rwkv7_scan",
        "git": git_metadata(),
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(Path(args.output_csv), rows)


if __name__ == "__main__":
    main()
