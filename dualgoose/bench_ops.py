"""Operator-level recurrent scan vs attention throughput benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .bench_scan import make_inputs, synchronize
from .rwkv7_ref import rwkv7_scan
from .train_toy import git_metadata


def make_attention_inputs(
    batch_size: int,
    seq_len: int,
    d_model: int,
    heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if d_model % heads != 0:
        raise ValueError("d_model must be divisible by heads")
    head_dim = d_model // heads
    shape = (batch_size, heads, seq_len, head_dim)
    return (
        torch.randn(shape, device=device),
        torch.randn(shape, device=device),
        torch.randn(shape, device=device),
    )


def benchmark_scan_case(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
    warmup: int,
    iters: int,
    compile_scan: bool,
) -> dict[str, object]:
    inputs = make_inputs(batch_size, seq_len, d_model, device)
    scan_fn = (
        torch.compile(rwkv7_scan, mode="reduce-overhead")
        if compile_scan
        else rwkv7_scan
    )
    with torch.no_grad():
        for _ in range(warmup):
            scan_fn(*inputs, use_rank=True)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            out = scan_fn(*inputs, use_rank=True)
        synchronize(device)
    elapsed = time.perf_counter() - start
    tokens = batch_size * seq_len * iters
    return {
        "op": "rwkv7_rank_scan",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "heads": "",
        "device": str(device),
        "compiled": compile_scan,
        "iters": iters,
        "elapsed_s": elapsed,
        "ms_per_iter": 1000.0 * elapsed / iters,
        "tokens_per_s": tokens / elapsed,
        "output_abs_mean": float(out.detach().abs().mean().cpu()),
    }


def benchmark_attention_case(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    heads: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    q, k, v = make_attention_inputs(batch_size, seq_len, d_model, heads, device)
    with torch.no_grad():
        for _ in range(warmup):
            F.scaled_dot_product_attention(q, k, v)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            out = F.scaled_dot_product_attention(q, k, v)
        synchronize(device)
    elapsed = time.perf_counter() - start
    tokens = batch_size * seq_len * iters
    return {
        "op": "scaled_dot_product_attention",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "heads": heads,
        "device": str(device),
        "compiled": False,
        "iters": iters,
        "elapsed_s": elapsed,
        "ms_per_iter": 1000.0 * elapsed / iters,
        "tokens_per_s": tokens / elapsed,
        "output_abs_mean": float(out.detach().abs().mean().cpu()),
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
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output-json", default="results/operator_bench.json")
    parser.add_argument("--output-csv", default="results/operator_bench.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    for seq_len in args.seq_lens:
        rows.append(
            benchmark_scan_case(
                batch_size=args.batch_size,
                seq_len=seq_len,
                d_model=args.d_model,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                compile_scan=True,
            )
        )
        rows.append(
            benchmark_attention_case(
                batch_size=args.batch_size,
                seq_len=seq_len,
                d_model=args.d_model,
                heads=args.heads,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
        )
    result = {
        "benchmark": "operator_scan_vs_attention",
        "git": git_metadata(),
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(Path(args.output_csv), rows)


if __name__ == "__main__":
    main()
