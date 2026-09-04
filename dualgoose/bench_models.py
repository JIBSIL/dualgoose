"""Model-level DualGoose vs baseline throughput benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn

from .bench_scan import synchronize
from .model import (
    TinyAttentionDenoiser,
    TinyDiffuMambaHDenoiser,
    TinyDualGooseDenoiser,
    TinyMamba2Denoiser,
    TinyMambaDenoiser,
    build_tiny_denoiser,
)
from .train_toy import git_metadata


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    t = torch.rand(batch_size, device=device).clamp(0.05, 0.95)
    return tokens, t


def build_model(
    model_type: str,
    *,
    vocab_size: int,
    seq_len: int,
    d_model: int,
    layers: int,
    heads: int,
    attention_window: int = 64,
    scan_backend: str = "reference",
) -> nn.Module:
    return build_tiny_denoiser(
        vocab_size,
        seq_len,
        model_type=model_type,
        d_model=d_model,
        n_layers=layers,
        direction="bidirectional",
        merge="static",
        use_rank=True,
        scan_backend=scan_backend,
        n_heads=heads,
        attention_window=attention_window,
    )


def benchmark_model_case(
    *,
    model_type: str,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    layers: int,
    heads: int,
    device: torch.device,
    warmup: int,
    iters: int,
    attention_window: int = 64,
    scan_backend: str = "reference",
    include_backward: bool = False,
    optimizer_step: bool = False,
) -> dict[str, object]:
    if optimizer_step and not include_backward:
        raise ValueError("optimizer_step requires include_backward")
    model = build_model(
        model_type,
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        layers=layers,
        heads=heads,
        attention_window=attention_window,
        scan_backend=scan_backend,
    ).to(device)
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=1e-3)
        if include_backward and optimizer_step
        else None
    )
    tokens, t = make_batch(batch_size, seq_len, vocab_size, device)

    def run_once() -> tuple[torch.Tensor, torch.Tensor | None]:
        if include_backward:
            if optimizer is None:
                model.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, t)
            loss = logits.square().mean()
            loss.backward()
            if optimizer is not None:
                optimizer.step()
            return logits, loss
        with torch.no_grad():
            return model(tokens, t), None

    model.train(include_backward)
    for _ in range(warmup):
        run_once()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iters):
        logits, loss = run_once()
    synchronize(device)
    elapsed = time.perf_counter() - start
    tokens_seen = batch_size * seq_len * iters
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else None
    )
    return {
        "model_type": model_type,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "layers": layers,
        "heads": heads if model_type in {"attention", "diffumamba_h"} else "",
        "attention_window": attention_window if model_type == "diffumamba_h" else "",
        "scan_backend": scan_backend if model_type == "dualgoose" else "",
        "include_backward": include_backward,
        "optimizer_step": optimizer_step,
        "parameters": parameter_count(model),
        "device": str(device),
        "iters": iters,
        "elapsed_s": elapsed,
        "ms_per_iter": 1000.0 * elapsed / iters,
        "tokens_per_s": tokens_seen / elapsed,
        "peak_memory_mb": peak_memory_mb,
        "logits_abs_mean": float(logits.detach().abs().mean().cpu()),
        "loss": (
            float(loss.detach().cpu())
            if include_backward and loss is not None
            else None
        ),
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--attention-window", type=int, default=64)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["dualgoose", "attention", "mamba", "diffumamba_h", "mamba2"],
        default=["dualgoose", "attention"],
    )
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--include-backward", action="store_true")
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument(
        "--scan-backend",
        choices=["reference", "compiled", "triton", "triton_recompute"],
        default="reference",
    )
    parser.add_argument("--output-json", default="results/model_bench.json")
    parser.add_argument("--output-csv", default="results/model_bench.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    for seq_len in args.seq_lens:
        for model_type in args.models:
            rows.append(
                benchmark_model_case(
                    model_type=model_type,
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    vocab_size=args.vocab_size,
                    d_model=args.d_model,
                    layers=args.layers,
                    heads=args.heads,
                    attention_window=args.attention_window,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                    scan_backend=args.scan_backend,
                    include_backward=args.include_backward,
                    optimizer_step=args.optimizer_step,
                )
            )
    result = {
        "benchmark": "model_baseline_throughput",
        "git": git_metadata(),
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(Path(args.output_csv), rows)


if __name__ == "__main__":
    main()
