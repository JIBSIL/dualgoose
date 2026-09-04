"""Fixed-mask-ratio bucketed evaluation for masked diffusion denoisers.

The training loop samples the diffusion timestep ``t`` (and therefore the mask
ratio) uniformly per batch, which only yields an aggregate validation number.
Claim C10 ("timestep-conditioned merge should beat static merge when the mask
ratio changes") needs metrics bucketed by mask ratio. This module loads a
trained checkpoint and reports masked NLL and masked accuracy at a set of fixed
mask ratios, on a held-out corpus, using the *same* text windows and masking
patterns across every checkpoint so that runs are directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .diffusion import apply_subs_sample, corrupt_absorbing, masked_nll_loss
from .model import build_tiny_denoiser
from .text_data import make_text_batcher


def _sample_clean_windows(
    text_batcher,
    *,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
    seed: int,
) -> list[torch.Tensor]:
    """Sample a fixed, deterministic set of clean token windows."""

    with torch.random.fork_rng(devices=[] if device.type != "cuda" else [device.index or 0]):
        torch.manual_seed(seed)
        return [
            text_batcher.sample(batch_size, seq_len, device=device).clean
            for _ in range(eval_batches)
        ]


def evaluate_buckets(
    model: torch.nn.Module,
    text_batcher,
    *,
    mask_ratios: list[float],
    batch_size: int,
    seq_len: int,
    mask_token_id: int,
    device: torch.device,
    eval_batches: int,
    seed: int,
    autocast_dtype: torch.dtype | None = None,
    use_autocast: bool = False,
) -> list[dict[str, float | int]]:
    """Return per-bucket masked NLL and masked accuracy at fixed mask ratios."""

    was_training = model.training
    model.eval()
    clean_windows = _sample_clean_windows(
        text_batcher,
        batch_size=batch_size,
        seq_len=seq_len,
        eval_batches=eval_batches,
        device=device,
        seed=seed,
    )
    results: list[dict[str, float | int]] = []
    with torch.no_grad():
        for ratio in mask_ratios:
            total_loss = 0.0
            total_masked = 0
            total_correct = 0
            total_tokens = 0
            # Deterministic, ratio-specific generator so the masking pattern is
            # identical for every checkpoint evaluated at this ratio.
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + int(round(ratio * 1000)))
            for clean in clean_windows:
                t = torch.full((clean.shape[0],), float(ratio), device=device)
                corrupted, target_mask = corrupt_absorbing(
                    clean, t, mask_token_id, generator=generator
                )
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype, enabled=use_autocast
                ):
                    logits = model(corrupted, t)
                # Raw (unweighted) cross-entropy per masked token for fair
                # cross-bucket comparison; the 1/t training weight is omitted.
                loss = masked_nll_loss(
                    logits.float(), clean, corrupted, mask_token_id, t=None
                )
                predicted = apply_subs_sample(logits, corrupted, mask_token_id)
                masked_count = int(target_mask.sum().detach().cpu())
                if masked_count:
                    total_loss += float(loss.detach().cpu()) * masked_count
                    total_correct += int(
                        predicted[target_mask].eq(clean[target_mask]).sum().detach().cpu()
                    )
                    total_masked += masked_count
                total_tokens += int(clean.numel())
            results.append(
                {
                    "mask_ratio": ratio,
                    "masked_nll": total_loss / max(1, total_masked),
                    "masked_accuracy": total_correct / max(1, total_masked),
                    "masked_tokens": total_masked,
                    "observed_mask_fraction": total_masked / max(1, total_tokens),
                    "eval_batches": eval_batches,
                }
            )
    if was_training:
        model.train()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument(
        "--text-tokenizer", choices=["byte", "word", "gpt2", "packed"], default="byte"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument(
        "--mask-ratios",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 0.8],
        help="Fixed mask ratios (diffusion t) to bucket by.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--precision", choices=["fp32", "bf16", "fp16"], default="fp32"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/bucketed_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_args" not in checkpoint:
        raise ValueError("checkpoint missing model_args; cannot rebuild model")
    model_args = checkpoint["model_args"]
    model = build_tiny_denoiser(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state"])
    vocab_size = model_args["vocab_size"]
    seq_len = model_args["max_length"]
    mask_token_id = vocab_size - 1
    text_batcher = make_text_batcher(
        args.text_file, vocab_size, mask_token_id, tokenizer=args.text_tokenizer
    )
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision)
    use_autocast = autocast_dtype is not None and device.type == "cuda"
    buckets = evaluate_buckets(
        model,
        text_batcher,
        mask_ratios=args.mask_ratios,
        batch_size=args.batch_size,
        seq_len=seq_len,
        mask_token_id=mask_token_id,
        device=device,
        eval_batches=args.eval_batches,
        seed=args.seed,
        autocast_dtype=autocast_dtype,
        use_autocast=use_autocast,
    )
    payload = {
        "checkpoint": args.checkpoint,
        "model_args": model_args,
        "train_args": checkpoint.get("train_args", {}),
        "checkpoint_git": checkpoint.get("git", {}),
        "step": checkpoint.get("step"),
        "mask_ratios": args.mask_ratios,
        "seed": args.seed,
        "buckets": buckets,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
