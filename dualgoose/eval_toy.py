"""Minimal toy evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .diffusion import apply_subs_sample, masked_nll_loss
from .model import build_tiny_denoiser
from .text_data import make_text_batcher
from .toy_data import make_bidirectional_copy_batch, make_mqar_batch
from .train_toy import batch_fingerprint, git_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["copy", "mqar", "text"], default="copy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-file", default=None)
    parser.add_argument(
        "--text-tokenizer", choices=["byte", "word", "gpt2", "packed"], default="byte"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--copy-direction", choices=["left", "right", "mixed"], default="mixed")
    parser.add_argument("--mqar-pairs", type=int, default=4)
    parser.add_argument("--mqar-layout", choices=["middle", "after_table"], default="middle")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--model-type",
        choices=["dualgoose", "attention", "mamba", "diffumamba_h", "mamba2"],
        default="dualgoose",
    )
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--attention-window", type=int, default=64)
    parser.add_argument("--scan-backend", choices=["reference", "compiled", "triton", "triton_recompute"], default="reference")
    parser.add_argument("--mamba2-d-state", type=int, default=64)
    parser.add_argument("--mamba2-chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/toy_eval.json")
    parser.add_argument("--direction", choices=["forward", "backward", "bidirectional"], default="bidirectional")
    parser.add_argument("--merge", choices=["film", "static"], default="film")
    parser.add_argument("--diagonal-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--log-batch-fingerprint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    mask_token_id = args.vocab_size - 1
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    seed = args.seed
    if seed is None and "train_args" in checkpoint:
        seed = checkpoint["train_args"].get("seed", 0)
    if "model_args" in checkpoint:
        model_args = checkpoint["model_args"]
        args.vocab_size = model_args["vocab_size"]
        args.seq_len = model_args["max_length"]
        mask_token_id = args.vocab_size - 1
    else:
        model_args = {
            "model_type": args.model_type,
            "vocab_size": args.vocab_size,
            "max_length": args.seq_len,
            "d_model": args.d_model,
            "n_layers": args.layers,
            "direction": args.direction,
            "merge": args.merge,
            "use_rank": not args.diagonal_only,
            "scan_backend": args.scan_backend,
            "n_heads": args.heads,
            "attention_window": args.attention_window,
            "mamba2_d_state": args.mamba2_d_state,
            "mamba2_chunk_size": args.mamba2_chunk_size,
        }
        checkpoint = {"model_state": checkpoint}
    model = build_tiny_denoiser(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    data_seed = args.data_seed if args.data_seed is not None else seed
    torch.manual_seed(0 if data_seed is None else data_seed)
    with torch.no_grad():
        if args.task == "text":
            if args.text_file is None:
                raise ValueError("text evaluation requires --text-file")
            text_batcher = make_text_batcher(
                args.text_file,
                args.vocab_size,
                mask_token_id,
                tokenizer=args.text_tokenizer,
            )
            total_loss = 0.0
            total_masked = 0
            total_correct = 0
            fingerprints: list[dict[str, float | str]] = []
            for _ in range(args.eval_batches):
                batch = text_batcher.sample(args.batch_size, args.seq_len, device=device)
                logits = model(batch.corrupted, batch.t)
                loss = masked_nll_loss(
                    logits,
                    batch.clean,
                    batch.corrupted,
                    mask_token_id,
                    t=batch.t,
                )
                masked_count = int(batch.target_mask.sum().detach().cpu())
                total_loss += float(loss.detach().cpu()) * max(1, masked_count)
                total_masked += masked_count
                pred = apply_subs_sample(logits, batch.corrupted, mask_token_id)
                total_correct += int(
                    pred[batch.target_mask].eq(batch.clean[batch.target_mask]).sum().detach().cpu()
                )
                if args.log_batch_fingerprint:
                    fingerprints.append(batch_fingerprint(batch))
            output_payload = {
                "args": vars(args),
                "checkpoint_git": checkpoint.get("git", {}),
                "git": git_metadata(),
                "loss": total_loss / max(1, total_masked),
                "masked_accuracy": total_correct / max(1, total_masked),
                "masked_tokens": total_masked,
                "tokens": args.batch_size * args.seq_len * args.eval_batches,
                "mask_fraction": total_masked / max(1, args.batch_size * args.seq_len * args.eval_batches),
                "eval_batches": args.eval_batches,
            }
            if args.log_batch_fingerprint:
                output_payload["batch_fingerprints"] = fingerprints
        elif args.task == "copy":
            batch = make_bidirectional_copy_batch(
                args.batch_size,
                args.seq_len,
                args.vocab_size,
                mask_token_id,
                direction=args.copy_direction,
                device=device,
            )
        else:
            batch = make_mqar_batch(
                args.batch_size,
                args.seq_len,
                args.vocab_size,
                mask_token_id,
                pairs=args.mqar_pairs,
                query_layout=args.mqar_layout,
                device=device,
            )
        if args.task != "text":
            pred = apply_subs_sample(model(batch.corrupted, batch.t), batch.corrupted, mask_token_id)
            correct = pred[batch.target_mask].eq(batch.clean[batch.target_mask]).float().mean()
            output_payload = {"accuracy": float(correct.cpu())}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
