"""Minimal toy training entrypoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import torch

from .diffusion import apply_subs_sample, masked_nll_loss
from .model import build_tiny_denoiser
from .text_data import (
    GPT2TextBatcher,
    ByteTextBatcher,
    PackedTextBatcher,
    WordTextBatcher,
    make_text_batcher,
)
from .toy_data import make_bidirectional_copy_batch, make_mqar_batch


def _load_config_defaults(parser: argparse.ArgumentParser) -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.config is None:
        return
    config = json.loads(Path(pre_args.config).read_text())
    parser.set_defaults(config=pre_args.config, **config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--task", choices=["copy", "mqar", "text"], default="copy")
    parser.add_argument("--text-file", default=None)
    parser.add_argument(
        "--text-tokenizer", choices=["byte", "word", "gpt2", "packed"], default="byte"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument(
        "--model-type",
        choices=["dualgoose", "attention", "mamba", "diffumamba_h", "mamba2"],
        default="dualgoose",
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--attention-window", type=int, default=64)
    parser.add_argument("--mamba2-d-state", type=int, default=64)
    parser.add_argument("--mamba2-chunk-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--fail-on-nonfinite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--log-batch-fingerprint", action="store_true")
    parser.add_argument("--eval-text-file", default=None)
    parser.add_argument(
        "--eval-text-tokenizer",
        choices=["byte", "word", "gpt2", "packed"],
        default=None,
    )
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--eval-data-seed", type=int, default=None)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/toy_train.json")
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--checkpoint-output", default="checkpoints/toy_model.pt")
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--checkpoint-latest-output", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--copy-direction", choices=["left", "right", "mixed"], default="mixed")
    parser.add_argument("--mqar-pairs", type=int, default=4)
    parser.add_argument("--mqar-layout", choices=["middle", "after_table"], default="middle")
    parser.add_argument("--direction", choices=["forward", "backward", "bidirectional"], default="bidirectional")
    parser.add_argument(
        "--merge", choices=["film", "film_warm", "film_raw", "static"], default="film_warm"
    )
    parser.add_argument(
        "--scan-backend",
        choices=["reference", "compiled", "triton", "triton_recompute"],
        default="reference",
    )
    parser.add_argument("--diagonal-only", action="store_true")
    parser.add_argument("--overfit-batch", action="store_true")
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Compute precision for the forward/backward pass. bf16 uses autocast "
        "with fp32 master weights; fp16 adds a GradScaler.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="Token budget. If set, overrides --steps with "
        "ceil(tokens / (batch_size * seq_len)).",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        default="constant",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=0.0,
        help="Final learning rate for the cosine schedule.",
    )
    parser.add_argument(
        "--lr-warmup",
        type=int,
        default=0,
        help="Linear warmup steps from 0 to --lr before the schedule applies.",
    )
    _load_config_defaults(parser)
    return parser.parse_args()


def lr_at_step(
    step: int,
    *,
    peak_lr: float,
    total_steps: int,
    schedule: str,
    min_lr: float,
    warmup_steps: int,
) -> float:
    """Learning rate for a 0-indexed optimizer step.

    Linear warmup from 0 to ``peak_lr`` over ``warmup_steps``; afterwards either a
    constant ``peak_lr`` or a cosine decay from ``peak_lr`` to ``min_lr`` across the
    remaining steps.
    """

    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if schedule == "constant":
        return peak_lr
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def git_metadata() -> dict[str, object]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except OSError:
            return ""

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "--short", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status,
    }


def _make_batch(
    args: argparse.Namespace,
    mask_token_id: int,
    device: torch.device,
    text_batcher: ByteTextBatcher | WordTextBatcher | GPT2TextBatcher | PackedTextBatcher | None,
):
    if args.task == "copy":
        return make_bidirectional_copy_batch(
            args.batch_size,
            args.seq_len,
            args.vocab_size,
            mask_token_id,
            direction=args.copy_direction,
            device=device,
        )
    if args.task == "mqar":
        return make_mqar_batch(
            args.batch_size,
            args.seq_len,
            args.vocab_size,
            mask_token_id,
            pairs=args.mqar_pairs,
            query_layout=args.mqar_layout,
            device=device,
        )
    if text_batcher is None:
        raise ValueError("text task requires --text-file")
    return text_batcher.sample(args.batch_size, args.seq_len, device=device)


def write_csv(path: str, history: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        output.write_text("")
        return
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _tensor_hash(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
    digest.update(str(cpu_tensor.dtype).encode("ascii"))
    digest.update(cpu_tensor.numpy().tobytes())
    return digest.hexdigest()


def batch_fingerprint(batch) -> dict[str, float | str]:
    return {
        "clean_hash": _tensor_hash(batch.clean),
        "corrupted_hash": _tensor_hash(batch.corrupted),
        "target_mask_hash": _tensor_hash(batch.target_mask),
        "t_hash": _tensor_hash(batch.t),
        "mask_fraction": float(batch.target_mask.float().mean().detach().cpu()),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latest_checkpoint_path(
    checkpoint_output: str,
    checkpoint_latest_output: str | None,
) -> Path:
    if checkpoint_latest_output is not None:
        return Path(checkpoint_latest_output)
    final = Path(checkpoint_output)
    suffix = final.suffix or ".pt"
    return final.with_name(f"{final.stem}_latest{suffix}")


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_args: dict[str, object],
    train_args: argparse.Namespace,
    metadata: dict[str, object],
    step: int,
    history: list[dict[str, object]],
    eval_history: list[dict[str, object]],
    device: torch.device,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_args": model_args,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_args": vars(train_args),
        "git": metadata,
        "step": step,
        "history": history,
        "eval_history": eval_history,
        "rng_state": torch.get_rng_state(),
    }
    if device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    return payload


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_args: dict[str, object],
    train_args: argparse.Namespace,
    metadata: dict[str, object],
    step: int,
    history: list[dict[str, object]],
    eval_history: list[dict[str, object]],
    device: torch.device,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            model_args=model_args,
            train_args=train_args,
            metadata=metadata,
            step=step,
            history=history,
            eval_history=eval_history,
            device=device,
        ),
        temporary,
    )
    temporary.replace(output)


def _rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _move_tensor_tree_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {
            key: _move_tensor_tree_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_tensor_tree_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree_to_device(item, device) for item in value)
    return value


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = _move_tensor_tree_to_device(value, device)


def _evaluate_text_model(
    model: torch.nn.Module,
    text_batcher: ByteTextBatcher | WordTextBatcher | GPT2TextBatcher | PackedTextBatcher,
    *,
    batch_size: int,
    seq_len: int,
    mask_token_id: int,
    device: torch.device,
    eval_batches: int,
    seed: int,
    autocast_dtype: torch.dtype | None = None,
    use_autocast: bool = False,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_masked = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad(), torch.random.fork_rng(devices=_rng_devices(device)):
        torch.manual_seed(seed)
        for _ in range(eval_batches):
            batch = text_batcher.sample(batch_size, seq_len, device=device)
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=use_autocast
            ):
                logits = model(batch.corrupted, batch.t)
            loss = masked_nll_loss(
                logits,
                batch.clean,
                batch.corrupted,
                mask_token_id,
                t=batch.t,
            )
            predicted = apply_subs_sample(logits, batch.corrupted, mask_token_id)
            masked = batch.target_mask
            masked_count = int(masked.sum().detach().cpu())
            if masked_count:
                total_loss += float(loss.detach().cpu()) * masked_count
                total_correct += int(
                    predicted[masked].eq(batch.clean[masked]).sum().detach().cpu()
                )
                total_masked += masked_count
            total_tokens += int(batch.clean.numel())
    if was_training:
        model.train()
    return {
        "validation_loss": total_loss / max(1, total_masked),
        "validation_masked_accuracy": total_correct / max(1, total_masked),
        "validation_masked_tokens": total_masked,
        "validation_tokens": total_tokens,
        "validation_mask_fraction": total_masked / max(1, total_tokens),
        "eval_batches": eval_batches,
    }


def main() -> None:
    args = parse_args()
    if args.eval_every_steps is not None and args.eval_every_steps <= 0:
        raise ValueError("--eval-every-steps must be positive")
    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be positive")
    if args.eval_every_steps is not None and args.eval_text_file is None:
        raise ValueError("--eval-every-steps requires --eval-text-file")
    if args.checkpoint_every_steps is not None and args.checkpoint_every_steps <= 0:
        raise ValueError("--checkpoint-every-steps must be positive")
    if args.resume_checkpoint is not None and args.overfit_batch:
        raise ValueError("--resume-checkpoint does not support --overfit-batch")
    if args.tokens is not None:
        if args.tokens <= 0:
            raise ValueError("--tokens must be positive")
        tokens_per_step = args.batch_size * args.seq_len
        args.steps = max(1, math.ceil(args.tokens / tokens_per_step))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    mask_token_id = args.vocab_size - 1
    text_batcher = (
        make_text_batcher(
            args.text_file,
            args.vocab_size,
            mask_token_id,
            tokenizer=args.text_tokenizer,
        )
        if args.task == "text"
        else None
    )
    eval_batcher = (
        make_text_batcher(
            args.eval_text_file,
            args.vocab_size,
            mask_token_id,
            tokenizer=args.eval_text_tokenizer or args.text_tokenizer,
        )
        if args.eval_text_file is not None
        else None
    )
    eval_seed = args.eval_data_seed
    if eval_seed is None:
        eval_seed = (args.data_seed if args.data_seed is not None else args.seed) + 1
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
    model = build_tiny_denoiser(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision)
    use_autocast = autocast_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device.type, enabled=(args.precision == "fp16" and device.type == "cuda")
    )
    history = []
    eval_history = []
    start_step = 0
    fixed_batch = None
    if args.resume_checkpoint is not None:
        resume = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(resume["model_state"])
        optimizer_state = resume.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            _move_optimizer_state_to_device(optimizer, device)
        start_step = int(resume.get("step", 0))
        history = list(resume.get("history", []))
        eval_history = list(resume.get("eval_history", []))
        rng_state = resume.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state.cpu())
        cuda_rng_state = resume.get("cuda_rng_state")
        if cuda_rng_state is not None and device.type == "cuda":
            torch.cuda.set_rng_state(cuda_rng_state.to(device="cpu"), device)
        del resume, optimizer_state, rng_state, cuda_rng_state
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elif args.data_seed is not None:
        torch.manual_seed(args.data_seed)
    if args.overfit_batch:
        fixed_batch = _make_batch(args, mask_token_id, device, text_batcher)

    synchronize(device)
    start_time = time.perf_counter()
    elapsed_offset = float(history[-1].get("elapsed_s", 0.0)) if history else 0.0
    metadata = git_metadata()
    latest_checkpoint = _latest_checkpoint_path(
        args.checkpoint_output,
        args.checkpoint_latest_output,
    )
    for step in range(start_step, args.steps):
        current_lr = lr_at_step(
            step,
            peak_lr=args.lr,
            total_steps=args.steps,
            schedule=args.lr_schedule,
            min_lr=args.lr_min,
            warmup_steps=args.lr_warmup,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        if fixed_batch is not None:
            batch = fixed_batch
        else:
            batch = _make_batch(args, mask_token_id, device, text_batcher)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=use_autocast
        ):
            logits = model(batch.corrupted, batch.t)
            loss = masked_nll_loss(
                logits, batch.clean, batch.corrupted, mask_token_id, t=batch.t
            )
        if args.fail_on_nonfinite and not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        grad_norm = None
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip_norm,
                error_if_nonfinite=args.fail_on_nonfinite,
            )
            grad_norm = float(grad_norm_tensor.detach().cpu())
        scaler.step(optimizer)
        scaler.update()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
            synchronize(device)
            elapsed = elapsed_offset + time.perf_counter() - start_time
            tokens_seen = args.batch_size * args.seq_len * (step + 1)
            row = {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "lr": current_lr,
                "elapsed_s": elapsed,
                "tokens_per_s": tokens_seen / elapsed,
            }
            if grad_norm is not None:
                row["grad_norm"] = grad_norm
            if args.log_batch_fingerprint:
                row.update(batch_fingerprint(batch))
            history.append(row)
        if (
            eval_batcher is not None
            and args.eval_every_steps is not None
            and ((step + 1) % args.eval_every_steps == 0 or step + 1 == args.steps)
        ):
            eval_metrics = _evaluate_text_model(
                model,
                eval_batcher,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                mask_token_id=mask_token_id,
                device=device,
                eval_batches=args.eval_batches,
                seed=eval_seed + step + 1,
                autocast_dtype=autocast_dtype,
                use_autocast=use_autocast,
            )
            validation_loss = torch.tensor(float(eval_metrics["validation_loss"]))
            if args.fail_on_nonfinite and not torch.isfinite(validation_loss):
                raise FloatingPointError(
                    f"non-finite validation loss at step {step + 1}: "
                    f"{eval_metrics['validation_loss']}"
                )
            synchronize(device)
            eval_history.append(
                {
                    "step": step + 1,
                    "elapsed_s": elapsed_offset + time.perf_counter() - start_time,
                    **eval_metrics,
                }
            )
        if (
            args.checkpoint_every_steps is not None
            and ((step + 1) % args.checkpoint_every_steps == 0 or step + 1 == args.steps)
        ):
            synchronize(device)
            save_training_checkpoint(
                latest_checkpoint,
                model=model,
                optimizer=optimizer,
                model_args=model_args,
                train_args=args,
                metadata=metadata,
                step=step + 1,
                history=history,
                eval_history=eval_history,
                device=device,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "args": vars(args),
        "model_args": model_args,
        "git": metadata,
        "history": history,
        "eval_history": eval_history,
        "elapsed_s": history[-1]["elapsed_s"] if history else 0.0,
        "tokens_per_s": history[-1]["tokens_per_s"] if history else 0.0,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.eval_output is not None:
        eval_output = Path(args.eval_output)
        eval_output.parent.mkdir(parents=True, exist_ok=True)
        eval_output.write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "model_args": model_args,
                    "git": metadata,
                    "eval_history": eval_history,
                },
                indent=2,
            )
            + "\n"
        )
    if args.csv_output is not None:
        write_csv(args.csv_output, history)
    checkpoint = Path(args.checkpoint_output)
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        model_args=model_args,
        train_args=args,
        metadata=metadata,
        step=args.steps,
        history=history,
        eval_history=eval_history,
        device=device,
    )


if __name__ == "__main__":
    main()
