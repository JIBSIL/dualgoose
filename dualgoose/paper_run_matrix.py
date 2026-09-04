"""Generate paper-target training run configs without launching training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch

from .baseline_preflight import PAPER_TARGET_MATRIX
from .model import build_tiny_denoiser
from .train_toy import git_metadata

DEFAULT_TEXT_FILE = "results/owt_stream_full_packed/train"
DEFAULT_VOCAB_SIZE = 50_258
TRITON_SAVED_STATE_BUDGET_GIB = 28.0
RECOMPUTE_CHECKPOINT_INTERVAL = 64
RECOMPUTE_CHECKPOINTED_SMOKE_MAX_STEPS = 5
DEFAULT_MODEL_WIDTHS = [
    128,
    192,
    256,
    384,
    512,
    640,
    768,
    896,
    1024,
    1280,
    1536,
    1792,
    2048,
    2560,
    3072,
    3584,
    4096,
]
DEFAULT_LAYER_COUNTS = [2, 4, 6, 8, 12, 16]

RUNNABLE_BASELINES = [
    {
        "baseline_id": "dualgoose",
        "paper_role": "core recurrent denoiser",
        "model_type": "dualgoose",
        "direction": "bidirectional",
        "merge": "static",
        "scan_backend": "triton",
    },
    {
        "baseline_id": "transformer_dit",
        "paper_role": "bidirectional attention DiT-style control",
        "model_type": "attention",
    },
    {
        "baseline_id": "diffumamba_local",
        "paper_role": "V100 replacement for official DiffuMamba/Mamba2 training",
        "model_type": "mamba",
    },
    {
        "baseline_id": "diffumamba_h",
        "paper_role": "Mamba-style plus sparse-attention hybrid upper control",
        "model_type": "diffumamba_h",
    },
    {
        "baseline_id": "causal_rwkv7",
        "paper_role": "causal-only recurrent ablation",
        "model_type": "dualgoose",
        "direction": "forward",
        "merge": "static",
        "scan_backend": "triton",
    },
]

REPLACED_BASELINES = [
    {
        "baseline_id": "diffumamba_official_mamba2",
        "paper_role": "official grad-enabled Mamba2 diffusion baseline",
        "status": "replaced_on_v100",
        "replacement_baseline_id": "diffumamba_local",
        "replacement_note": (
            "Official Mamba2 remains compile/autotune-blocked for grad-enabled "
            "training on this V100, so bounded V100 training comparisons use the "
            "local trainable DiffuMamba-style SSM baseline."
        ),
    },
]

BLOCKED_BASELINES: list[dict[str, str]] = []


def attention_heads(d_model: int) -> int:
    return max(1, d_model // 64)


def model_args_for_candidate(
    baseline: dict[str, str],
    *,
    vocab_size: int,
    context: int,
    d_model: int,
    layers: int,
) -> dict[str, object]:
    model_type = baseline["model_type"]
    args: dict[str, object] = {
        "model_type": model_type,
        "vocab_size": vocab_size,
        "max_length": context,
        "d_model": d_model,
        "n_layers": layers,
        "direction": baseline.get("direction", "bidirectional"),
        "merge": baseline.get("merge", "static"),
        "use_rank": True,
        "scan_backend": baseline.get("scan_backend", "reference"),
        "n_heads": attention_heads(d_model),
        "attention_window": 64,
        "mamba2_d_state": 64,
        "mamba2_chunk_size": 64,
    }
    return args


def parameter_count(model_args: dict[str, object]) -> int:
    with torch.device("meta"):
        model = build_tiny_denoiser(**model_args)
    return sum(parameter.numel() for parameter in model.parameters())


def triton_saved_state_gib(
    model_args: dict[str, object],
    *,
    batch_size: int,
) -> float:
    if model_args["model_type"] != "dualgoose":
        return 0.0
    if model_args["scan_backend"] != "triton":
        return 0.0
    directions = 2 if model_args["direction"] == "bidirectional" else 1
    elements = (
        batch_size
        * int(model_args["max_length"])
        * int(model_args["d_model"])
        * int(model_args["d_model"])
        * int(model_args["n_layers"])
        * directions
    )
    return elements * 4 / 1024**3


def v100_launch_model_args(
    model_args: dict[str, object],
    *,
    batch_size: int,
) -> dict[str, object]:
    launch_args = dict(model_args)
    if triton_saved_state_gib(model_args, batch_size=batch_size) > TRITON_SAVED_STATE_BUDGET_GIB:
        launch_args["scan_backend"] = "triton_recompute"
    return launch_args


def triton_recompute_backward_work_units(
    model_args: dict[str, object],
    *,
    batch_size: int,
    checkpoint_interval: int = RECOMPUTE_CHECKPOINT_INTERVAL,
) -> int:
    """Relative work estimate for checkpointed recompute backward kernels.

    The Triton recompute path now stores sparse recurrent-state checkpoints and
    replays at most one checkpoint chunk inside each backward step. This tracks
    the bounded state-replay term rather than exact FLOPs.
    """

    if model_args["model_type"] != "dualgoose":
        return 0
    if model_args["scan_backend"] != "triton_recompute":
        return 0
    directions = 2 if model_args["direction"] == "bidirectional" else 1
    replay_span = min(int(model_args["max_length"]), checkpoint_interval)
    return (
        batch_size
        * directions
        * int(model_args["n_layers"])
        * int(model_args["max_length"])
        * replay_span
        * int(model_args["d_model"])
        * int(model_args["d_model"])
    )


def v100_recompute_runtime_gate(
    model_args: dict[str, object],
    *,
    batch_size: int,
) -> dict[str, object]:
    work_units = triton_recompute_backward_work_units(
        model_args,
        batch_size=batch_size,
    )
    if work_units == 0:
        return {
            "estimated_triton_recompute_backward_work_units": 0,
            "v100_recompute_runtime_gate": "not_applicable",
            "triton_recompute_checkpoint_interval": None,
            "v100_recompute_max_steps": None,
        }
    return {
        "estimated_triton_recompute_backward_work_units": work_units,
        "v100_recompute_runtime_gate": "checkpointed_smoke_only",
        "triton_recompute_checkpoint_interval": RECOMPUTE_CHECKPOINT_INTERVAL,
        "v100_recompute_max_steps": RECOMPUTE_CHECKPOINTED_SMOKE_MAX_STEPS,
    }


def v100_status_for_runtime_gate(runtime_gate: dict[str, object]) -> str:
    if runtime_gate["v100_recompute_runtime_gate"] == "checkpointed_smoke_only":
        return "smoke_only_config"
    return "ready_config"


def v100_effective_smoke_steps(
    *,
    requested_steps: int,
    runtime_gate: dict[str, object],
) -> int:
    max_steps = runtime_gate["v100_recompute_max_steps"]
    if max_steps is None:
        return requested_steps
    return min(requested_steps, int(max_steps))


def choose_model_shape(
    baseline: dict[str, str],
    *,
    target_parameters: int,
    context: int,
    vocab_size: int,
    widths: Iterable[int] = DEFAULT_MODEL_WIDTHS,
    layers: Iterable[int] = DEFAULT_LAYER_COUNTS,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for layer_count in layers:
        for width in widths:
            if baseline["model_type"] in {
                "attention",
                "diffumamba_h",
            } and width % attention_heads(width):
                continue
            model_args = model_args_for_candidate(
                baseline,
                vocab_size=vocab_size,
                context=context,
                d_model=width,
                layers=layer_count,
            )
            params = parameter_count(model_args)
            relative_error = abs(params - target_parameters) / target_parameters
            candidate = {
                "d_model": width,
                "layers": layer_count,
                "parameters": params,
                "relative_error": relative_error,
                "model_args": model_args,
            }
            if best is None or relative_error < float(best["relative_error"]):
                best = candidate
    if best is None:
        raise RuntimeError(f"no model shape candidates for {baseline['baseline_id']}")
    return best


def candidate_shapes(
    baseline: dict[str, str],
    *,
    context: int,
    vocab_size: int,
    widths: Iterable[int] = DEFAULT_MODEL_WIDTHS,
    layers: Iterable[int] = DEFAULT_LAYER_COUNTS,
) -> list[dict[str, object]]:
    candidates = []
    for layer_count in layers:
        for width in widths:
            if baseline["model_type"] in {
                "attention",
                "diffumamba_h",
            } and width % attention_heads(width):
                continue
            model_args = model_args_for_candidate(
                baseline,
                vocab_size=vocab_size,
                context=context,
                d_model=width,
                layers=layer_count,
            )
            candidates.append(
                {
                    "d_model": width,
                    "layers": layer_count,
                    "parameters": parameter_count(model_args),
                    "model_args": model_args,
                }
            )
    if not candidates:
        raise RuntimeError(f"no model shape candidates for {baseline['baseline_id']}")
    return candidates


def closest_shape(
    candidates: list[dict[str, object]],
    *,
    target_parameters: int,
) -> dict[str, object]:
    best = min(
        candidates,
        key=lambda candidate: abs(int(candidate["parameters"]) - target_parameters),
    )
    return {
        **best,
        "relative_error": abs(int(best["parameters"]) - target_parameters)
        / target_parameters,
    }


def train_config_for_shape(
    baseline: dict[str, str],
    shape: dict[str, object],
    *,
    context: int,
    text_file: str,
    smoke_steps: int,
    batch_size: int,
) -> dict[str, object]:
    model_args = v100_launch_model_args(
        dict(shape["model_args"]),
        batch_size=batch_size,
    )
    runtime_gate = v100_recompute_runtime_gate(
        model_args,
        batch_size=batch_size,
    )
    steps = v100_effective_smoke_steps(
        requested_steps=smoke_steps,
        runtime_gate=runtime_gate,
    )
    baseline_id = baseline["baseline_id"]
    target = int(shape["parameters"])
    return {
        "task": "text",
        "text_file": text_file,
        "text_tokenizer": "packed",
        "model_type": model_args["model_type"],
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": context,
        "vocab_size": model_args["vocab_size"],
        "d_model": model_args["d_model"],
        "layers": model_args["n_layers"],
        "heads": model_args["n_heads"],
        "direction": model_args["direction"],
        "merge": model_args["merge"],
        "scan_backend": model_args["scan_backend"],
        "attention_window": model_args["attention_window"],
        "mamba2_d_state": model_args["mamba2_d_state"],
        "mamba2_chunk_size": model_args["mamba2_chunk_size"],
        "lr": 0.0001,
        "grad_clip_norm": 1.0,
        "fail_on_nonfinite": True,
        "device": "cuda",
        "output": f"results/paper_{baseline_id}_{target}_{context}.json",
        "csv_output": f"results/paper_{baseline_id}_{target}_{context}.csv",
        "checkpoint_output": f"checkpoints/paper_{baseline_id}_{target}_{context}.pt",
    }


def build_run_matrix(
    *,
    text_file: str = DEFAULT_TEXT_FILE,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    smoke_steps: int = 100,
    batch_size: int = 1,
    widths: Iterable[int] = DEFAULT_MODEL_WIDTHS,
    layers: Iterable[int] = DEFAULT_LAYER_COUNTS,
    target_matrix: Iterable[dict[str, object]] = PAPER_TARGET_MATRIX,
) -> dict[str, object]:
    target_items = list(target_matrix)
    width_values = tuple(widths)
    layer_values = tuple(layers)
    contexts: list[int] = []
    seen_contexts: set[int] = set()
    for target_item in target_items:
        for context in target_item["contexts"]:
            context_int = int(context)
            if context_int not in seen_contexts:
                contexts.append(context_int)
                seen_contexts.add(context_int)

    candidates_by_context = {
        context: {
            baseline["baseline_id"]: candidate_shapes(
                baseline,
                context=context,
                vocab_size=vocab_size,
                widths=width_values,
                layers=layer_values,
            )
            for baseline in RUNNABLE_BASELINES
        }
        for context in contexts
    }

    runs = []
    for target_item in target_items:
        target = int(target_item["parameter_target"])
        for context in target_item["contexts"]:
            context_int = int(context)
            for baseline in RUNNABLE_BASELINES:
                shape = closest_shape(
                    candidates_by_context[context_int][baseline["baseline_id"]],
                    target_parameters=target,
                )
                saved_state_gib = triton_saved_state_gib(
                    dict(shape["model_args"]),
                    batch_size=batch_size,
                )
                launch_model_args = v100_launch_model_args(
                    dict(shape["model_args"]),
                    batch_size=batch_size,
                )
                runtime_gate = v100_recompute_runtime_gate(
                    launch_model_args,
                    batch_size=batch_size,
                )
                train_config = train_config_for_shape(
                    baseline,
                    shape,
                    context=context_int,
                    text_file=text_file,
                    smoke_steps=smoke_steps,
                    batch_size=batch_size,
                )
                runs.append(
                    {
                        "baseline_id": baseline["baseline_id"],
                        "paper_role": baseline["paper_role"],
                        "status": v100_status_for_runtime_gate(runtime_gate),
                        "target_parameters": target,
                        "context": context_int,
                        "parameters": shape["parameters"],
                        "relative_error": shape["relative_error"],
                        "estimated_triton_saved_state_gib": saved_state_gib,
                        **runtime_gate,
                        "v100_backend_adjusted": (
                            train_config["scan_backend"]
                            != dict(shape["model_args"])["scan_backend"]
                        ),
                        "train_config": train_config,
                    }
                )
            for blocked in BLOCKED_BASELINES:
                runs.append(
                    {
                        **blocked,
                        "target_parameters": target,
                        "context": context_int,
                    }
                )
    return {
        "target_hardware": "Tesla V100-SXM2-32GB",
        "text_file": text_file,
        "vocab_size": vocab_size,
        "smoke_steps": smoke_steps,
        "batch_size": batch_size,
        "git": git_metadata(),
        "replaced_baselines": REPLACED_BASELINES,
        "runs": runs,
    }


def write_matrix(path: str | Path, matrix: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(matrix, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", default=DEFAULT_TEXT_FILE)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--smoke-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default="configs/paper_run_matrix.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = build_run_matrix(
        text_file=args.text_file,
        vocab_size=args.vocab_size,
        smoke_steps=args.smoke_steps,
        batch_size=args.batch_size,
    )
    write_matrix(args.output, matrix)
    ready = sum(1 for run in matrix["runs"] if run["status"] == "ready_config")
    smoke_only = sum(
        1 for run in matrix["runs"] if run["status"] == "smoke_only_config"
    )
    blocked = sum(1 for run in matrix["runs"] if run["status"] == "blocked")
    print(
        json.dumps(
            {
                "output": args.output,
                "ready": ready,
                "smoke_only": smoke_only,
                "blocked": blocked,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
