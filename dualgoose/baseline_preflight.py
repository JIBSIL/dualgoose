"""Preflight checks for paper-scale baseline and data execution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from .prepare_owt import load_policy, read_jsonl, resolve_token_dtype
from .train_toy import git_metadata

BYTES_PER_GIB = 1024**3
DEFAULT_OWT_TOKEN_COUNT = 8_600_000_000
DEFAULT_HEADROOM_GIB = 5.0
DEFAULT_TARGET_GPU_GIB = 31.0

PAPER_BASELINE_MODELS = [
    {
        "id": "dualgoose",
        "paper_role": "core recurrent denoiser",
        "current_status": "toy_implemented",
        "required_status": "paper_scale_configs_and_training_runs",
    },
    {
        "id": "transformer_dit",
        "paper_role": "full bidirectional attention denoiser",
        "current_status": "toy_attention_only",
        "required_status": "paper_scale_dit_implementation_and_training_runs",
    },
    {
        "id": "diffumamba",
        "paper_role": "bidirectional Mamba-2 diffusion baseline",
        "current_status": "v100_replaced_by_local_trainable_diffumamba_style_ssm",
        "required_status": "bounded_v100_replacement_training_runs",
    },
    {
        "id": "diffumamba_h",
        "paper_role": "Mamba-2 plus sparse-attention hybrid upper control",
        "current_status": "toy_hybrid_implemented",
        "required_status": "matched_hybrid_training_runs",
    },
    {
        "id": "causal_rwkv7",
        "paper_role": "causal-only recurrence ablation",
        "current_status": "toy_direction_ablation_available",
        "required_status": "matched_paper_scale_ablation_runs",
    },
]

PAPER_TARGET_MATRIX = [
    {"parameter_target": 100_000_000, "contexts": [1024, 2048, 4096]},
    {"parameter_target": 200_000_000, "contexts": [1024, 2048, 4096]},
    {"parameter_target": 400_000_000, "contexts": [1024, 2048, 4096]},
]


def bytes_to_gib(value: int | float) -> float:
    return float(value) / BYTES_PER_GIB


def disk_status(path: str | Path) -> dict[str, float | str]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gib": bytes_to_gib(usage.total),
        "used_gib": bytes_to_gib(usage.used),
        "free_gib": bytes_to_gib(usage.free),
    }


def cuda_status() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"available": False}
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gib": bytes_to_gib(props.total_memory),
        "torch_cuda": torch.version.cuda,
    }


def dependency_status() -> dict[str, object]:
    return {
        "torch": torch.__version__,
        "triton": importlib.util.find_spec("triton") is not None,
        "mamba_ssm": importlib.util.find_spec("mamba_ssm") is not None,
        "causal_conv1d": importlib.util.find_spec("causal_conv1d") is not None,
        "pyarrow": importlib.util.find_spec("pyarrow") is not None,
        "huggingface_hub": importlib.util.find_spec("huggingface_hub") is not None,
        "tiktoken": importlib.util.find_spec("tiktoken") is not None,
    }


def manifest_raw_size(manifest_path: str | Path) -> dict[str, object]:
    path = Path(manifest_path)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "shards": 0,
            "raw_bytes": 0,
            "max_shard_bytes": 0,
        }
    rows = read_jsonl(path)
    sizes = [int(row.get("size") or 0) for row in rows]
    return {
        "path": str(path),
        "exists": True,
        "shards": len(rows),
        "raw_bytes": sum(sizes),
        "raw_gib": bytes_to_gib(sum(sizes)),
        "max_shard_bytes": max(sizes) if sizes else 0,
        "max_shard_gib": bytes_to_gib(max(sizes) if sizes else 0),
    }


def estimate_owt_storage(
    policy: dict[str, object],
    manifest: dict[str, object],
    *,
    expected_tokens: int,
    headroom_gib: float,
) -> dict[str, object]:
    token_dtype = resolve_token_dtype(
        str(policy.get("packed_dtype", "uint16")),
        int(policy["mask_token_id"]),
    )
    packed_bytes = expected_tokens * token_dtype.itemsize
    raw_bytes = int(manifest.get("raw_bytes") or 0)
    max_shard_bytes = int(manifest.get("max_shard_bytes") or 0)
    headroom_bytes = int(headroom_gib * BYTES_PER_GIB)
    return {
        "expected_tokens": expected_tokens,
        "token_dtype": token_dtype.name,
        "packed_bytes": packed_bytes,
        "packed_gib": bytes_to_gib(packed_bytes),
        "raw_bytes": raw_bytes,
        "raw_gib": bytes_to_gib(raw_bytes),
        "max_shard_bytes": max_shard_bytes,
        "max_shard_gib": bytes_to_gib(max_shard_bytes),
        "headroom_gib": headroom_gib,
        "required_without_raw_cleanup_gib": bytes_to_gib(
            raw_bytes + packed_bytes + headroom_bytes
        ),
        "required_with_streaming_cleanup_gib": bytes_to_gib(
            max_shard_bytes + packed_bytes + headroom_bytes
        ),
    }


def implementation_status(deps: dict[str, object]) -> list[dict[str, object]]:
    statuses: list[dict[str, object]] = []
    for item in PAPER_BASELINE_MODELS:
        status = dict(item)
        if item["id"] == "diffumamba":
            status["dependency_present"] = bool(deps["mamba_ssm"])
            status["replacement_note"] = (
                "official Mamba2 remains diagnosed as grad-enabled V100 "
                "compile/autotune-blocked; bounded training comparisons use "
                "the local DiffuMamba-style SSM replacement"
            )
        elif item["id"] == "transformer_dit":
            status["blocking_note"] = (
                "only toy controls exist; no matched paper-scale training harness"
            )
        elif item["id"] == "diffumamba_h":
            status["blocking_note"] = (
                "toy sparse-attention hybrid exists; matched paper-scale "
                "training runs are still missing"
            )
        else:
            status["blocking_note"] = "toy path exists; paper-scale config/run missing"
        statuses.append(status)
    return statuses


def blocking_reasons(
    *,
    cuda: dict[str, object],
    workspace: dict[str, object],
    root: dict[str, object],
    deps: dict[str, object],
    manifest: dict[str, object],
    owt_storage: dict[str, object],
    target_gpu_gib: float,
) -> list[str]:
    reasons: list[str] = []
    if not cuda.get("available"):
        reasons.append("CUDA is not available.")
    elif float(cuda["total_memory_gib"]) + 1e-6 < target_gpu_gib:
        reasons.append(
            f"GPU memory {float(cuda['total_memory_gib']):.2f} GiB is below "
            f"the configured target {target_gpu_gib:.2f} GiB."
        )
    if float(root["free_gib"]) < 1.0:
        reasons.append(
            f"Root filesystem free space is {float(root['free_gib']):.2f} GiB; "
            "package/cache operations are likely to fail."
        )
    if not manifest.get("exists"):
        reasons.append("Pinned OpenWebText manifest is missing.")
    if int(manifest.get("shards") or 0) != 80:
        reasons.append(
            f"OpenWebText manifest has {int(manifest.get('shards') or 0)} shards; "
            "the selected policy expects 80."
        )
    required_cleanup = float(owt_storage["required_with_streaming_cleanup_gib"])
    if float(workspace["free_gib"]) < required_cleanup:
        reasons.append(
            f"Workspace free space is {float(workspace['free_gib']):.2f} GiB; "
            f"streaming all-shard OWT with raw cleanup is estimated to need "
            f"{required_cleanup:.2f} GiB."
        )
    reasons.append(
        "Full paper-scale DiT, DiffuMamba-H, and matched ablation training "
        "runs are not complete; current code has toy controls, a bounded local "
        "DiffuMamba-style replacement, and a toy sparse-attention hybrid."
    )
    return reasons


def collect_preflight(
    *,
    policy_path: str | Path,
    manifest_path: str | Path,
    workspace_path: str | Path,
    root_path: str | Path,
    expected_owt_tokens: int = DEFAULT_OWT_TOKEN_COUNT,
    storage_headroom_gib: float = DEFAULT_HEADROOM_GIB,
    target_gpu_gib: float = DEFAULT_TARGET_GPU_GIB,
) -> dict[str, object]:
    policy = load_policy(policy_path)
    deps = dependency_status()
    cuda = cuda_status()
    workspace = disk_status(workspace_path)
    root = disk_status(root_path)
    manifest = manifest_raw_size(manifest_path)
    owt_storage = estimate_owt_storage(
        policy,
        manifest,
        expected_tokens=expected_owt_tokens,
        headroom_gib=storage_headroom_gib,
    )
    reasons = blocking_reasons(
        cuda=cuda,
        workspace=workspace,
        root=root,
        deps=deps,
        manifest=manifest,
        owt_storage=owt_storage,
        target_gpu_gib=target_gpu_gib,
    )
    return {
        "preflight": "paper_baseline_and_owt_scale",
        "status": "blocked" if reasons else "ready",
        "git": git_metadata(),
        "target_hardware": "Tesla V100-SXM2-32GB",
        "environment_policy": (
            "Disposable mutable container; dependencies and caches may be modified."
        ),
        "policy_path": str(policy_path),
        "manifest_path": str(manifest_path),
        "paper_target_matrix": PAPER_TARGET_MATRIX,
        "baseline_models": implementation_status(deps),
        "cuda": cuda,
        "disk": {
            "workspace": workspace,
            "root": root,
        },
        "dependencies": deps,
        "openwebtext_storage": owt_storage,
        "blocking_reasons": reasons,
    }


def write_markdown(path: str | Path, result: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cuda = result["cuda"]
    disk = result["disk"]
    owt = result["openwebtext_storage"]
    lines = [
        "# Paper Baseline Preflight Result",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Environment",
        "",
        f"- Target hardware: {result['target_hardware']}",
        f"- CUDA: {json.dumps(cuda, sort_keys=True)}",
        f"- Workspace free GiB: {disk['workspace']['free_gib']:.2f}",
        f"- Root free GiB: {disk['root']['free_gib']:.2f}",
        f"- Git: `{result['git']['commit']}` dirty={result['git']['dirty']}",
        "",
        "## OpenWebText Storage Estimate",
        "",
        f"- Expected tokens: {owt['expected_tokens']:,}",
        f"- Packed dtype: `{owt['token_dtype']}`",
        f"- Raw shards: {owt['raw_gib']:.2f} GiB",
        f"- Packed tokens: {owt['packed_gib']:.2f} GiB",
        (
            f"- Required with streaming cleanup: "
            f"{owt['required_with_streaming_cleanup_gib']:.2f} GiB"
        ),
        (
            f"- Required without raw cleanup: "
            f"{owt['required_without_raw_cleanup_gib']:.2f} GiB"
        ),
        "",
        "## Blocking Reasons",
        "",
    ]
    reasons = result["blocking_reasons"]
    if reasons:
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Target Matrix",
            "",
            "| Parameter Target | Contexts |",
            "| ---: | --- |",
        ]
    )
    for row in result["paper_target_matrix"]:
        contexts = ", ".join(str(item) for item in row["contexts"])
        lines.append(f"| {row['parameter_target']:,} | {contexts} |")
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| Baseline | Current Status | Required Status |",
            "| --- | --- | --- |",
        ]
    )
    for row in result["baseline_models"]:
        lines.append(
            f"| {row['id']} | {row['current_status']} | {row['required_status']} |"
        )
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/owt_source_policy.json")
    parser.add_argument("--manifest", default="results/owt_stream_manifest_probe.jsonl")
    parser.add_argument("--workspace-path", default="/workspace")
    parser.add_argument("--root-path", default="/")
    parser.add_argument("--expected-owt-tokens", type=int, default=DEFAULT_OWT_TOKEN_COUNT)
    parser.add_argument("--storage-headroom-gib", type=float, default=DEFAULT_HEADROOM_GIB)
    parser.add_argument("--target-gpu-gib", type=float, default=DEFAULT_TARGET_GPU_GIB)
    parser.add_argument("--output-json", default="results/paper_baseline_preflight.json")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect_preflight(
        policy_path=args.policy,
        manifest_path=args.manifest,
        workspace_path=args.workspace_path,
        root_path=args.root_path,
        expected_owt_tokens=args.expected_owt_tokens,
        storage_headroom_gib=args.storage_headroom_gib,
        target_gpu_gib=args.target_gpu_gib,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_md is not None:
        write_markdown(args.output_md, result)
    print(json.dumps({"status": result["status"], "output": str(output_json)}, indent=2))
    if args.fail_on_blocked and result["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
