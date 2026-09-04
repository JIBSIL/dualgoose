"""Retention helpers for generated validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_RESULT_DIR = "results"
DEFAULT_CHECKPOINT_PATTERN = "paper_slice_*.pt"


def checkpoint_metric_paths(checkpoint: Path, result_dir: Path) -> dict[str, Path]:
    stem = checkpoint.stem
    return {
        "json": result_dir / f"{stem}.json",
        "csv": result_dir / f"{stem}.csv",
    }


def build_checkpoint_plan(
    *,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    result_dir: str | Path = DEFAULT_RESULT_DIR,
    pattern: str = DEFAULT_CHECKPOINT_PATTERN,
    require_metrics: bool = True,
) -> dict[str, Any]:
    checkpoint_root = Path(checkpoint_dir)
    result_root = Path(result_dir)
    entries: list[dict[str, Any]] = []
    for checkpoint in sorted(checkpoint_root.glob(pattern)):
        if not checkpoint.is_file():
            continue
        metrics = checkpoint_metric_paths(checkpoint, result_root)
        json_exists = metrics["json"].is_file()
        csv_exists = metrics["csv"].is_file()
        metrics_present = json_exists and csv_exists
        eligible = metrics_present or not require_metrics
        if metrics_present:
            reason = "metrics_present"
        elif require_metrics:
            reason = "missing_metrics"
        else:
            reason = "metrics_not_required"
        entries.append(
            {
                "checkpoint": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "metric_json": str(metrics["json"]),
                "metric_csv": str(metrics["csv"]),
                "metric_json_exists": json_exists,
                "metric_csv_exists": csv_exists,
                "eligible_for_delete": eligible,
                "reason": reason,
            }
        )

    eligible_bytes = sum(entry["bytes"] for entry in entries if entry["eligible_for_delete"])
    return {
        "checkpoint_dir": str(checkpoint_root),
        "result_dir": str(result_root),
        "pattern": pattern,
        "require_metrics": require_metrics,
        "total_candidates": len(entries),
        "eligible_count": sum(1 for entry in entries if entry["eligible_for_delete"]),
        "eligible_bytes": eligible_bytes,
        "eligible_gib": eligible_bytes / 1024**3,
        "entries": entries,
    }


def apply_checkpoint_plan(plan: dict[str, Any], *, delete_checkpoints: bool) -> dict[str, Any]:
    deleted_count = 0
    deleted_bytes = 0
    for entry in plan["entries"]:
        if not entry["eligible_for_delete"]:
            entry["action"] = "kept"
            continue
        if delete_checkpoints:
            checkpoint = Path(entry["checkpoint"])
            if checkpoint.is_file():
                checkpoint.unlink()
                deleted_count += 1
                deleted_bytes += int(entry["bytes"])
                entry["action"] = "deleted"
            else:
                entry["action"] = "already_missing"
        else:
            entry["action"] = "would_delete"

    result = dict(plan)
    result["delete_checkpoints"] = delete_checkpoints
    result["deleted_count"] = deleted_count
    result["deleted_bytes"] = deleted_bytes
    result["deleted_gib"] = deleted_bytes / 1024**3
    return result


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--pattern", default=DEFAULT_CHECKPOINT_PATTERN)
    parser.add_argument(
        "--allow-missing-metrics",
        action="store_true",
        help="Allow checkpoint pruning even when matching JSON/CSV metrics are missing.",
    )
    parser.add_argument(
        "--delete-checkpoints",
        action="store_true",
        help="Delete eligible checkpoints. Omit for a dry-run manifest.",
    )
    parser.add_argument("--manifest-output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_checkpoint_plan(
        checkpoint_dir=args.checkpoint_dir,
        result_dir=args.result_dir,
        pattern=args.pattern,
        require_metrics=not args.allow_missing_metrics,
    )
    manifest = apply_checkpoint_plan(plan, delete_checkpoints=args.delete_checkpoints)
    if args.manifest_output:
        write_manifest(args.manifest_output, manifest)
    summary = {
        "total_candidates": manifest["total_candidates"],
        "eligible_count": manifest["eligible_count"],
        "eligible_gib": manifest["eligible_gib"],
        "deleted_count": manifest["deleted_count"],
        "deleted_gib": manifest["deleted_gib"],
        "manifest_output": args.manifest_output,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
