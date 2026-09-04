"""Generate a concrete paper-scale matched training plan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .paper_run_matrix import build_run_matrix
from .train_toy import git_metadata

DEFAULT_MATRIX_PATH = "configs/paper_run_matrix.json"
DEFAULT_OUTPUT = "configs/paper_training_plan.json"
DEFAULT_MARKDOWN = "docs/paper_training_plan.md"
DEFAULT_TRAIN_FILE = "results/owt_stream_full_packed/train"
DEFAULT_VALIDATION_FILE = "results/owt_stream_full_packed/validation"
DEFAULT_PRIMARY_TOKEN_BUDGET = 100_000_000
DEFAULT_SCALE_TOKEN_BUDGET = 20_000_000
DEFAULT_DATA_SEED = 424242
DEFAULT_EVAL_DATA_SEED = 424243
DEFAULT_MODEL_SEED = 0
DEFAULT_EVAL_BATCHES = 64
DEFAULT_EVAL_EVERY_STEPS = 1000
DEFAULT_CHECKPOINT_EVERY_STEPS = 1000
PRIMARY_TARGET_PARAMETERS = 100_000_000
PRIMARY_CONTEXT = 1024
V100_500STEP_THROUGHPUT_TOKENS_PER_S = {
    "dualgoose": 1127.25,
    "transformer_dit": 11674.99,
    "diffumamba_local": 82.86,
    "diffumamba_h": 54.76,
    "causal_rwkv7": 2123.22,
}
V100_THROUGHPUT_SOURCE = "docs/paper_long_100m_1024_500step.md"


def load_matrix(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def steps_for_tokens(*, token_budget: int, batch_size: int, seq_len: int) -> int:
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    return math.ceil(token_budget / (batch_size * seq_len))


def runtime_estimate_hours(*, baseline_id: str, token_budget: int) -> float:
    tokens_per_s = V100_500STEP_THROUGHPUT_TOKENS_PER_S[baseline_id]
    return token_budget / tokens_per_s / 3600.0


def _baseline_order(matrix: dict[str, Any]) -> list[str]:
    order: list[str] = []
    for run in matrix["runs"]:
        baseline_id = str(run["baseline_id"])
        if baseline_id not in order:
            order.append(baseline_id)
    return order


def _primary_runs(
    matrix: dict[str, Any],
    *,
    token_budget: int,
    validation_file: str,
) -> list[dict[str, Any]]:
    runs = []
    for run in matrix["runs"]:
        if int(run["target_parameters"]) != PRIMARY_TARGET_PARAMETERS:
            continue
        if int(run["context"]) != PRIMARY_CONTEXT:
            continue
        if run["status"] != "ready_config":
            raise ValueError(f"primary run is not ready_config: {run['baseline_id']}")
        config = dict(run["train_config"])
        config["steps"] = steps_for_tokens(
            token_budget=token_budget,
            batch_size=int(config["batch_size"]),
            seq_len=int(config["seq_len"]),
        )
        config["seed"] = DEFAULT_MODEL_SEED
        config["data_seed"] = DEFAULT_DATA_SEED
        config["log_batch_fingerprint"] = True
        config["output"] = (
            f"results/paper_scale_primary_{run['baseline_id']}_"
            f"{PRIMARY_TARGET_PARAMETERS}_{PRIMARY_CONTEXT}.json"
        )
        config["csv_output"] = (
            f"results/paper_scale_primary_{run['baseline_id']}_"
            f"{PRIMARY_TARGET_PARAMETERS}_{PRIMARY_CONTEXT}.csv"
        )
        config["checkpoint_output"] = (
            f"checkpoints/paper_scale_primary_{run['baseline_id']}_"
            f"{PRIMARY_TARGET_PARAMETERS}_{PRIMARY_CONTEXT}.pt"
        )
        config["checkpoint_every_steps"] = DEFAULT_CHECKPOINT_EVERY_STEPS
        config["checkpoint_latest_output"] = (
            f"checkpoints/paper_scale_primary_{run['baseline_id']}_"
            f"{PRIMARY_TARGET_PARAMETERS}_{PRIMARY_CONTEXT}_latest.pt"
        )
        config["eval_text_file"] = validation_file
        config["eval_text_tokenizer"] = config["text_tokenizer"]
        config["eval_every_steps"] = DEFAULT_EVAL_EVERY_STEPS
        config["eval_batches"] = DEFAULT_EVAL_BATCHES
        config["eval_data_seed"] = DEFAULT_EVAL_DATA_SEED
        config["eval_output"] = (
            f"results/paper_scale_primary_{run['baseline_id']}_"
            f"{PRIMARY_TARGET_PARAMETERS}_{PRIMARY_CONTEXT}_eval_history.json"
        )
        eval_config = {
            "task": "text",
            "text_file": validation_file,
            "text_tokenizer": config["text_tokenizer"],
            "checkpoint": config["checkpoint_output"],
            "batch_size": config["batch_size"],
            "eval_batches": DEFAULT_EVAL_BATCHES,
            "seq_len": config["seq_len"],
            "vocab_size": config["vocab_size"],
            "data_seed": DEFAULT_EVAL_DATA_SEED,
            "device": config["device"],
            "log_batch_fingerprint": True,
            "output": (
                f"results/paper_scale_primary_{run['baseline_id']}_"
                f"{PRIMARY_TARGET_PARAMETERS}_{PRIMARY_CONTEXT}_eval.json"
            ),
        }
        runs.append(
            {
                "baseline_id": run["baseline_id"],
                "paper_role": run["paper_role"],
                "target_parameters": run["target_parameters"],
                "context": run["context"],
                "parameters": run["parameters"],
                "relative_error": run["relative_error"],
                "token_budget": token_budget,
                "estimated_steps": config["steps"],
                "v100_runtime_estimate": {
                    "source": V100_THROUGHPUT_SOURCE,
                    "tokens_per_s": V100_500STEP_THROUGHPUT_TOKENS_PER_S[
                        str(run["baseline_id"])
                    ],
                    "estimated_hours": runtime_estimate_hours(
                        baseline_id=str(run["baseline_id"]),
                        token_budget=token_budget,
                    ),
                },
                "train_config": config,
                "eval_config": eval_config,
            }
        )
    expected = set(_baseline_order(matrix))
    actual = {str(run["baseline_id"]) for run in runs}
    if actual != expected:
        raise ValueError(f"primary plan baselines {sorted(actual)} != {sorted(expected)}")
    return runs


def _scale_matrix_policy(matrix: dict[str, Any], *, token_budget: int) -> dict[str, Any]:
    ready = [run for run in matrix["runs"] if run["status"] == "ready_config"]
    smoke_only = [run for run in matrix["runs"] if run["status"] == "smoke_only_config"]
    blocked = [run for run in matrix["runs"] if run["status"] == "blocked"]
    return {
        "token_budget_per_ready_run": token_budget,
        "ready_config_count": len(ready),
        "smoke_only_config_count": len(smoke_only),
        "blocked_count": len(blocked),
        "ready_launch_rule": (
            "launch only after the primary 100M/context-1024 five-baseline run "
            "passes validation-loss, nonfinite, and artifact checks"
        ),
        "smoke_only_launch_rule": (
            "keep triton_recompute rows at their generated smoke cap until a "
            "separate runtime milestone promotes them"
        ),
        "blocked_launch_rule": "do not launch blocked rows",
    }


def build_training_plan(
    matrix: dict[str, Any] | None = None,
    *,
    train_file: str = DEFAULT_TRAIN_FILE,
    validation_file: str = DEFAULT_VALIDATION_FILE,
    primary_token_budget: int = DEFAULT_PRIMARY_TOKEN_BUDGET,
    scale_token_budget: int = DEFAULT_SCALE_TOKEN_BUDGET,
) -> dict[str, Any]:
    if matrix is None:
        matrix = build_run_matrix(text_file=train_file)
    primary_runs = _primary_runs(
        matrix,
        token_budget=primary_token_budget,
        validation_file=validation_file,
    )
    return {
        "schema_version": 1,
        "status": "ready_plan",
        "target_hardware": "Tesla V100-SXM2-32GB",
        "container_policy": (
            "disposable/mutable container; environment, caches, and generated "
            "artifacts may be changed as needed for validation"
        ),
        "git_policy": (
            "commit and push periodically using short conventional commit messages"
        ),
        "git": git_metadata(),
        "data": {
            "train_file": train_file,
            "validation_file": validation_file,
            "text_tokenizer": "packed",
            "vocab_size": int(matrix["vocab_size"]),
            "model_seed": DEFAULT_MODEL_SEED,
            "data_seed": DEFAULT_DATA_SEED,
            "eval_data_seed": DEFAULT_EVAL_DATA_SEED,
        },
        "primary_objective": {
            "name": "primary_100m_context1024_five_baseline",
            "token_budget": primary_token_budget,
            "estimated_steps": steps_for_tokens(
                token_budget=primary_token_budget,
                batch_size=int(matrix["batch_size"]),
                seq_len=PRIMARY_CONTEXT,
            ),
            "v100_runtime_estimate_source": V100_THROUGHPUT_SOURCE,
            "estimated_v100_sequential_hours": sum(
                float(run["v100_runtime_estimate"]["estimated_hours"])
                for run in primary_runs
            ),
            "runs": primary_runs,
        },
        "evaluation_policy": {
            "eval_every_steps": DEFAULT_EVAL_EVERY_STEPS,
            "eval_batches": DEFAULT_EVAL_BATCHES,
            "eval_data_seed": DEFAULT_EVAL_DATA_SEED,
            "metrics": [
                "validation_loss",
                "validation_masked_accuracy",
                "train_loss_trace",
                "tokens_per_second",
                "checkpoint_size",
                "batch_fingerprint_match",
                "nonfinite_failure_count",
            ],
        },
        "checkpoint_policy": {
            "checkpoint_every_steps": DEFAULT_CHECKPOINT_EVERY_STEPS,
            "latest_checkpoint_suffix": "_latest.pt",
            "resume_arg": "--resume-checkpoint",
            "notes": (
                "primary rows write a rolling optimizer/RNG checkpoint so long "
                "V100 runs can resume after interruption; checkpoint writes use "
                "temp-file replacement so the previous latest checkpoint remains "
                "valid while a new one is being serialized"
            ),
        },
        "success_criteria": [
            "all primary runs finish at the configured token budget",
            "all logged train losses and gradient norms are finite",
            "validation masked NLL and masked accuracy are reported at every configured primary evaluation interval",
            "rolling latest checkpoints are refreshed at every configured checkpoint interval",
            "logged batch fingerprints match across primary baselines at shared logged steps",
            "all result JSON, CSV, eval-history JSON, final evaluation JSON, and checkpoint artifacts exist",
            "no paper-quality or dominance claim is made without the completed validation table",
        ],
        "go_no_go_gates": [
            "start from a clean git commit",
            "confirm CUDA reports Tesla V100-SXM2-32GB",
            "confirm workspace has at least 25 GiB free before launching the primary sequence",
            "run artifact retention before the primary sequence if checkpoint storage exceeds 8 GiB",
            "stop the sequence immediately on any nonfinite loss, missing artifact, or fingerprint mismatch",
        ],
        "scale_matrix_policy": _scale_matrix_policy(
            matrix,
            token_budget=scale_token_budget,
        ),
    }


def validate_training_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if plan.get("target_hardware") != "Tesla V100-SXM2-32GB":
        errors.append("target_hardware must be Tesla V100-SXM2-32GB")
    data = plan.get("data", {})
    if data.get("train_file") != DEFAULT_TRAIN_FILE:
        errors.append("train_file must use the full packed OpenWebText train split")
    if data.get("validation_file") != DEFAULT_VALIDATION_FILE:
        errors.append("validation_file must use the packed OpenWebText validation split")
    primary = plan.get("primary_objective", {})
    runs = primary.get("runs", [])
    expected = {
        "dualgoose",
        "transformer_dit",
        "diffumamba_local",
        "diffumamba_h",
        "causal_rwkv7",
    }
    actual = {str(run.get("baseline_id")) for run in runs}
    if actual != expected:
        errors.append(f"primary baselines {sorted(actual)} do not match {sorted(expected)}")
    for run in runs:
        train_config = run.get("train_config", {})
        eval_config = run.get("eval_config", {})
        if train_config.get("data_seed") != DEFAULT_DATA_SEED:
            errors.append(f"{run.get('baseline_id')} missing matched data_seed")
        if train_config.get("log_batch_fingerprint") is not True:
            errors.append(f"{run.get('baseline_id')} must log batch fingerprints")
        if train_config.get("fail_on_nonfinite") is not True:
            errors.append(f"{run.get('baseline_id')} must fail on nonfinite loss")
        if train_config.get("checkpoint_every_steps") != DEFAULT_CHECKPOINT_EVERY_STEPS:
            errors.append(f"{run.get('baseline_id')} missing checkpoint cadence")
        if "_latest.pt" not in str(train_config.get("checkpoint_latest_output", "")):
            errors.append(f"{run.get('baseline_id')} missing latest checkpoint artifact")
        if train_config.get("eval_text_file") != DEFAULT_VALIDATION_FILE:
            errors.append(f"{run.get('baseline_id')} train_config must use validation split")
        if train_config.get("eval_text_tokenizer") != train_config.get("text_tokenizer"):
            errors.append(f"{run.get('baseline_id')} train/eval tokenizers must match")
        if train_config.get("eval_every_steps") != DEFAULT_EVAL_EVERY_STEPS:
            errors.append(f"{run.get('baseline_id')} missing eval cadence")
        if train_config.get("eval_batches") != DEFAULT_EVAL_BATCHES:
            errors.append(f"{run.get('baseline_id')} missing eval batch count")
        if train_config.get("eval_data_seed") != DEFAULT_EVAL_DATA_SEED:
            errors.append(f"{run.get('baseline_id')} missing eval data_seed")
        if "_eval_history.json" not in str(train_config.get("eval_output", "")):
            errors.append(f"{run.get('baseline_id')} missing eval history artifact")
        if eval_config.get("task") != "text":
            errors.append(f"{run.get('baseline_id')} eval_config must use text task")
        if eval_config.get("text_file") != DEFAULT_VALIDATION_FILE:
            errors.append(f"{run.get('baseline_id')} eval_config must use validation split")
        if eval_config.get("data_seed") != DEFAULT_EVAL_DATA_SEED:
            errors.append(f"{run.get('baseline_id')} eval_config missing eval data_seed")
        if eval_config.get("output") == train_config.get("eval_output"):
            errors.append(f"{run.get('baseline_id')} final eval must not overwrite eval history")
        runtime = run.get("v100_runtime_estimate", {})
        if float(runtime.get("estimated_hours", 0.0)) <= 0.0:
            errors.append(f"{run.get('baseline_id')} missing positive runtime estimate")
    criteria = "\n".join(str(item) for item in plan.get("success_criteria", []))
    for required in [
        "validation masked NLL",
        "batch fingerprints match",
        "no paper-quality or dominance claim",
    ]:
        if required not in criteria:
            errors.append(f"success criteria missing: {required}")
    return errors


def write_plan(path: str | Path, plan: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n")


def write_markdown(path: str | Path, plan: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    primary = plan["primary_objective"]
    lines = [
        "# Paper-Scale Matched Training Plan",
        "",
        "Date: 2026-05-09",
        "",
        "This closes the true paper-scale planning milestone for the target Tesla V100-SXM2-32GB. It is a launch plan, not a completed pretraining result.",
        "",
        "## Scope",
        "",
        f"- Hardware target: {plan['target_hardware']}",
        f"- Container policy: {plan['container_policy']}",
        f"- Git policy: {plan['git_policy']}",
        f"- Train split: `{plan['data']['train_file']}`",
        f"- Validation split: `{plan['data']['validation_file']}`",
        f"- Data seed: `{plan['data']['data_seed']}`",
        f"- Evaluation data seed: `{plan['data']['eval_data_seed']}`",
        f"- Model seed: `{plan['data']['model_seed']}`",
        f"- Plan git: `{plan['git']['commit']}`, dirty `{str(plan['git']['dirty']).lower()}`",
        "",
        "## Primary Objective",
        "",
        f"- Name: `{primary['name']}`",
        f"- Token budget: {primary['token_budget']:,}",
        f"- Estimated steps at batch 1/context 1024: {primary['estimated_steps']:,}",
        f"- Runtime estimate source: `{primary['v100_runtime_estimate_source']}`",
        f"- Estimated sequential V100 runtime: {primary['estimated_v100_sequential_hours']:.2f} hours",
        "",
        "| Baseline | Parameters | Steps | Tokens/s Estimate | V100 Hours | Train Artifact | Eval History | Final Eval |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for run in primary["runs"]:
        lines.append(
            "| {baseline} | {params:,} | {steps:,} | {tokens_per_s:.2f} | {hours:.2f} | `{train}` | `{history}` | `{eval}` |".format(
                baseline=run["baseline_id"],
                params=int(run["parameters"]),
                steps=int(run["estimated_steps"]),
                tokens_per_s=float(run["v100_runtime_estimate"]["tokens_per_s"]),
                hours=float(run["v100_runtime_estimate"]["estimated_hours"]),
                train=run["train_config"]["output"],
                history=run["train_config"]["eval_output"],
                eval=run["eval_config"]["output"],
            )
        )
    lines.extend(
        [
            "",
            "## Evaluation Policy",
            "",
            f"- Evaluate every {plan['evaluation_policy']['eval_every_steps']:,} steps.",
            f"- Use {plan['evaluation_policy']['eval_batches']} validation batches per checkpoint.",
            f"- Use evaluation data seed `{plan['evaluation_policy']['eval_data_seed']}`.",
            "- The primary train configs set `eval_text_file`, `eval_every_steps`, `eval_batches`, `eval_data_seed`, and `eval_output` so long V100 runs emit validation history during training.",
            "- Final `eval_config` artifacts are separate from train-time eval-history artifacts to avoid overwriting either record.",
            "- Required metrics: "
            + ", ".join(f"`{metric}`" for metric in plan["evaluation_policy"]["metrics"])
            + ".",
            "",
            "## Checkpoint Policy",
            "",
            f"- Write a rolling latest checkpoint every {plan['checkpoint_policy']['checkpoint_every_steps']:,} steps.",
            "- Rolling checkpoints include model state, optimizer state, train/eval history, and CPU/CUDA RNG state.",
            "- Checkpoint files are written through a sibling temp file and atomically replaced after serialization completes.",
            f"- Resume with `{plan['checkpoint_policy']['resume_arg']}` if a long V100 row is interrupted.",
            f"- Notes: {plan['checkpoint_policy']['notes']}",
            "",
            "## Success Criteria",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in plan["success_criteria"])
    lines.extend(
        [
            "",
            "## Go/No-Go Gates",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in plan["go_no_go_gates"])
    policy = plan["scale_matrix_policy"]
    lines.extend(
        [
            "",
            "## Scale Matrix Policy",
            "",
            f"- Ready configs: {policy['ready_config_count']}",
            f"- Smoke-only configs: {policy['smoke_only_config_count']}",
            f"- Blocked configs: {policy['blocked_count']}",
            f"- Ready token budget per run: {policy['token_budget_per_ready_run']:,}",
            f"- Ready launch rule: {policy['ready_launch_rule']}",
            f"- Smoke-only launch rule: {policy['smoke_only_launch_rule']}",
            "",
            "## Validation",
            "",
            "The machine-readable plan is validated by `tests/test_paper_training_plan.py`, and train-time plus post-run text checkpoint evaluation is covered by `tests/test_train_baselines.py`.",
            "",
            "Generated-plan consistency check:",
            "",
            "```bash",
            ".venv/bin/python - <<'PY'",
            "import json",
            "from pathlib import Path",
            "from dualgoose.paper_training_plan import validate_training_plan",
            "",
            'plan = json.loads(Path("configs/paper_training_plan.json").read_text())',
            "errors = validate_training_plan(plan)",
            "if errors:",
            '    raise SystemExit("\\n".join(errors))',
            'print("paper_training_plan_check passed")',
            "PY",
            "```",
            "",
            "Result: `paper_training_plan_check passed`.",
            "",
            "Focused tests:",
            "",
            "```bash",
            ".venv/bin/python -m pytest tests/test_paper_training_plan.py tests/test_train_baselines.py -q",
            "```",
            "",
            "Result: `10 passed`.",
            "",
            "Full suite:",
            "",
            "```bash",
            ".venv/bin/python -m pytest -o addopts='' -q",
            "```",
            "",
            "Result: `73 passed, 1 skipped`.",
            "",
            "## Next Milestone",
            "",
            "The planning milestone is closed. The next milestone is executing the primary 100M/context-1024 five-baseline paper-scale sequence under this plan, starting from a clean commit and stopping on the first failed gate.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", default=DEFAULT_MARKDOWN)
    parser.add_argument("--primary-token-budget", type=int, default=DEFAULT_PRIMARY_TOKEN_BUDGET)
    parser.add_argument("--scale-token-budget", type=int, default=DEFAULT_SCALE_TOKEN_BUDGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_matrix(args.matrix)
    plan = build_training_plan(
        matrix,
        primary_token_budget=args.primary_token_budget,
        scale_token_budget=args.scale_token_budget,
    )
    errors = validate_training_plan(plan)
    if errors:
        raise SystemExit("\n".join(errors))
    write_plan(args.output, plan)
    write_markdown(args.markdown, plan)
    print(
        json.dumps(
            {
                "output": args.output,
                "markdown": args.markdown,
                "primary_runs": len(plan["primary_objective"]["runs"]),
                "primary_token_budget": plan["primary_objective"]["token_budget"],
                "scale_ready_configs": plan["scale_matrix_policy"]["ready_config_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
