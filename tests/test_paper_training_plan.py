from dualgoose.paper_run_matrix import build_run_matrix
from dualgoose.paper_training_plan import (
    DEFAULT_CHECKPOINT_EVERY_STEPS,
    DEFAULT_DATA_SEED,
    DEFAULT_EVAL_BATCHES,
    DEFAULT_EVAL_DATA_SEED,
    DEFAULT_EVAL_EVERY_STEPS,
    build_training_plan,
    runtime_estimate_hours,
    steps_for_tokens,
    validate_training_plan,
)


def test_steps_for_tokens_rounds_up():
    assert steps_for_tokens(token_budget=1025, batch_size=1, seq_len=1024) == 2
    assert runtime_estimate_hours(baseline_id="dualgoose", token_budget=1000) > 0.0


def test_build_training_plan_primary_runs_are_matched():
    matrix = build_run_matrix(
        text_file="results/owt_stream_full_packed/train",
        vocab_size=128,
        smoke_steps=5,
        batch_size=1,
        widths=[32],
        layers=[1],
        target_matrix=[{"parameter_target": 100_000_000, "contexts": [1024]}],
    )
    plan = build_training_plan(matrix, primary_token_budget=4096, scale_token_budget=2048)
    errors = validate_training_plan(plan)

    assert errors == []
    assert plan["target_hardware"] == "Tesla V100-SXM2-32GB"
    assert plan["primary_objective"]["estimated_steps"] == 4
    assert {run["baseline_id"] for run in plan["primary_objective"]["runs"]} == {
        "dualgoose",
        "transformer_dit",
        "diffumamba_local",
        "diffumamba_h",
        "causal_rwkv7",
    }
    for run in plan["primary_objective"]["runs"]:
        train_config = run["train_config"]
        eval_config = run["eval_config"]
        assert train_config["steps"] == 4
        assert train_config["data_seed"] == DEFAULT_DATA_SEED
        assert train_config["log_batch_fingerprint"] is True
        assert train_config["fail_on_nonfinite"] is True
        assert train_config["checkpoint_every_steps"] == DEFAULT_CHECKPOINT_EVERY_STEPS
        assert train_config["checkpoint_latest_output"].endswith("_latest.pt")
        assert train_config["eval_text_file"] == "results/owt_stream_full_packed/validation"
        assert train_config["eval_text_tokenizer"] == train_config["text_tokenizer"]
        assert train_config["eval_every_steps"] == DEFAULT_EVAL_EVERY_STEPS
        assert train_config["eval_batches"] == DEFAULT_EVAL_BATCHES
        assert train_config["eval_data_seed"] == DEFAULT_EVAL_DATA_SEED
        assert train_config["eval_output"].endswith("_eval_history.json")
        assert eval_config["task"] == "text"
        assert eval_config["text_file"] == "results/owt_stream_full_packed/validation"
        assert eval_config["data_seed"] == DEFAULT_EVAL_DATA_SEED
        assert eval_config["output"] != train_config["eval_output"]
        assert run["v100_runtime_estimate"]["estimated_hours"] > 0.0
    assert (
        plan["checkpoint_policy"]["checkpoint_every_steps"]
        == DEFAULT_CHECKPOINT_EVERY_STEPS
    )
    assert plan["primary_objective"]["estimated_v100_sequential_hours"] > 0.0


def test_validate_training_plan_rejects_missing_baseline():
    matrix = build_run_matrix(
        text_file="results/owt_stream_full_packed/train",
        vocab_size=128,
        smoke_steps=5,
        batch_size=1,
        widths=[32],
        layers=[1],
        target_matrix=[{"parameter_target": 100_000_000, "contexts": [1024]}],
    )
    plan = build_training_plan(matrix, primary_token_budget=4096, scale_token_budget=2048)
    plan["primary_objective"]["runs"].pop()

    errors = validate_training_plan(plan)

    assert any("primary baselines" in error for error in errors)
