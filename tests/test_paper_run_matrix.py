from dualgoose.model import build_tiny_denoiser
from dualgoose.paper_run_matrix import (
    RUNNABLE_BASELINES,
    build_run_matrix,
    choose_model_shape,
    model_args_for_candidate,
    parameter_count,
    train_config_for_shape,
    triton_recompute_backward_work_units,
    triton_saved_state_gib,
)


def test_parameter_count_matches_real_small_model():
    model_args = {
        "model_type": "attention",
        "vocab_size": 32,
        "max_length": 16,
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 4,
    }
    real_model = build_tiny_denoiser(**model_args)
    assert parameter_count(model_args) == sum(
        parameter.numel() for parameter in real_model.parameters()
    )


def test_choose_model_shape_returns_closest_candidate():
    baseline = next(item for item in RUNNABLE_BASELINES if item["baseline_id"] == "dualgoose")
    shape = choose_model_shape(
        baseline,
        target_parameters=10_000,
        context=16,
        vocab_size=32,
        widths=[8, 16],
        layers=[1, 2],
    )
    assert shape["d_model"] in {8, 16}
    assert shape["layers"] in {1, 2}
    assert shape["parameters"] > 0
    assert 0.0 <= shape["relative_error"]


def test_build_run_matrix_marks_runnable_and_blocked_configs():
    matrix = build_run_matrix(
        text_file="results/example/train",
        vocab_size=32,
        smoke_steps=3,
        batch_size=2,
        widths=[8],
        layers=[1],
        target_matrix=[{"parameter_target": 10_000, "contexts": [16]}],
    )
    ready = [run for run in matrix["runs"] if run["status"] == "ready_config"]
    blocked = [run for run in matrix["runs"] if run["status"] == "blocked"]

    assert len(ready) == 5
    assert len(blocked) == 0
    assert matrix["replaced_baselines"] == [
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
        }
    ]
    assert {run["baseline_id"] for run in ready} == {
        "dualgoose",
        "transformer_dit",
        "diffumamba_local",
        "diffumamba_h",
        "causal_rwkv7",
    }
    hybrid = next(run for run in ready if run["baseline_id"] == "diffumamba_h")
    assert hybrid["train_config"]["model_type"] == "diffumamba_h"
    assert hybrid["train_config"]["attention_window"] == 64
    assert ready[0]["train_config"]["text_file"] == "results/example/train"
    assert ready[0]["train_config"]["steps"] == 3
    assert ready[0]["train_config"]["batch_size"] == 2


def test_v100_matrix_uses_recompute_for_large_bidirectional_triton_state():
    baseline = next(item for item in RUNNABLE_BASELINES if item["baseline_id"] == "dualgoose")
    model_args = model_args_for_candidate(
        baseline,
        vocab_size=128,
        context=1024,
        d_model=896,
        layers=8,
    )
    shape = {
        "parameters": parameter_count(model_args),
        "model_args": model_args,
    }
    config = train_config_for_shape(
        baseline,
        shape,
        context=1024,
        text_file="results/example/train",
        smoke_steps=1,
        batch_size=1,
    )
    assert triton_saved_state_gib(model_args, batch_size=1) > 28.0
    assert config["scan_backend"] == "triton_recompute"


def test_v100_matrix_keeps_triton_when_saved_state_fits_budget():
    baseline = next(item for item in RUNNABLE_BASELINES if item["baseline_id"] == "causal_rwkv7")
    model_args = model_args_for_candidate(
        baseline,
        vocab_size=128,
        context=1024,
        d_model=896,
        layers=8,
    )
    shape = {
        "parameters": parameter_count(model_args),
        "model_args": model_args,
    }
    config = train_config_for_shape(
        baseline,
        shape,
        context=1024,
        text_file="results/example/train",
        smoke_steps=1,
        batch_size=1,
    )
    assert 0.0 < triton_saved_state_gib(model_args, batch_size=1) <= 28.0
    assert config["scan_backend"] == "triton"


def test_recompute_rows_are_checkpointed_smoke_only_on_v100():
    matrix = build_run_matrix(
        text_file="results/example/train",
        vocab_size=128,
        smoke_steps=100,
        batch_size=1,
        widths=[896],
        layers=[8],
        target_matrix=[{"parameter_target": 10_000_000, "contexts": [1024]}],
    )
    dualgoose = next(run for run in matrix["runs"] if run["baseline_id"] == "dualgoose")
    causal = next(run for run in matrix["runs"] if run["baseline_id"] == "causal_rwkv7")

    assert dualgoose["train_config"]["scan_backend"] == "triton_recompute"
    assert dualgoose["status"] == "smoke_only_config"
    assert dualgoose["train_config"]["steps"] == 5
    assert dualgoose["v100_recompute_runtime_gate"] == "checkpointed_smoke_only"
    assert dualgoose["triton_recompute_checkpoint_interval"] == 64
    assert dualgoose["v100_recompute_max_steps"] == 5
    assert dualgoose["estimated_triton_recompute_backward_work_units"] == (
        1 * 2 * 8 * 1024 * 64 * 896 * 896
    )

    work_model_args = dualgoose["train_config"] | {
        "n_layers": dualgoose["train_config"]["layers"],
        "max_length": 1024,
    }
    assert (
        triton_recompute_backward_work_units(work_model_args, batch_size=1)
        == dualgoose["estimated_triton_recompute_backward_work_units"]
    )
    assert causal["status"] == "ready_config"
    assert causal["train_config"]["steps"] == 100
