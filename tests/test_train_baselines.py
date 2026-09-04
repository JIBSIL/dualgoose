import json
import math
import subprocess
import sys

import torch

from dualgoose.model import (
    TinyAttentionDenoiser,
    TinyDiffuMambaHDenoiser,
    TinyMambaDenoiser,
    build_tiny_denoiser,
)
from dualgoose.train_toy import _move_optimizer_state_to_device


def run_train(
    tmp_path,
    model_type: str,
    *,
    extra_args: list[str] | None = None,
) -> tuple[dict, dict]:
    output = tmp_path / f"{model_type}.json"
    csv_output = tmp_path / f"{model_type}.csv"
    checkpoint = tmp_path / f"{model_type}.pt"
    command = [
        sys.executable,
        "-m",
        "dualgoose.train_toy",
        "--task",
        "copy",
        "--copy-direction",
        "right",
        "--model-type",
        model_type,
        "--steps",
        "1",
        "--batch-size",
        "2",
        "--seq-len",
        "8",
        "--vocab-size",
        "32",
        "--d-model",
        "8",
        "--layers",
        "1",
        "--heads",
        "2",
        "--attention-window",
        "2",
        "--device",
        "cpu",
        "--grad-clip-norm",
        "1.0",
        "--fail-on-nonfinite",
        "--output",
        str(output),
        "--csv-output",
        str(csv_output),
        "--checkpoint-output",
        str(checkpoint),
    ]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, check=True)
    result = json.loads(output.read_text())
    checkpoint_blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return result, checkpoint_blob


def test_model_factory_builds_baseline_denoisers():
    attention = build_tiny_denoiser(
        32,
        8,
        model_type="attention",
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    mamba = build_tiny_denoiser(32, 8, model_type="mamba", d_model=8, n_layers=1)
    diffumamba_h = build_tiny_denoiser(
        32,
        8,
        model_type="diffumamba_h",
        d_model=8,
        n_layers=1,
        n_heads=2,
        attention_window=2,
    )

    assert isinstance(attention, TinyAttentionDenoiser)
    assert isinstance(mamba, TinyMambaDenoiser)
    assert isinstance(diffumamba_h, TinyDiffuMambaHDenoiser)


def test_train_toy_supports_attention_mamba_and_hybrid_baselines(tmp_path):
    for model_type in ["attention", "mamba", "diffumamba_h"]:
        result, checkpoint = run_train(tmp_path, model_type)
        history = result["history"]
        assert result["model_args"]["model_type"] == model_type
        assert result["model_args"]["attention_window"] == 2
        assert checkpoint["model_args"]["model_type"] == model_type
        assert history[0]["step"] == 1
        assert math.isfinite(history[0]["loss"])
        assert math.isfinite(history[0]["grad_norm"])

        model = build_tiny_denoiser(**checkpoint["model_args"])
        model.load_state_dict(checkpoint["model_state"])


def test_train_toy_data_seed_matches_logged_batch_fingerprints(tmp_path):
    extra_args = ["--data-seed", "123", "--log-batch-fingerprint"]
    attention, _ = run_train(tmp_path, "attention", extra_args=extra_args)
    mamba, _ = run_train(tmp_path, "mamba", extra_args=extra_args)

    assert attention["args"]["data_seed"] == 123
    assert attention["args"]["log_batch_fingerprint"] is True

    for key in ["clean_hash", "corrupted_hash", "target_mask_hash", "t_hash"]:
        assert attention["history"][0][key] == mamba["history"][0][key]
    assert attention["history"][0]["mask_fraction"] == mamba["history"][0]["mask_fraction"]


def test_eval_toy_loads_attention_checkpoint(tmp_path):
    _, checkpoint = run_train(tmp_path, "attention")
    checkpoint_path = tmp_path / "attention_eval.pt"
    torch.save(checkpoint, checkpoint_path)
    output = tmp_path / "eval.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "dualgoose.eval_toy",
            "--checkpoint",
            str(checkpoint_path),
            "--task",
            "copy",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
        check=True,
    )

    result = json.loads(output.read_text())
    assert 0.0 <= result["accuracy"] <= 1.0


def test_eval_toy_reports_text_validation_metrics(tmp_path):
    text_file = tmp_path / "text.txt"
    text_file.write_text("DualGoose validates text checkpoints. " * 8)
    output = tmp_path / "text_train.json"
    csv_output = tmp_path / "text_train.csv"
    checkpoint = tmp_path / "text_train.pt"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dualgoose.train_toy",
            "--task",
            "text",
            "--text-file",
            str(text_file),
            "--text-tokenizer",
            "byte",
            "--model-type",
            "attention",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--seq-len",
            "16",
            "--vocab-size",
            "32",
            "--d-model",
            "8",
            "--layers",
            "1",
            "--heads",
            "2",
            "--device",
            "cpu",
            "--grad-clip-norm",
            "1.0",
            "--fail-on-nonfinite",
            "--output",
            str(output),
            "--csv-output",
            str(csv_output),
            "--checkpoint-output",
            str(checkpoint),
        ],
        check=True,
    )
    eval_output = tmp_path / "text_eval.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dualgoose.eval_toy",
            "--task",
            "text",
            "--checkpoint",
            str(checkpoint),
            "--text-file",
            str(text_file),
            "--text-tokenizer",
            "byte",
            "--batch-size",
            "4",
            "--eval-batches",
            "3",
            "--device",
            "cpu",
            "--data-seed",
            "777",
            "--log-batch-fingerprint",
            "--output",
            str(eval_output),
        ],
        check=True,
    )

    result = json.loads(eval_output.read_text())
    assert math.isfinite(result["loss"])
    assert 0.0 <= result["masked_accuracy"] <= 1.0
    assert result["masked_tokens"] > 0
    assert result["tokens"] == 4 * 16 * 3
    assert len(result["batch_fingerprints"]) == 3


def test_train_toy_reports_text_validation_without_perturbing_train_rng(tmp_path):
    train_text = tmp_path / "train.txt"
    eval_text = tmp_path / "eval.txt"
    train_text.write_text("DualGoose trains with periodic text validation. " * 8)
    eval_text.write_text("Validation uses a held out deterministic stream. " * 8)

    def run_text_train(name: str, extra_args: list[str] | None = None) -> dict:
        output = tmp_path / f"{name}.json"
        checkpoint = tmp_path / f"{name}.pt"
        command = [
            sys.executable,
            "-m",
            "dualgoose.train_toy",
            "--task",
            "text",
            "--text-file",
            str(train_text),
            "--text-tokenizer",
            "byte",
            "--model-type",
            "attention",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--seq-len",
            "16",
            "--vocab-size",
            "32",
            "--d-model",
            "8",
            "--layers",
            "1",
            "--heads",
            "2",
            "--device",
            "cpu",
            "--grad-clip-norm",
            "1.0",
            "--fail-on-nonfinite",
            "--data-seed",
            "123",
            "--log-batch-fingerprint",
            "--output",
            str(output),
            "--checkpoint-output",
            str(checkpoint),
        ]
        if extra_args:
            command.extend(extra_args)
        subprocess.run(command, check=True)
        return json.loads(output.read_text())

    baseline = run_text_train("baseline")
    eval_output = tmp_path / "eval_history.json"
    validated = run_text_train(
        "validated",
        [
            "--eval-text-file",
            str(eval_text),
            "--eval-text-tokenizer",
            "byte",
            "--eval-every-steps",
            "1",
            "--eval-batches",
            "2",
            "--eval-data-seed",
            "777",
            "--eval-output",
            str(eval_output),
        ],
    )

    assert [row["step"] for row in validated["eval_history"]] == [1, 2]
    for row in validated["eval_history"]:
        assert math.isfinite(row["validation_loss"])
        assert 0.0 <= row["validation_masked_accuracy"] <= 1.0
        assert row["validation_masked_tokens"] > 0
        assert row["validation_tokens"] == 2 * 16 * 2
        assert row["eval_batches"] == 2
    for baseline_row, validated_row in zip(baseline["history"], validated["history"]):
        for key in ["clean_hash", "corrupted_hash", "target_mask_hash", "t_hash"]:
            assert baseline_row[key] == validated_row[key]
    artifact = json.loads(eval_output.read_text())
    assert artifact["eval_history"] == validated["eval_history"]


def test_train_toy_writes_rolling_checkpoint_and_resumes(tmp_path):
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    final_checkpoint = tmp_path / "final.pt"
    latest_checkpoint = tmp_path / "latest.pt"
    base_command = [
        sys.executable,
        "-m",
        "dualgoose.train_toy",
        "--task",
        "copy",
        "--copy-direction",
        "right",
        "--model-type",
        "attention",
        "--batch-size",
        "2",
        "--seq-len",
        "8",
        "--vocab-size",
        "32",
        "--d-model",
        "8",
        "--layers",
        "1",
        "--heads",
        "2",
        "--attention-window",
        "2",
        "--device",
        "cpu",
        "--grad-clip-norm",
        "1.0",
        "--fail-on-nonfinite",
        "--data-seed",
        "321",
        "--checkpoint-every-steps",
        "1",
        "--checkpoint-latest-output",
        str(latest_checkpoint),
        "--checkpoint-output",
        str(final_checkpoint),
    ]
    subprocess.run(
        [
            *base_command,
            "--steps",
            "1",
            "--output",
            str(first_output),
        ],
        check=True,
    )
    first_latest = torch.load(latest_checkpoint, map_location="cpu", weights_only=True)
    assert first_latest["step"] == 1
    assert "optimizer_state" in first_latest
    assert "rng_state" in first_latest

    subprocess.run(
        [
            *base_command,
            "--steps",
            "2",
            "--resume-checkpoint",
            str(latest_checkpoint),
            "--output",
            str(second_output),
        ],
        check=True,
    )

    result = json.loads(second_output.read_text())
    resumed_latest = torch.load(latest_checkpoint, map_location="cpu", weights_only=True)
    final = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
    assert [row["step"] for row in result["history"]] == [1, 2]
    assert resumed_latest["step"] == 2
    assert final["step"] == 2
    assert len(final["history"]) == 2
    assert not latest_checkpoint.with_name(f".{latest_checkpoint.name}.tmp").exists()
    assert not final_checkpoint.with_name(f".{final_checkpoint.name}.tmp").exists()


def test_optimizer_state_move_handles_nested_tensors():
    model = build_tiny_denoiser(
        32,
        8,
        model_type="attention",
        d_model=8,
        n_layers=1,
        n_heads=2,
        attention_window=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    parameter = next(model.parameters())
    optimizer.state[parameter]["step"] = torch.tensor(1.0)
    optimizer.state[parameter]["nested"] = {
        "list": [torch.ones(1)],
        "tuple": (torch.ones(1),),
    }

    _move_optimizer_state_to_device(optimizer, torch.device("cpu"))

    state = optimizer.state[parameter]
    assert state["step"].device.type == "cpu"
    assert state["nested"]["list"][0].device.type == "cpu"
    assert state["nested"]["tuple"][0].device.type == "cpu"
