import torch

from dualgoose.sweep_toy import (
    TrialSpec,
    mqar_grid_suite,
    mqar_k16_probe_suite,
    mqar_k16_solve_suite,
    mqar_k8_long_suite,
    mqar_k8_solve_suite,
    quick_suite,
    run_trial,
    summarize,
)


def test_sweep_trial_runs_and_summarizes():
    spec = TrialSpec(
        name="copy_smoke",
        task="copy",
        seed=0,
        steps=1,
        batch_size=2,
        eval_batch_size=2,
        seq_len=8,
        vocab_size=16,
        d_model=8,
        layers=1,
        lr=0.001,
        direction="bidirectional",
        copy_direction="right",
    )
    row = run_trial(spec, torch.device("cpu"))
    assert row["name"] == "copy_smoke"
    assert 0.0 <= row["eval_accuracy"] <= 1.0
    summary = summarize([row])
    assert summary[0]["name"] == "copy_smoke"
    assert summary[0]["n"] == 1


def test_sweep_suites_have_expected_trials():
    quick = quick_suite()
    grid = mqar_grid_suite()
    k8_long = mqar_k8_long_suite()
    k8_solve = mqar_k8_solve_suite()
    k16_probe = mqar_k16_probe_suite()
    k16_solve = mqar_k16_solve_suite()
    assert len(quick) == 8
    assert len(grid) == 8
    assert len(k8_long) == 4
    assert len(k8_solve) == 6
    assert len(k16_probe) == 4
    assert len(k16_solve) == 2
    assert {spec.task for spec in grid} == {"mqar"}
    assert {spec.mqar_pairs for spec in grid} == {4, 8}
    assert {spec.mqar_pairs for spec in k8_long} == {8}
    assert {spec.steps for spec in k8_long} == {480}
    assert {spec.mqar_layout for spec in k8_solve} == {"after_table"}
    assert {spec.direction for spec in k8_solve} == {"forward"}
    assert sum(not spec.diagonal_only for spec in k8_solve) == 4
    assert sum(spec.diagonal_only for spec in k8_solve) == 2
    assert {spec.mqar_pairs for spec in k16_probe} == {16}
    assert {spec.mqar_layout for spec in k16_probe} == {"after_table"}
    assert {spec.seq_len for spec in k16_probe} == {40}
    assert {spec.mqar_pairs for spec in k16_solve} == {16}
    assert {spec.steps for spec in k16_solve} == {3000}
    assert {spec.lr for spec in k16_solve} == {0.002}
    assert {spec.layers for spec in k16_solve} == {2}
