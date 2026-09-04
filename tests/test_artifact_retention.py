from dualgoose.artifact_retention import apply_checkpoint_plan, build_checkpoint_plan


def test_checkpoint_plan_requires_matching_metrics(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    result_dir = tmp_path / "results"
    checkpoint_dir.mkdir()
    result_dir.mkdir()
    eligible = checkpoint_dir / "paper_slice_dualgoose_100m_1024_smoke.pt"
    missing = checkpoint_dir / "paper_slice_dualgoose_200m_1024_smoke.pt"
    unrelated = checkpoint_dir / "other.pt"
    eligible.write_bytes(b"1234")
    missing.write_bytes(b"12")
    unrelated.write_bytes(b"123456")
    (result_dir / "paper_slice_dualgoose_100m_1024_smoke.json").write_text("{}\n")
    (result_dir / "paper_slice_dualgoose_100m_1024_smoke.csv").write_text("loss\n")
    (result_dir / "paper_slice_dualgoose_200m_1024_smoke.json").write_text("{}\n")

    plan = build_checkpoint_plan(checkpoint_dir=checkpoint_dir, result_dir=result_dir)

    assert plan["total_candidates"] == 2
    assert plan["eligible_count"] == 1
    assert plan["eligible_bytes"] == 4
    assert plan["entries"][0]["eligible_for_delete"] is True
    assert plan["entries"][1]["eligible_for_delete"] is False


def test_checkpoint_plan_can_allow_missing_metrics(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    result_dir = tmp_path / "results"
    checkpoint_dir.mkdir()
    result_dir.mkdir()
    checkpoint = checkpoint_dir / "paper_slice_dualgoose_100m_1024_smoke.pt"
    checkpoint.write_bytes(b"1234")

    plan = build_checkpoint_plan(
        checkpoint_dir=checkpoint_dir,
        result_dir=result_dir,
        require_metrics=False,
    )

    assert plan["eligible_count"] == 1
    assert plan["entries"][0]["reason"] == "metrics_not_required"


def test_checkpoint_plan_supports_matched_pattern(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    result_dir = tmp_path / "results"
    checkpoint_dir.mkdir()
    result_dir.mkdir()
    matched = checkpoint_dir / "paper_matched_dualgoose_100m_2048_5step.pt"
    slice_checkpoint = checkpoint_dir / "paper_slice_dualgoose_100m_2048_smoke.pt"
    matched.write_bytes(b"1234")
    slice_checkpoint.write_bytes(b"123456")
    (result_dir / "paper_matched_dualgoose_100m_2048_5step.json").write_text("{}\n")
    (result_dir / "paper_matched_dualgoose_100m_2048_5step.csv").write_text("loss\n")
    (result_dir / "paper_slice_dualgoose_100m_2048_smoke.json").write_text("{}\n")
    (result_dir / "paper_slice_dualgoose_100m_2048_smoke.csv").write_text("loss\n")

    plan = build_checkpoint_plan(
        checkpoint_dir=checkpoint_dir,
        result_dir=result_dir,
        pattern="paper_matched_*.pt",
    )

    assert plan["pattern"] == "paper_matched_*.pt"
    assert plan["total_candidates"] == 1
    assert plan["eligible_count"] == 1
    assert plan["eligible_bytes"] == 4
    assert plan["entries"][0]["checkpoint"].endswith(str(matched.name))


def test_apply_checkpoint_plan_dry_run_and_delete(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    result_dir = tmp_path / "results"
    checkpoint_dir.mkdir()
    result_dir.mkdir()
    checkpoint = checkpoint_dir / "paper_slice_dualgoose_100m_1024_smoke.pt"
    checkpoint.write_bytes(b"1234")
    (result_dir / "paper_slice_dualgoose_100m_1024_smoke.json").write_text("{}\n")
    (result_dir / "paper_slice_dualgoose_100m_1024_smoke.csv").write_text("loss\n")

    plan = build_checkpoint_plan(checkpoint_dir=checkpoint_dir, result_dir=result_dir)
    dry_run = apply_checkpoint_plan(plan, delete_checkpoints=False)
    assert checkpoint.exists()
    assert dry_run["deleted_count"] == 0
    assert dry_run["entries"][0]["action"] == "would_delete"

    plan = build_checkpoint_plan(checkpoint_dir=checkpoint_dir, result_dir=result_dir)
    deleted = apply_checkpoint_plan(plan, delete_checkpoints=True)
    assert not checkpoint.exists()
    assert deleted["deleted_count"] == 1
    assert deleted["deleted_bytes"] == 4
