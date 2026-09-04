import json

from dualgoose.baseline_preflight import (
    DEFAULT_OWT_TOKEN_COUNT,
    bytes_to_gib,
    collect_preflight,
    estimate_owt_storage,
    manifest_raw_size,
    write_markdown,
)


def test_manifest_raw_size_and_owt_storage_estimate(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"rfilename": "a.parquet", "size": 100}),
                json.dumps({"rfilename": "b.parquet", "size": 200}),
            ]
        )
        + "\n"
    )
    policy = {"packed_dtype": "uint16", "mask_token_id": 50257}
    raw = manifest_raw_size(manifest)
    assert raw["exists"] is True
    assert raw["shards"] == 2
    assert raw["raw_bytes"] == 300
    assert raw["max_shard_bytes"] == 200

    estimate = estimate_owt_storage(
        policy,
        raw,
        expected_tokens=1000,
        headroom_gib=0.0,
    )
    assert estimate["token_dtype"] == "uint16"
    assert estimate["packed_bytes"] == 2000
    assert estimate["required_without_raw_cleanup_gib"] == bytes_to_gib(2300)
    assert estimate["required_with_streaming_cleanup_gib"] == bytes_to_gib(2200)


def test_collect_preflight_blocks_paper_scale_when_resources_are_tiny(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "dataset_name": "example/openwebtext",
                "dataset_url": "https://example.invalid/openwebtext",
                "dataset_revision": "abc",
                "selected_file_pattern": "plain_text/*.parquet",
                "expected_shards": 80,
                "raw_text_key": "text",
                "tokenizer": "gpt2",
                "mask_token_id": 50257,
                "vocab_size": 50258,
                "packed_dtype": "uint16",
                "validation_fraction": 0.001,
            }
        )
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"rfilename": "a.parquet", "size": 1024}) + "\n")

    result = collect_preflight(
        policy_path=policy,
        manifest_path=manifest,
        workspace_path=tmp_path,
        root_path=tmp_path,
        expected_owt_tokens=DEFAULT_OWT_TOKEN_COUNT,
        storage_headroom_gib=0.0,
        target_gpu_gib=0.0,
    )
    assert result["status"] == "blocked"
    assert result["openwebtext_storage"]["packed_gib"] > 10.0
    assert any("OpenWebText manifest has 1 shards" in reason for reason in result["blocking_reasons"])
    assert any("toy sparse-attention hybrid" in reason for reason in result["blocking_reasons"])


def test_write_markdown_records_blockers(tmp_path):
    result = {
        "status": "blocked",
        "target_hardware": "Tesla V100-SXM2-32GB",
        "cuda": {"available": True, "name": "test", "total_memory_gib": 32.0},
        "disk": {
            "workspace": {"free_gib": 1.0},
            "root": {"free_gib": 1.0},
        },
        "git": {"commit": "abc", "dirty": False},
        "openwebtext_storage": {
            "expected_tokens": 100,
            "token_dtype": "uint16",
            "raw_gib": 1.0,
            "packed_gib": 2.0,
            "required_with_streaming_cleanup_gib": 3.0,
            "required_without_raw_cleanup_gib": 4.0,
        },
        "blocking_reasons": ["not enough disk"],
        "paper_target_matrix": [{"parameter_target": 100, "contexts": [1024]}],
        "baseline_models": [
            {
                "id": "dualgoose",
                "current_status": "toy",
                "required_status": "paper",
            }
        ],
    }
    output = tmp_path / "preflight.md"
    write_markdown(output, result)
    text = output.read_text()
    assert "Status: `blocked`" in text
    assert "not enough disk" in text
    assert "| 100 | 1024 |" in text
