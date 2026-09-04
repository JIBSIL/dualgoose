import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dualgoose.prepare_owt as prepare_owt
from dualgoose.prepare_owt import (
    parquet_to_jsonl,
    read_jsonl,
    resolve_token_dtype,
    sha256_file,
    shard_rows_from_siblings,
    stream_pack_manifest,
    write_jsonl,
)
from dualgoose.text_data import PackedTextBatcher


def test_resolve_token_dtype_rejects_mask_that_does_not_fit():
    assert resolve_token_dtype("uint16", mask_token_id=50257) == np.dtype(np.uint16)
    with pytest.raises(ValueError, match="does not fit"):
        resolve_token_dtype("uint16", mask_token_id=70000)


def test_shard_rows_from_siblings_filters_and_records_lfs_metadata(tmp_path):
    policy = {
        "dataset_name": "example/openwebtext",
        "dataset_url": "https://example.invalid/datasets/example/openwebtext",
        "dataset_revision": "abc123",
        "selected_file_pattern": "plain_text/*.parquet",
    }
    siblings = [
        SimpleNamespace(
            rfilename="README.md",
            size=10,
            blob_id="readme",
            lfs=None,
        ),
        SimpleNamespace(
            rfilename="plain_text/train-00000-of-00080.parquet",
            size=123,
            blob_id="blob",
            lfs=SimpleNamespace(sha256="sha"),
        ),
    ]

    rows = shard_rows_from_siblings(policy, siblings)
    assert rows == [
        {
            "dataset_name": "example/openwebtext",
            "dataset_url": "https://example.invalid/datasets/example/openwebtext",
            "dataset_revision": "abc123",
            "rfilename": "plain_text/train-00000-of-00080.parquet",
            "size": 123,
            "blob_id": "blob",
            "lfs_sha256": "sha",
            "resolve_url": (
                "https://huggingface.co/datasets/example/openwebtext/resolve/"
                "abc123/plain_text/train-00000-of-00080.parquet"
            ),
        }
    ]

    path = tmp_path / "manifest.jsonl"
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_parquet_to_jsonl_smoke(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    parquet_path = tmp_path / "sample.parquet"
    pq.write_table(pa.table({"text": ["hello", "world"]}), parquet_path)

    output_path = tmp_path / "sample.jsonl"
    written = parquet_to_jsonl(
        parquet_path,
        output_path,
        text_key="text",
        source={
            "dataset_name": "example/openwebtext",
            "dataset_revision": "abc123",
            "rfilename": "plain_text/train-00000-of-00080.parquet",
        },
    )

    rows = read_jsonl(output_path)
    assert written == 2
    assert [row["text"] for row in rows] == ["hello", "world"]
    assert rows[0]["source_dataset"] == "example/openwebtext"
    assert rows[1]["source_row"] == 1


def test_stream_pack_manifest_writes_incremental_shards(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    parquet_path = tmp_path / "sample.parquet"
    good_text = "alpha beta gamma delta " * 16
    pq.write_table(
        pa.table({"text": [good_text, good_text, "1234 !!!!"]}),
        parquet_path,
    )
    manifest_path = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest_path,
        [
            {
                "dataset_name": "example/openwebtext",
                "dataset_url": "https://example.invalid/datasets/example/openwebtext",
                "dataset_revision": "abc123",
                "rfilename": "plain_text/train-00000-of-00080.parquet",
                "size": parquet_path.stat().st_size,
                "lfs_sha256": sha256_file(parquet_path),
                "local_path": str(parquet_path),
            }
        ],
    )
    output = tmp_path / "packed"
    metadata = stream_pack_manifest(
        {
            "dataset_name": "example/openwebtext",
            "dataset_url": "https://example.invalid/datasets/example/openwebtext",
            "dataset_revision": "abc123",
            "archive_reference_url": "https://example.invalid/archive",
            "raw_text_key": "text",
        },
        manifest_path=manifest_path,
        output_dir=output,
        download_dir=tmp_path / "downloads",
        tokenizer="byte",
        vocab_size=257,
        mask_token_id=256,
        shard_tokens=16,
        validation_fraction=0.0,
        min_chars=16,
        min_alpha_fraction=0.25,
        deduplicate=True,
        batch_size=2,
        token_dtype="uint16",
    )

    assert metadata["format"] == "stream_split_packed"
    assert metadata["dtype"] == "uint16"
    assert metadata["document_count"] == 1
    assert metadata["token_count"] > 16
    assert {row["reason"] for row in metadata["filtered_documents"]} == {
        "duplicate_document",
        "below_min_chars",
    }
    train_metadata = json.loads((output / "train" / "metadata.json").read_text())
    assert train_metadata["dtype"] == "uint16"
    first_shard = np.load(output / "train" / train_metadata["shards"][0]["file"])
    assert first_shard.dtype == np.uint16
    batcher = PackedTextBatcher(output / "train", vocab_size=257, mask_token_id=256)
    assert batcher.total_tokens == metadata["splits"]["train"]["token_count"]


def test_stream_pack_manifest_respects_token_budget_and_deletes_download(
    tmp_path, monkeypatch
):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    download_dir = tmp_path / "downloads"
    parquet_path = download_dir / "plain_text" / "train-00000-of-00080.parquet"
    parquet_path.parent.mkdir(parents=True)
    good_text = "alpha beta gamma delta " * 16
    pq.write_table(pa.table({"text": [good_text, good_text]}), parquet_path)
    manifest_path = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest_path,
        [
            {
                "dataset_name": "example/openwebtext",
                "dataset_url": "https://example.invalid/datasets/example/openwebtext",
                "dataset_revision": "abc123",
                "rfilename": "plain_text/train-00000-of-00080.parquet",
                "size": parquet_path.stat().st_size,
                "lfs_sha256": sha256_file(parquet_path),
            }
        ],
    )

    def fake_download(row, *, local_dir):
        assert Path(local_dir) == download_dir
        return parquet_path

    monkeypatch.setattr(prepare_owt, "download_shard", fake_download)

    output = tmp_path / "packed"
    metadata = stream_pack_manifest(
        {
            "dataset_name": "example/openwebtext",
            "dataset_url": "https://example.invalid/datasets/example/openwebtext",
            "dataset_revision": "abc123",
            "archive_reference_url": "https://example.invalid/archive",
            "raw_text_key": "text",
        },
        manifest_path=manifest_path,
        output_dir=output,
        download_dir=download_dir,
        tokenizer="byte",
        vocab_size=257,
        mask_token_id=256,
        shard_tokens=16,
        validation_fraction=0.0,
        min_chars=1,
        min_alpha_fraction=0.0,
        deduplicate=False,
        batch_size=2,
        token_dtype="uint16",
        max_output_tokens=20,
        delete_raw_after_pack=True,
    )

    assert metadata["token_count"] == 20
    assert metadata["token_budget_reached"] is True
    assert metadata["truncated_documents"] == [
        {
            "source": "plain_text/train-00000-of-00080.parquet:1",
            "reason": "max_output_tokens",
        }
    ]
    assert metadata["deleted_raw_shards"] == [str(parquet_path)]
    assert not parquet_path.exists()
    batcher = PackedTextBatcher(output / "train", vocab_size=257, mask_token_id=256)
    assert batcher.total_tokens == 20


def test_stream_pack_manifest_refuses_to_delete_outside_download_dir(
    tmp_path, monkeypatch
):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    download_dir = tmp_path / "downloads"
    parquet_path = tmp_path / "outside.parquet"
    good_text = "alpha beta gamma delta"
    pq.write_table(pa.table({"text": [good_text]}), parquet_path)
    manifest_path = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest_path,
        [
            {
                "dataset_name": "example/openwebtext",
                "dataset_url": "https://example.invalid/datasets/example/openwebtext",
                "dataset_revision": "abc123",
                "rfilename": "plain_text/train-00000-of-00080.parquet",
                "size": parquet_path.stat().st_size,
                "lfs_sha256": sha256_file(parquet_path),
            }
        ],
    )

    def fake_download(row, *, local_dir):
        assert Path(local_dir) == download_dir
        return parquet_path

    monkeypatch.setattr(prepare_owt, "download_shard", fake_download)

    with pytest.raises(RuntimeError, match="outside download_dir"):
        stream_pack_manifest(
            {
                "dataset_name": "example/openwebtext",
                "dataset_url": "https://example.invalid/datasets/example/openwebtext",
                "dataset_revision": "abc123",
                "archive_reference_url": "https://example.invalid/archive",
                "raw_text_key": "text",
            },
            manifest_path=manifest_path,
            output_dir=tmp_path / "packed",
            download_dir=download_dir,
            tokenizer="byte",
            vocab_size=257,
            mask_token_id=256,
            shard_tokens=16,
            validation_fraction=0.0,
            min_chars=1,
            min_alpha_fraction=0.0,
            deduplicate=False,
            batch_size=1,
            token_dtype="uint16",
            max_output_tokens=1,
            delete_raw_after_pack=True,
        )
    assert parquet_path.exists()
