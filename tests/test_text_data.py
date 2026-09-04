from pathlib import Path
import json

from dualgoose.text_data import (
    GPT2TextBatcher,
    ByteTextBatcher,
    PackedTextBatcher,
    WordTextBatcher,
    discover_text_paths,
    make_text_batcher,
    pack_text_files,
)


def test_byte_text_batcher_samples_mdlm_batch(tmp_path: Path):
    text_file = tmp_path / "tiny.txt"
    text_file.write_text("hello DualGoose")
    batcher = ByteTextBatcher(text_file, vocab_size=32, mask_token_id=31)
    batch = batcher.sample(batch_size=4, seq_len=12)
    assert batch.clean.shape == (4, 12)
    assert batch.corrupted.shape == (4, 12)
    assert batch.target_mask.shape == (4, 12)
    assert batch.t.shape == (4,)
    assert batch.clean.max().item() < 31
    assert batch.corrupted.max().item() <= 31


def test_word_text_batcher_builds_vocab_and_samples(tmp_path: Path):
    text_file = tmp_path / "tiny.txt"
    text_file.write_text("Goose validates masked diffusion. Goose validates recurrence.")
    batcher = WordTextBatcher(text_file, vocab_size=16, mask_token_id=15)
    batch = batcher.sample(batch_size=3, seq_len=6)
    assert batch.clean.shape == (3, 6)
    assert batch.clean.max().item() < 15
    assert "goose" in batcher.vocab
    assert batcher.vocab["<unk>"] == 0


def test_make_text_batcher_selects_tokenizer(tmp_path: Path):
    text_file = tmp_path / "tiny.txt"
    text_file.write_text("hello world")
    assert isinstance(make_text_batcher(text_file, 16, 15, tokenizer="byte"), ByteTextBatcher)
    assert isinstance(make_text_batcher(text_file, 16, 15, tokenizer="word"), WordTextBatcher)


def test_gpt2_text_batcher_samples_with_mask_token_above_vocab(tmp_path: Path):
    text_file = tmp_path / "tiny.txt"
    text_file.write_text("DualGoose validates GPT-2 tokenization.")
    batcher = GPT2TextBatcher(text_file, vocab_size=50258, mask_token_id=50257)
    batch = batcher.sample(batch_size=2, seq_len=8)
    assert batch.clean.shape == (2, 8)
    assert batch.clean.max().item() < 50257
    assert batch.corrupted.max().item() <= 50257


def test_make_text_batcher_selects_gpt2(tmp_path: Path):
    text_file = tmp_path / "tiny.txt"
    text_file.write_text("hello world")
    assert isinstance(
        make_text_batcher(text_file, 50258, 50257, tokenizer="gpt2"), GPT2TextBatcher
    )


def test_pack_text_files_and_packed_batcher_sample(tmp_path: Path):
    text_file = tmp_path / "tiny.txt"
    text_file.write_text("packed text data path")
    packed_dir = tmp_path / "packed"
    metadata = pack_text_files(
        [text_file],
        packed_dir,
        tokenizer="byte",
        vocab_size=257,
        mask_token_id=256,
    )
    assert metadata["token_count"] > 0
    batcher = PackedTextBatcher(packed_dir, vocab_size=257, mask_token_id=256)
    batch = batcher.sample(batch_size=2, seq_len=8)
    assert batch.clean.shape == (2, 8)
    assert batch.clean.max().item() < 256
    assert batch.corrupted.max().item() <= 256
    assert isinstance(make_text_batcher(packed_dir, 257, 256, tokenizer="packed"), PackedTextBatcher)


def test_discover_text_paths_from_directory_and_manifest(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    first = corpus / "a.txt"
    second = corpus / "b.md"
    first.write_text("alpha")
    second.write_text("beta")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("corpus/b.md\n")

    paths = discover_text_paths(
        input_dirs=[corpus],
        manifest=manifest,
        pattern="*.txt",
    )
    assert paths == [first, second]


def test_pack_text_files_can_write_shards(tmp_path: Path):
    text_file = tmp_path / "tiny.txt"
    text_file.write_text("abcdefghijklmnopqrstuvwxyz")
    packed_dir = tmp_path / "packed_sharded"
    metadata = pack_text_files(
        [text_file],
        packed_dir,
        tokenizer="byte",
        vocab_size=257,
        mask_token_id=256,
        shard_tokens=8,
    )
    assert metadata["format"] == "sharded_npy"
    assert len(metadata["shards"]) > 1
    assert not (packed_dir / "tokens.npy").exists()

    batcher = PackedTextBatcher(packed_dir, vocab_size=257, mask_token_id=256)
    assert batcher.total_tokens == metadata["token_count"]
    batch = batcher.sample(batch_size=4, seq_len=6)
    assert batch.clean.shape == (4, 6)
    assert batch.clean.max().item() < 256


def test_pack_text_files_can_write_train_validation_splits(tmp_path: Path):
    text_file = tmp_path / "tiny.txt"
    text_file.write_text("0123456789" * 10)
    packed_dir = tmp_path / "packed_split"
    metadata = pack_text_files(
        [text_file],
        packed_dir,
        tokenizer="byte",
        vocab_size=257,
        mask_token_id=256,
        shard_tokens=16,
        validation_fraction=0.2,
    )
    assert metadata["format"] == "split_packed"
    assert metadata["splits"]["train"]["token_count"] == 80
    assert metadata["splits"]["validation"]["token_count"] == 20
    assert (packed_dir / "train" / "metadata.json").exists()
    assert (packed_dir / "validation" / "metadata.json").exists()

    train_batcher = PackedTextBatcher(packed_dir / "train", vocab_size=257, mask_token_id=256)
    validation_batcher = PackedTextBatcher(
        packed_dir / "validation", vocab_size=257, mask_token_id=256
    )
    assert train_batcher.total_tokens == 80
    assert validation_batcher.total_tokens == 20
    assert train_batcher.sample(batch_size=2, seq_len=8).clean.shape == (2, 8)
    assert validation_batcher.sample(batch_size=2, seq_len=8).clean.shape == (2, 8)


def test_pack_text_files_can_filter_and_deduplicate_inputs(tmp_path: Path):
    first = tmp_path / "first.txt"
    duplicate = tmp_path / "duplicate.txt"
    short = tmp_path / "short.txt"
    first.write_text("keep me for packed training")
    duplicate.write_text("keep me for packed training")
    short.write_text("tiny")
    packed_dir = tmp_path / "packed_hygiene"

    metadata = pack_text_files(
        [first, duplicate, short],
        packed_dir,
        tokenizer="byte",
        vocab_size=257,
        mask_token_id=256,
        min_bytes=10,
        deduplicate=True,
    )
    assert metadata["input_files"] == [str(first)]
    assert metadata["deduplicate_exact"] is True
    assert metadata["min_bytes"] == 10
    assert metadata["token_count"] == len(first.read_bytes())
    assert {row["reason"] for row in metadata["filtered_files"]} == {
        "duplicate",
        "below_min_bytes",
    }


def test_pack_text_files_can_ingest_jsonl_with_document_filters(tmp_path: Path):
    jsonl_file = tmp_path / "owt_like.jsonl"
    rows = [
        {"text": "This is a valid web text document with alphabetic content."},
        {"text": "This is a valid web text document with alphabetic content."},
        {"text": "short"},
        {"text": "1234567890 !!!!! ?????"},
        {"no_text": "missing"},
    ]
    jsonl_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    packed_dir = tmp_path / "packed_jsonl"
    metadata = pack_text_files(
        [jsonl_file],
        packed_dir,
        tokenizer="byte",
        vocab_size=257,
        mask_token_id=256,
        input_format="jsonl",
        jsonl_text_key="text",
        min_chars=10,
        min_alpha_fraction=0.5,
        deduplicate=True,
        provenance={
            "source_name": "owt-shaped-smoke",
            "source_url": "https://example.invalid/dataset",
            "dataset_version": "smoke",
        },
    )

    assert metadata["input_format"] == "jsonl"
    assert metadata["document_count"] == 1
    assert metadata["provenance"]["source_name"] == "owt-shaped-smoke"
    assert {row["reason"] for row in metadata["filtered_documents"]} == {
        "duplicate_document",
        "below_min_chars",
        "below_min_alpha_fraction",
        "missing_text_key",
    }
    batcher = PackedTextBatcher(packed_dir, vocab_size=257, mask_token_id=256)
    assert batcher.sample(batch_size=2, seq_len=8).clean.shape == (2, 8)
