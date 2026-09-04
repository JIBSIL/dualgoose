"""Tests for the 8B corpus filters, dedup, holdout routing and shard/manifest writer."""

import ctypes
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest

from dualgoose.corpus8b import (
    HOLDOUT_MOD, R_BLOB, R_DUP, R_LONG_LINES, R_LOW_ALPHA, R_SHORT, SEQ_LEN,
    ByteBloom, FilterConfig, ShardWriter, bloom_false_positive_rate, build_manifest,
    doc_hash, holdout_bucket, longest_unbroken_run, normalize_for_hash, pack_rows,
    sha256_file, structural_reject, write_manifest,
)

GOOD = "\n".join([
    "import os",
    "",
    "def collect(paths):",
    "    \"\"\"Sum the sizes of every path given.\"\"\"",
    "    total = 0",
    "    for p in paths:",
    "        total += os.path.getsize(p)",
    "    return total",
    "",
    "class Walker:",
    "    def __init__(self, root):",
    "        self.root = root",
    "        self.seen = set()",
    "",
    "    def walk(self):",
    "        for dirpath, dirnames, filenames in os.walk(self.root):",
    "            for name in filenames:",
    "                self.seen.add(os.path.join(dirpath, name))",
    "        return sorted(self.seen)",
]) + "\n"
assert len(GOOD) > FilterConfig().min_chars


# ------------------------------------------------------------------ filters
def test_good_code_document_survives_every_filter():
    assert structural_reject(GOOD) is None


def test_too_short_by_chars_and_by_lines():
    assert structural_reject("x = 1\ny = 2\nz = 3\n") == R_SHORT
    # long enough in chars, but only two lines
    two_lines = ("a" * 300) + "\n" + ("b" * 300)
    assert structural_reject(two_lines) in (R_SHORT, R_LONG_LINES)
    assert structural_reject("w" * 100) == R_SHORT


def test_minified_single_long_line_rejected():
    doc = GOOD + "\n" + ("var a=1;" * 400) + "\n"
    assert structural_reject(doc) == R_LONG_LINES


def test_mean_line_length_rejects_data_dump():
    cfg = FilterConfig()
    # every line under max_line_chars but the mean is way over the data-dump bound
    line = "q" * 400
    doc = "\n".join([line] * 30)
    assert max(len(x) for x in doc.split("\n")) <= cfg.max_line_chars
    assert structural_reject(doc) == R_LONG_LINES


def test_low_alpha_ratio_rejects_numeric_dump():
    doc = "\n".join([",".join(["123456"] * 12) for _ in range(40)])
    assert structural_reject(doc) == R_LOW_ALPHA


def test_unbroken_blob_rejected_but_only_after_cheaper_filters():
    cfg = FilterConfig()
    blob = "A1b2C3d4" * 400          # 3200 chars, no whitespace, high alpha
    # wrap so line/mean-line filters do not fire first: split blob across lines
    # would break the run, so instead give one very long line -> long_lines wins.
    doc = GOOD + "\n" + blob + "\n"
    assert structural_reject(doc) == R_LONG_LINES
    # raise the line bounds so the blob detector is the one that fires
    loose = FilterConfig(max_line_chars=10_000, max_mean_line_chars=10_000.0)
    assert structural_reject(doc, loose) == R_BLOB
    assert longest_unbroken_run(doc) == len(blob)
    assert longest_unbroken_run("ab cd\tef\nghij") == 4


def test_filter_order_is_cheapest_first():
    # a doc that trips several filters reports the earliest one
    doc = "0" * 5000                      # short-circuit: one line -> too_short
    assert structural_reject(doc) == R_SHORT


# ------------------------------------------------------- normalization / hash
def test_normalize_collapses_whitespace_and_case():
    assert normalize_for_hash("  def  f(x):\n\n\treturn X  ") == "def f(x): return x"


def test_doc_hash_is_stable_and_reformat_insensitive():
    a = "def f(x):\n    return x\n"
    b = "DEF F(x):\n\n\t\treturn X"
    assert doc_hash(a) == doc_hash(b)
    assert doc_hash(a) == doc_hash(a)
    assert doc_hash(a) != doc_hash("def g(y):\n    return y\n")
    assert 0 <= doc_hash(a) < 2**64


# --------------------------------------------------------------------- bloom
def test_bloom_rejects_second_sighting_and_admits_new():
    buf = mp.RawArray(ctypes.c_uint8, 1 << 20)
    b = ByteBloom(buf)
    h = doc_hash(GOOD)
    assert b.add_if_absent(h) is True
    assert b.add_if_absent(h) is False
    assert b.add_if_absent(doc_hash(GOOD + "# other\n")) is True


def test_bloom_has_no_false_negatives_over_many_items():
    buf = mp.RawArray(ctypes.c_uint8, 1 << 22)
    b = ByteBloom(buf)
    hashes = [doc_hash(f"def f{i}():\n    return {i}\n") for i in range(5000)]
    for h in hashes:
        b.add_if_absent(h)
    # every item already inserted must be reported as seen (no false negatives)
    assert all(b.add_if_absent(h) is False for h in hashes)


def test_bloom_false_positive_rate_is_small_at_build_sizing():
    # 256MiB, k=4, 10M docs -> the rate quoted in the manifest
    assert bloom_false_positive_rate(256 << 20, 4, 10_000_000) < 1e-3
    assert bloom_false_positive_rate(256 << 20, 4, 0) == 0.0


def test_dedup_pipeline_drops_reformatted_duplicate():
    buf = mp.RawArray(ctypes.c_uint8, 1 << 20)
    b = ByteBloom(buf)
    reasons = []
    for doc in (GOOD, GOOD.replace("    ", "\t").upper(), GOOD + "# tail\n"):
        r = structural_reject(doc)
        if r is None:
            r = None if b.add_if_absent(doc_hash(doc)) else R_DUP
        reasons.append(r)
    assert reasons == [None, R_DUP, None]


# ------------------------------------------------------------------- holdout
def test_holdout_routing_is_deterministic_and_disjoint():
    docs = [f"def f{i}(x):\n    return x + {i}\n" * 8 for i in range(4000)]
    hs = [doc_hash(d) for d in docs]
    hold = {d for d, h in zip(docs, hs) if holdout_bucket(h, 3)}
    train = {d for d, h in zip(docs, hs) if not holdout_bucket(h, 3)}
    assert hold and train
    assert hold.isdisjoint(train)          # the invariant the eval gate depends on
    # deterministic across calls
    assert all(holdout_bucket(doc_hash(d), 3) for d in hold)


def test_holdout_bucket_fraction_tracks_the_bucket_count():
    hs = [doc_hash(f"x{i}" * 50) for i in range(20000)]
    frac = sum(holdout_bucket(h, 32) for h in hs) / len(hs)
    assert 32 / HOLDOUT_MOD * 0.7 < frac < 32 / HOLDOUT_MOD * 1.3
    assert not any(holdout_bucket(h, 0) for h in hs)


# ---------------------------------------------------------------- pack_rows
def test_pack_rows_carries_remainder_across_calls():
    rng = np.random.default_rng(0)
    stream = rng.integers(0, 1000, size=SEQ_LEN * 5 + 300, dtype=np.int32)
    carry = np.empty(0, dtype=np.int32)
    rows_a, carry = pack_rows(carry, stream[:1000])
    rows_b, carry = pack_rows(carry, stream[1000:])
    out = np.concatenate([r for r in (rows_a, rows_b) if r.size]).reshape(-1)
    assert rows_a.shape[1] == rows_b.shape[1] == SEQ_LEN
    assert out.size == 5 * SEQ_LEN
    assert carry.size == 300
    # the packed rows are the head of the stream, unpermuted and unduplicated
    np.testing.assert_array_equal(out, stream[: 5 * SEQ_LEN])
    np.testing.assert_array_equal(carry, stream[5 * SEQ_LEN :])


def test_pack_rows_emits_nothing_until_a_full_row_exists():
    carry = np.empty(0, dtype=np.int32)
    rows, carry = pack_rows(carry, np.arange(10, dtype=np.int32))
    assert rows.shape == (0, SEQ_LEN) and carry.size == 10
    assert rows.dtype == np.int32


# -------------------------------------------------------------- ShardWriter
def _feed(writer, total_tokens, seed=0):
    """Push a token stream through pack_rows -> writer; returns the stream."""
    rng = np.random.default_rng(seed)
    stream = rng.integers(0, 150_000, size=total_tokens, dtype=np.int32)
    carry = np.empty(0, dtype=np.int32)
    for i in range(0, stream.size, 4096):
        rows, carry = pack_rows(carry, stream[i : i + 4096])
        writer.add_rows(rows)
    writer.finalize()
    return stream, carry


def test_shards_concatenate_to_the_single_matrix(tmp_path: Path):
    w = ShardWriter(out_dir=str(tmp_path), rows_per_shard=7)
    stream, carry = _feed(w, SEQ_LEN * 30 + 77)
    assert len(w.shards) == 5                     # 30 rows / 7 -> 4 full + 1 of 2
    assert [s.rows for s in w.shards] == [7, 7, 7, 7, 2]
    mats = [np.load(tmp_path / s.name) for s in w.shards]
    concat = np.concatenate(mats, axis=0)
    assert concat.shape == (30, SEQ_LEN)
    assert concat.dtype == np.int32
    # THE contract: shard concatenation == the single non-sharded matrix
    np.testing.assert_array_equal(concat.reshape(-1), stream[: 30 * SEQ_LEN])
    assert carry.size == 77                        # only the tail is dropped


def test_shard_records_carry_true_rows_tokens_and_sha256(tmp_path: Path):
    w = ShardWriter(out_dir=str(tmp_path), rows_per_shard=4)
    _feed(w, SEQ_LEN * 9)
    for s in w.shards:
        p = tmp_path / s.name
        assert s.tokens == s.rows * SEQ_LEN
        assert s.bytes == p.stat().st_size
        assert s.sha256 == sha256_file(p)
        assert np.load(p).shape == (s.rows, SEQ_LEN)
    assert sum(s.rows for s in w.shards) == 9
    assert [s.name for s in w.shards] == [f"tokens-{i:05d}.npy" for i in range(3)]


def test_shard_writer_finalize_is_idempotent_and_empty_safe(tmp_path: Path):
    w = ShardWriter(out_dir=str(tmp_path), rows_per_shard=4)
    assert w.finalize() == []
    _feed(w, SEQ_LEN * 4)
    n = len(w.shards)
    assert w.finalize() == []
    assert len(w.shards) == n


# ------------------------------------------------------------------ manifest
def test_manifest_exposes_both_shard_list_and_concat_view(tmp_path: Path):
    w = ShardWriter(out_dir=str(tmp_path), rows_per_shard=4)
    _feed(w, SEQ_LEN * 10)
    total_rows = sum(s.rows for s in w.shards)
    man = build_manifest(
        shards=w.shards,
        holdout={"name": "holdout8b.npy", "rows": 3, "tokens": 3 * SEQ_LEN},
        total_tokens=total_rows * SEQ_LEN, total_rows=total_rows,
        source_mix={"code_repo": "x", "tokens_by_source": {"code": 1}},
        filter_stats={"kept": 5, "removed_by_filter": {R_SHORT: 1}},
        language_tokens={"Python": 10}, tokenizer="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        tokenizer_snapshot="deadbeef", build_date="2026-08-21", wall_seconds=1.5,
    )
    p = tmp_path / "manifest8b.json"
    write_manifest(p, man)
    loaded = json.loads(p.read_text())

    assert loaded["seq_len"] == SEQ_LEN
    assert loaded["tokens_dtype"] == "int32"
    assert loaded["add_special_tokens"] is True
    assert loaded["n_shards"] == len(w.shards) == 3
    # shard-list view: every entry is loadable and matches its record
    assert sum(s["rows"] for s in loaded["shards"]) == loaded["total_rows"]
    assert sum(s["tokens"] for s in loaded["shards"]) == loaded["total_tokens"]
    for s in loaded["shards"]:
        assert np.load(tmp_path / s["name"]).shape == (s["rows"], SEQ_LEN)
        assert sha256_file(tmp_path / s["name"]) == s["sha256"]
    # concatenated-matrix view: the declared shape is what you actually get
    concat = np.concatenate([np.load(tmp_path / s["name"]) for s in loaded["shards"]])
    assert list(concat.shape) == loaded["concat_equivalent"]["shape"]


def test_manifest_shard_order_is_the_stream_order(tmp_path: Path):
    w = ShardWriter(out_dir=str(tmp_path), rows_per_shard=3)
    stream, _ = _feed(w, SEQ_LEN * 11)
    man = build_manifest(
        shards=w.shards, holdout={}, total_tokens=11 * SEQ_LEN, total_rows=11,
        source_mix={}, filter_stats={}, language_tokens={}, tokenizer="t",
        tokenizer_snapshot="s", build_date="d", wall_seconds=0.0)
    names = [s["name"] for s in man["shards"]]
    assert names == sorted(names)          # listed order == lexical order == stream order
    concat = np.concatenate([np.load(tmp_path / n) for n in names]).reshape(-1)
    np.testing.assert_array_equal(concat, stream[: 11 * SEQ_LEN])


def test_prose_config_keeps_long_paragraphs_that_code_config_rejects():
    """Regression: the code geometry rejected 75% of fineweb-edu, and specifically
    the well-written long-paragraph docs. Prose must use its own config."""
    from scripts.build_corpus_8b import PROSE_FILTER

    para = ("The mitochondrion is an organelle found in most eukaryotic cells, where "
            "it generates most of the chemical energy needed to power the cell's "
            "biochemical reactions, stored in adenosine triphosphate. ") * 4
    doc = "\n\n".join([para] * 3)          # 3 paragraphs, each one very long line
    lines = doc.split("\n")
    mean_line = sum(len(x) for x in lines) / len(lines)
    # real fineweb-edu geometry: median MEAN-line is 229 chars, far over the code bound
    assert mean_line > FilterConfig().max_mean_line_chars
    assert structural_reject(doc, FilterConfig()) == R_LONG_LINES   # the bug
    assert structural_reject(doc, PROSE_FILTER) is None             # the fix


def test_prose_config_still_rejects_junk():
    from scripts.build_corpus_8b import PROSE_FILTER

    assert structural_reject("hi", PROSE_FILTER) == R_SHORT
    # numeric dump: fails the raised alpha floor even though lines are short
    assert structural_reject("\n".join([",".join(["4242"] * 20)] * 30),
                             PROSE_FILTER) == R_LOW_ALPHA
    # a base64 blob inside otherwise wordy text still trips the blob guard
    wordy = ("the quick brown fox jumps over the lazy dog and keeps running " * 30)
    assert structural_reject(wordy + "\n" + "QUJD" * 800 + "\n",
                             PROSE_FILTER) == R_BLOB


def test_alpha_ratio_sampling_agrees_with_exact_on_uniform_text():
    from dualgoose.corpus8b import _ALPHA_EXACT_MAX, alpha_ratio

    big = ("abcdefgh12" * ((_ALPHA_EXACT_MAX * 3) // 10))    # 80% letters, uniform
    assert len(big) > _ALPHA_EXACT_MAX
    assert abs(alpha_ratio(big) - 0.8) < 0.02
    small = "abcdefgh12"
    assert alpha_ratio(small) == 0.8
    assert alpha_ratio("") == 0.0


@pytest.mark.parametrize("reason", [R_SHORT, R_LONG_LINES, R_LOW_ALPHA, R_BLOB])
def test_every_filter_reason_is_reachable(reason):
    cases = {
        R_SHORT: ("hi\n", FilterConfig()),
        R_LONG_LINES: (GOOD + "\n" + "z" * 2000 + "\n", FilterConfig()),
        R_LOW_ALPHA: ("\n".join([",".join(["9999"] * 20)] * 30), FilterConfig()),
        R_BLOB: (GOOD + "\n" + "Ab3" * 900 + "\n",
                 FilterConfig(max_line_chars=10**5, max_mean_line_chars=10**5)),
    }
    doc, cfg = cases[reason]
    assert structural_reject(doc, cfg) == reason
