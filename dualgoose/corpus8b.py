"""Corpus-8B: document filters, dedup, and shard/manifest writing.

New module (append-only): nothing here is imported by the existing 1.5B pathfinder
data path (``scripts/fetch_coder_data.py`` / ``dualgoose.pack_text``), which stays
bit-unchanged. The driver that uses this lives in ``scripts/build_corpus_8b.py``.

THE RULE THIS EXISTS TO SATISFY: unique tokens >= visit budget. The 1.5B pathfinder
memorised a 2.8M-token pool over ~350 epochs (pool NLL 0.006 vs fresh-holdout 5.72,
HumanEval 0/164). The 8B runs ~1B visits, so the pool must carry >=1-2B *unique*
tokens and a genuinely fresh holdout.

Tokenization contract (identical to tpu_bd15a/data/tokens.npy provenance):
  * tokenizer Qwen/Qwen2.5-Coder-1.5B-Instruct, ``add_special_tokens=True``
    (that tokenizer adds no BOS/EOS, so the stream is pure content)
  * int32, contiguous non-overlapping rows of ``SEQ_LEN`` (512), drop remainder
  * shards carry the token stream continuously: a shard boundary never drops
    tokens, so ``np.concatenate([load(s) for s in shards])`` is exactly the
    single matrix a non-sharded build would have produced. Only the final shard
    drops a <512-token remainder.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

SEQ_LEN = 512
TOKENS_DTYPE = np.int32

# Filter reasons, in the order they are applied. Kept as constants so the driver,
# the manifest and the tests all agree on the spelling.
R_SHORT = "too_short"
R_LONG_LINES = "long_lines_minified"
R_LOW_ALPHA = "low_alpha_ratio"
R_BLOB = "unbroken_blob"
R_DUP = "near_duplicate"
R_KEPT = "kept"
FILTER_REASONS = (R_SHORT, R_LONG_LINES, R_LOW_ALPHA, R_BLOB, R_DUP)


@dataclass(frozen=True)
class FilterConfig:
    """Cheap, high-value document filters (the wins the campaign analysis called out).

    Deliberately not a quality *model* — just the four cheap structural rejects
    plus normalized-content dedup.
    """

    min_chars: int = 200
    min_lines: int = 3
    max_line_chars: int = 1000       # any single line this long => minified/data
    max_mean_line_chars: float = 150.0  # whole file reads like a data dump
    min_alpha_ratio: float = 0.25    # letters/total => rejects numeric + binary dumps
    max_unbroken_run: int = 2000     # longest whitespace-free run => base64/blob


_WS_RUN = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    """Normalize a document for near-duplicate detection.

    Collapses every whitespace run to a single space, strips, and lowercases, so
    that reformatted / re-indented / case-shifted copies of the same file collide.
    This is the "near" in near-duplicate: it is an exact hash over a normalized
    form, not an exact hash over raw bytes.
    """
    return _WS_RUN.sub(" ", text).strip().lower()


def doc_hash(text: str) -> int:
    """64-bit blake2b of the normalized document. Stable across processes/runs."""
    norm = normalize_for_hash(text)
    return int.from_bytes(hashlib.blake2b(norm.encode("utf-8", "ignore"), digest_size=8).digest(), "big")


def longest_unbroken_run(text: str) -> int:
    """Length of the longest whitespace-free substring (base64/minified signal).

    ``str.split()`` splits on runs of whitespace in C, so this is ~6x faster than
    scanning characters in Python — and it is exact, not an approximation.
    """
    return max((len(s) for s in text.split()), default=0)


# Above this size the alpha-ratio is estimated from sampled windows instead of
# the whole document: it is a ratio, so three 2KiB windows pin it down far more
# precisely than the 0.25 threshold needs, and it turns an O(n) Python character
# loop (the single hottest thing in the filter) into O(1).
_ALPHA_EXACT_MAX = 8192
_ALPHA_WINDOW = 2048


def alpha_ratio(text: str) -> float:
    """Fraction of characters that are letters; sampled for large documents."""
    n = len(text)
    if n == 0:
        return 0.0
    if n <= _ALPHA_EXACT_MAX:
        return sum(1 for ch in text if ch.isalpha()) / n
    w = _ALPHA_WINDOW
    mid = (n - w) // 2
    sample = text[:w] + text[mid : mid + w] + text[-w:]
    return sum(1 for ch in sample if ch.isalpha()) / len(sample)


def structural_reject(text: str, cfg: FilterConfig = FilterConfig()) -> str | None:
    """Return a filter reason if the doc should be dropped, else ``None``.

    Order is cheapest-first: length, then line geometry, then the character-class
    ratio, then the blob scan. Every stage is a C-level string op or an O(1)
    sample, so this runs at hundreds of MB/s per worker.
    """
    n = len(text)
    if n < cfg.min_chars:
        return R_SHORT
    line_lens = [len(ln) for ln in text.split("\n")]
    if len(line_lens) < cfg.min_lines:
        return R_SHORT
    if max(line_lens) > cfg.max_line_chars:
        return R_LONG_LINES
    if (sum(line_lens) / len(line_lens)) > cfg.max_mean_line_chars:
        return R_LONG_LINES
    if alpha_ratio(text) < cfg.min_alpha_ratio:
        return R_LOW_ALPHA
    if longest_unbroken_run(text) > cfg.max_unbroken_run:
        return R_BLOB
    return None


# --------------------------------------------------------------------------
# holdout routing
# --------------------------------------------------------------------------
HOLDOUT_MOD = 1024


def holdout_bucket(h: int, holdout_buckets: int) -> bool:
    """Deterministic content-addressed holdout routing.

    A document lands in the holdout stratum iff ``h % HOLDOUT_MOD <
    holdout_buckets``. Because the decision is a pure function of the document's
    own normalized hash, a document can NEVER appear in both the train shards and
    the holdout, no matter the order files are processed in or how many workers
    run. Holdout-stratum documents that arrive after the holdout is full are
    dropped outright rather than spilled into training — that disjointness is the
    whole point of the gate (the campaign burned a run on a non-fresh holdout).
    """
    return (h % HOLDOUT_MOD) < holdout_buckets


# --------------------------------------------------------------------------
# shared-memory Bloom filter for cross-process dedup
# --------------------------------------------------------------------------
class ByteBloom:
    """Byte-granular Bloom filter over a shared-memory buffer.

    One *byte* per slot rather than one bit: setting a byte is a single store, so
    concurrent workers never lose a set to a read-modify-write race (a bit-packed
    filter would). Costs 8x memory and buys lock-free correctness.

    Sized by the driver; at 256MiB with k=4 and ~10M documents the false-positive
    rate is ~5e-4, i.e. ~1 in 2000 documents is dropped as a phantom duplicate.
    That is reported in the manifest and is well inside the noise for a 2B-token
    pool.
    """

    def __init__(self, buf, k: int = 4):
        self.buf = buf
        self.size = len(buf)
        self.k = k

    def _slots(self, h: int) -> Iterable[int]:
        # Derive k independent positions from one 64-bit hash by mixing with
        # distinct odd multipliers (Kirsch-Mitzenmacher style double hashing).
        h1 = h & 0xFFFFFFFF
        h2 = (h >> 32) | 1
        for i in range(self.k):
            yield ((h1 + i * h2) * 0x9E3779B1) % self.size

    def add_if_absent(self, h: int) -> bool:
        """Return True if newly added (not seen), False if probably already seen."""
        slots = list(self._slots(h))
        seen = True
        for s in slots:
            if not self.buf[s]:
                seen = False
                break
        if seen:
            return False
        for s in slots:
            self.buf[s] = 1
        return True


def bloom_false_positive_rate(size_bytes: int, k: int, n_items: int) -> float:
    """Analytic FP rate for the byte-granular filter (reported in the manifest)."""
    if size_bytes <= 0 or n_items <= 0:
        return 0.0
    import math

    fill = 1.0 - math.exp(-k * n_items / size_bytes)
    return float(fill**k)


# --------------------------------------------------------------------------
# shard packing
# --------------------------------------------------------------------------
def pack_rows(carry: np.ndarray, new_tokens: np.ndarray, seq_len: int = SEQ_LEN):
    """Split a token stream into whole rows plus the leftover carry.

    ``carry`` is the <seq_len tail left over from the previous call. Returns
    ``(rows, new_carry)`` where ``rows`` is ``[m, seq_len]`` int32. Carrying the
    remainder forward is what makes the shard set concatenate to exactly the
    single-matrix build.
    """
    stream = np.concatenate([carry, new_tokens]) if carry.size else new_tokens
    m = stream.size // seq_len
    if m == 0:
        return np.empty((0, seq_len), dtype=TOKENS_DTYPE), stream
    rows = stream[: m * seq_len].reshape(m, seq_len).astype(TOKENS_DTYPE, copy=False)
    return rows, stream[m * seq_len :]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


@dataclass
class ShardStats:
    name: str
    rows: int
    tokens: int
    bytes: int
    sha256: str

    def to_dict(self) -> dict:
        return {"name": self.name, "rows": self.rows, "tokens": self.tokens,
                "bytes": self.bytes, "sha256": self.sha256}


@dataclass
class ShardWriter:
    """Accumulates rows and flushes fixed-size ``[rows, 512]`` int32 shards.

    The token stream is continuous across shards (see module docstring), so the
    loader may either concatenate the shard list into one matrix or stream shard
    by shard — both yield identical rows in identical order.
    """

    out_dir: str
    prefix: str = "tokens"
    rows_per_shard: int = 700_000  # 700k * 512 * 4B = 1.43 GiB
    seq_len: int = SEQ_LEN
    shards: list = field(default_factory=list)
    _buf: list = field(default_factory=list)
    _buf_rows: int = 0
    _index: int = 0

    def add_rows(self, rows: np.ndarray) -> list:
        """Add rows; returns the list of shard paths completed by this call."""
        written = []
        if rows.shape[0]:
            self._buf.append(rows)
            self._buf_rows += rows.shape[0]
        while self._buf_rows >= self.rows_per_shard:
            allrows = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
            out, rest = allrows[: self.rows_per_shard], allrows[self.rows_per_shard :]
            written.append(self._flush(out))
            self._buf = [rest] if rest.shape[0] else []
            self._buf_rows = rest.shape[0]
        return written

    def finalize(self) -> list:
        """Flush the trailing partial shard (if any)."""
        if not self._buf_rows:
            return []
        allrows = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
        self._buf, self._buf_rows = [], 0
        return [self._flush(allrows)]

    def _flush(self, rows: np.ndarray) -> str:
        import os

        name = f"{self.prefix}-{self._index:05d}.npy"
        path = os.path.join(self.out_dir, name)
        np.save(path, np.ascontiguousarray(rows, dtype=TOKENS_DTYPE))
        self.shards.append(ShardStats(name=name, rows=int(rows.shape[0]),
                                      tokens=int(rows.shape[0]) * self.seq_len,
                                      bytes=os.path.getsize(path), sha256=sha256_file(path)))
        self._index += 1
        return path


def build_manifest(*, shards, holdout, total_tokens, total_rows, source_mix,
                   filter_stats, language_tokens, tokenizer, tokenizer_snapshot,
                   build_date, wall_seconds, extra=None) -> dict:
    """Assemble the manifest mirroring the existing provenance files.

    Carries BOTH views the loader may want: ``shards`` (ordered list with rows /
    tokens / sha256) and ``concat_equivalent``, which states that concatenating
    the shards in listed order reproduces the single ``[total_rows, 512]`` matrix.
    """
    man = {
        "kind": "pretrain_tokens_sharded",
        "purpose": "8B block-diffusion + 3:1 hybrid pretrain pool (unique tokens >= visit budget)",
        "builder": "scripts/build_corpus_8b.py",
        "seq_len": SEQ_LEN,
        "tokens_dtype": "int32",
        "tokenizer": tokenizer,
        "tokenizer_snapshot": tokenizer_snapshot,
        "add_special_tokens": True,
        "bos_added": False,
        "eos_added": False,
        "total_tokens": int(total_tokens),
        "total_rows": int(total_rows),
        "n_shards": len(shards),
        "shards": [s.to_dict() if isinstance(s, ShardStats) else s for s in shards],
        "concat_equivalent": {
            "note": "np.concatenate([np.load(s) for s in shards], axis=0) == the single "
                    "[total_rows, 512] matrix a non-sharded build would emit; the token "
                    "stream is carried across shard boundaries, only the final shard drops "
                    "a <512-token remainder.",
            "shape": [int(total_rows), SEQ_LEN],
            "global_row_index": "shard i holds global rows [sum(rows[:i]), sum(rows[:i+1]))",
        },
        "row_order": {
            "order": "corpus order — documents in the order they were consumed, packed "
                     "contiguously; no shuffling is applied by the builder",
            "clustering": (
                "Rows are LANGUAGE- and SOURCE-CLUSTERED: each source parquet contributes a "
                "long contiguous run and the prose source is one block at the end. This is "
                "harmless for a uniform-random index sampler (jax_port/data.py "
                "sample_indices draws uniform over [0, n_total) with replacement, so every "
                "batch is already a random mix of the whole pool). It matters ONLY for a "
                "SEQUENTIAL reader, which would train one language at a time and see all "
                "prose last — such a reader must shuffle."),
            "shard_order_is_load_bearing": (
                "the shard list order defines the global row index space; pass the shards to "
                "open_token_pool in exactly the order listed here or row indices name "
                "different rows"),
            "safe_to_permute": ("yes — every row is an independent 512-token training "
                                "example; adjacency between consecutive rows is not relied "
                                "upon by the block-diffusion objective"),
        },
        "holdout": holdout,
        "source_mix": source_mix,
        "language_tokens": language_tokens,
        "filter_stats": filter_stats,
        "build_date": build_date,
        "wall_seconds": round(float(wall_seconds), 1),
    }
    if extra:
        man.update(extra)
    return man


def write_manifest(path, manifest: dict) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")
