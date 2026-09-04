"""Synthetic batches for DualGoose toy validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ToyBatch:
    clean: torch.Tensor
    corrupted: torch.Tensor
    target_mask: torch.Tensor
    t: torch.Tensor
    evidence_side: torch.Tensor | None = None


def _randint(
    low: int,
    high: int,
    shape: tuple[int, ...],
    *,
    device: torch.device | str | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randint(low, high, shape, device=device, generator=generator)


def make_bidirectional_copy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    mask_token_id: int,
    *,
    direction: str = "mixed",
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> ToyBatch:
    """Create a left/right evidence copy task.

    Token `1` marks evidence, token `2` marks the query, and the token after an
    evidence marker is the answer that must be copied into the masked query.
    """

    if seq_len < 8:
        raise ValueError("seq_len must be at least 8")
    clean = _randint(3, vocab_size - 1, (batch_size, seq_len), device=device, generator=generator)
    answer_pos = torch.full((batch_size,), seq_len // 2, device=device, dtype=torch.long)
    if direction == "left":
        side = torch.zeros(batch_size, device=device, dtype=torch.long)
    elif direction == "right":
        side = torch.ones(batch_size, device=device, dtype=torch.long)
    elif direction == "mixed":
        side = _randint(0, 2, (batch_size,), device=device, generator=generator)
    else:
        raise ValueError("direction must be left, right, or mixed")

    left_pos = torch.full_like(answer_pos, 1)
    right_pos = torch.full_like(answer_pos, seq_len - 3)
    evidence_pos = torch.where(side.eq(0), left_pos, right_pos)
    values = _randint(3, vocab_size - 1, (batch_size,), device=device, generator=generator)
    rows = torch.arange(batch_size, device=device)
    clean[rows, evidence_pos] = 1
    clean[rows, evidence_pos + 1] = values
    clean[rows, answer_pos - 1] = 2
    clean[rows, answer_pos] = values

    corrupted = clean.clone()
    corrupted[rows, answer_pos] = mask_token_id
    target_mask = torch.zeros_like(clean, dtype=torch.bool)
    target_mask[rows, answer_pos] = True
    t = torch.full((batch_size,), 0.5, device=device)
    return ToyBatch(clean, corrupted, target_mask, t, evidence_side=side)


def make_mqar_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    mask_token_id: int,
    *,
    pairs: int = 4,
    query_layout: str = "middle",
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> ToyBatch:
    """Create a compact masked MQAR-style batch."""

    min_len = 2 * pairs + 4
    if seq_len < min_len:
        raise ValueError("seq_len too short for requested pair count")
    if query_layout not in {"middle", "after_table"}:
        raise ValueError("query_layout must be middle or after_table")
    clean = _randint(5, vocab_size - 1, (batch_size, seq_len), device=device, generator=generator)
    rows = torch.arange(batch_size, device=device)
    if query_layout == "after_table":
        query_index = 2 * pairs
        pair_starts = [2 * pair_idx for pair_idx in range(pairs)]
    else:
        query_index = seq_len // 2
        answer_index = query_index + 1
        pair_starts = []
        pos = 0
        while len(pair_starts) < pairs and pos < seq_len - 1:
            overlaps_query = pos <= answer_index and pos + 1 >= query_index
            if not overlaps_query:
                pair_starts.append(pos)
                pos += 2
            else:
                pos += 1
        if len(pair_starts) < pairs:
            raise ValueError("seq_len too short for non-overlapping MQAR pairs")

    query_pos = torch.full((batch_size,), query_index, device=device, dtype=torch.long)
    answer_pos = query_pos + 1
    selected = _randint(0, pairs, (batch_size,), device=device, generator=generator)
    selected_values = torch.empty(batch_size, device=device, dtype=torch.long)

    for pair_idx in range(pairs):
        key = 5 + pair_idx
        value = _randint(5 + pairs, vocab_size - 1, (batch_size,), device=device, generator=generator)
        pos = pair_starts[pair_idx]
        clean[:, pos] = key
        clean[:, pos + 1] = value
        selected_values = torch.where(selected.eq(pair_idx), value, selected_values)

    clean[rows, query_pos] = 5 + selected
    clean[rows, answer_pos] = selected_values
    corrupted = clean.clone()
    corrupted[rows, answer_pos] = mask_token_id
    target_mask = torch.zeros_like(clean, dtype=torch.bool)
    target_mask[rows, answer_pos] = True
    t = torch.full((batch_size,), 0.5, device=device)
    return ToyBatch(clean, corrupted, target_mask, t)


def make_long_retrieval_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    mask_token_id: int,
    *,
    pairs: int = 4,
    n_queries: int = 1,
    gap: int | None = None,
    collision: bool = False,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> ToyBatch:
    """Long-context retrieval (S-NIAH / long-MQAR).

    Layout: [pre-distractors][table of `pairs` KV pairs][`gap` distractor haystack][queries block].
    The queries block holds `n_queries` DISTINCT keys, each followed by a masked answer slot — all
    supervised (n_queries > 1 = JRT-style dense supervision). `gap` sets the retrieval distance
    between table and queries (None = maximum, table pushed to the start); sampling gap per batch
    during training gives a distance curriculum at fixed seq_len. Distractors are drawn from the
    NON-key range, so a key token appears only in the table and the queries block — the binding must
    be carried across the gap. Defaults (n_queries=1, gap=None) reproduce the original layout.

    `collision=True` (the airtight JRT discriminator): the gap haystack becomes DECOY KV pairs that
    REUSE the table's own keys with garbage values; ground truth = the FIRST (table) binding. This
    removes the lexical write-gate ("is it a key?" no longer discriminates) with no capacity confound
    (still `pairs` real bindings). A causal delta cell whose write gate sees only the current token
    naturally overwrites on key reuse -> retains the LAST (garbage) binding; a backward stream reads
    the query first and, scanning in reverse, its natural overwrite retains the first-in-forward
    (correct) binding. Predicts bidirectional >> causal iff the backward stream is doing the work.
    """

    tail = 2 * n_queries
    if not 1 <= n_queries <= pairs:
        raise ValueError("n_queries must be in [1, pairs]")
    max_gap = seq_len - 2 * pairs - tail
    if max_gap < 0:
        raise ValueError("seq_len too short for requested pair/query count")
    gap = max_gap if gap is None else gap
    if not 0 <= gap <= max_gap:
        raise ValueError("gap out of range")
    key_lo = 5
    val_lo = 5 + pairs  # values + distractors live in [val_lo, vocab_size - 1)
    if val_lo >= vocab_size - 1:
        raise ValueError("vocab_size too small for requested pair count")
    # whole sequence starts as a non-key distractor haystack; table/queries overwrite it
    clean = _randint(val_lo, vocab_size - 1, (batch_size, seq_len), device=device, generator=generator)
    table_start = seq_len - tail - gap - 2 * pairs
    values = torch.empty(batch_size, pairs, dtype=torch.long, device=device)
    for pair_idx in range(pairs):
        value = _randint(val_lo, vocab_size - 1, (batch_size,), device=device, generator=generator)
        pos = table_start + 2 * pair_idx
        clean[:, pos] = key_lo + pair_idx
        clean[:, pos + 1] = value
        values[:, pair_idx] = value
    if collision:
        # fill the gap with decoy pairs reusing the TABLE's keys + garbage values (first binding wins)
        gap_start = table_start + 2 * pairs
        for d in range(gap // 2):
            dk = _randint(0, pairs, (batch_size,), device=device, generator=generator)
            dv = _randint(val_lo, vocab_size - 1, (batch_size,), device=device, generator=generator)
            clean[:, gap_start + 2 * d] = key_lo + dk
            clean[:, gap_start + 2 * d + 1] = dv
        # odd leftover position (if any) stays a plain non-key distractor
    # n_queries DISTINCT pair indices per row (random subset via argsort of uniform noise)
    noise = torch.rand(batch_size, pairs, device=device, generator=generator)
    sel = torch.argsort(noise, dim=1)[:, :n_queries]  # (batch, n_queries)
    corrupted_positions = []
    for q in range(n_queries):
        qpos = seq_len - tail + 2 * q
        clean[:, qpos] = key_lo + sel[:, q]
        clean[:, qpos + 1] = values.gather(1, sel[:, q : q + 1]).squeeze(1)
        corrupted_positions.append(qpos + 1)
    corrupted = clean.clone()
    target_mask = torch.zeros_like(clean, dtype=torch.bool)
    for apos in corrupted_positions:
        corrupted[:, apos] = mask_token_id
        target_mask[:, apos] = True
    t = torch.full((batch_size,), 0.5, device=device)
    return ToyBatch(clean, corrupted, target_mask, t)
