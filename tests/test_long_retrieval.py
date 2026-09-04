"""Long-context retrieval generator + probe wiring (docs/armed_gdn_experiment.md follow-up).

Checks the make_long_retrieval_batch invariants that make it a valid distance/length test: answer
masked at the END, key/value table at the START, and — critically — the haystack contains NO key
tokens, so the key->value binding must be carried across the full context. Pure CPU."""

import torch

from dualgoose.model import build_tiny_denoiser
from dualgoose.toy_data import make_long_retrieval_batch


def test_long_retrieval_invariants():
    torch.manual_seed(0)
    B, L, V, pairs = 8, 64, 64, 4
    mask = V - 1
    b = make_long_retrieval_batch(B, L, V, mask, pairs=pairs)
    assert b.clean.shape == (B, L) and b.corrupted.shape == (B, L)
    # exactly one masked target, at the final (query answer) position
    assert b.target_mask[:, -1].all() and int(b.target_mask.sum()) == B
    assert (b.corrupted[:, -1] == mask).all()
    # keys (5..5+pairs-1) must appear ONLY in the table [0,2p) and at the query slot (L-2) —
    # never in the haystack [2p, L-2). Otherwise the long-range binding is ambiguous.
    keys = torch.arange(5, 5 + pairs)
    haystack = b.clean[:, 2 * pairs : L - 2]
    assert not torch.isin(haystack, keys).any(), "haystack leaked a key token"
    # the answer equals the value bound to the query's key
    qkey = b.clean[:, L - 2]
    for r in range(B):
        pidx = int(qkey[r]) - 5
        assert 0 <= pidx < pairs
        assert int(b.clean[r, L - 1]) == int(b.clean[r, 2 * pidx + 1])


def test_long_retrieval_dense_and_gap():
    # JRT extensions: n_queries>1 (dense supervision) + explicit gap (curriculum placement)
    torch.manual_seed(0)
    B, L, V, pairs, nq, gap = 8, 64, 64, 4, 3, 10
    mask = V - 1
    b = make_long_retrieval_batch(B, L, V, mask, pairs=pairs, n_queries=nq, gap=gap)
    tail = 2 * nq
    table_start = L - tail - gap - 2 * pairs
    # all nq answers masked + supervised, at odd offsets of the queries block
    assert int(b.target_mask.sum()) == B * nq
    for q in range(nq):
        apos = L - tail + 2 * q + 1
        assert b.target_mask[:, apos].all() and (b.corrupted[:, apos] == mask).all()
    keys = torch.arange(5, 5 + pairs)
    # keys appear ONLY in the table and at query slots — not in pre-region or gap haystack
    pre = b.clean[:, :table_start]
    gap_region = b.clean[:, table_start + 2 * pairs : L - tail]
    assert not torch.isin(pre, keys).any() and not torch.isin(gap_region, keys).any()
    for r in range(B):
        qkeys = [int(b.clean[r, L - tail + 2 * q]) for q in range(nq)]
        assert len(set(qkeys)) == nq  # distinct queried keys
        for q, qk in enumerate(qkeys):
            pidx = qk - 5
            assert int(b.clean[r, L - tail + 2 * q + 1]) == int(b.clean[r, table_start + 2 * pidx + 1])
    # gap=None (default) still puts the table at position 0 (backward compat)
    b0 = make_long_retrieval_batch(B, L, V, mask, pairs=pairs)
    assert int(b0.clean[0, 0]) in range(5, 5 + pairs)


def test_long_retrieval_collision():
    # collision mode: gap = decoy pairs reusing TABLE keys with garbage values; answer = FIRST binding
    torch.manual_seed(0)
    B, L, V, pairs = 8, 64, 64, 4
    mask = V - 1
    b = make_long_retrieval_batch(B, L, V, mask, pairs=pairs, n_queries=2, collision=True)
    tail, table_start = 4, 0  # gap=None => table at 0
    keys = torch.arange(5, 5 + pairs)
    gap_region = b.clean[:, 2 * pairs : L - tail]
    # decoy KEY slots (even offsets) must be table keys — the lexical gate is dead
    decoy_keys = gap_region[:, 0::2][:, : (L - tail - 2 * pairs) // 2]
    assert torch.isin(decoy_keys, keys).all()
    # answers must equal the FIRST (table) binding, not any decoy garbage
    for r in range(B):
        for q in range(2):
            qk = int(b.clean[r, L - tail + 2 * q]); pidx = qk - 5
            assert int(b.clean[r, L - tail + 2 * q + 1]) == int(b.clean[r, 2 * pidx + 1])
    # non-collision default is unchanged: no keys in the gap
    b2 = make_long_retrieval_batch(B, L, V, mask, pairs=pairs)
    assert not torch.isin(b2.clean[:, 2 * pairs : L - 2], keys).any()


def test_long_retrieval_builds_at_long_seq():
    # the state-matched Mamba reference and an armed GDN both build + forward at long context on CPU
    for mt, kw in [("mamba2_ref", dict(mamba2_d_state=16)),
                   ("gated_deltanet", dict(short_conv=4, merge="static", direction="bidirectional"))]:
        m = build_tiny_denoiser(64, 512, model_type=mt, d_model=32, n_layers=1, **kw)
        y = m(torch.randint(0, 64, (2, 512)), torch.full((2,), 0.5))
        assert y.shape == (2, 512, 64)
