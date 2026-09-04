import torch

from dualgoose.diffusion import masked_nll_loss
from dualgoose.mixers import BidirectionalRWKVMixer
from dualgoose.model import TinyDualGooseDenoiser
from dualgoose.toy_data import make_bidirectional_copy_batch, make_mqar_batch


def test_mixer_variants_shape():
    x = torch.randn(2, 6, 8)
    t = torch.tensor([0.25, 0.75])
    for direction in ["forward", "backward", "bidirectional"]:
        mixer = BidirectionalRWKVMixer(
            8, direction=direction, merge="static", scan_backend="reference"
        )
        out = mixer(x, t)
        assert out.shape == x.shape


def test_forward_mixer_has_no_future_leakage():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 8)
    mixer = BidirectionalRWKVMixer(8, direction="forward", merge="static")
    out = mixer(x)
    x_changed = x.clone()
    x_changed[:, -1] = x_changed[:, -1] + 5.0
    out_changed = mixer(x_changed)
    assert torch.allclose(out[:, :-1], out_changed[:, :-1])


def test_tiny_denoiser_and_toy_batches_are_training_ready():
    vocab_size = 32
    mask_token_id = vocab_size - 1
    copy_batch = make_bidirectional_copy_batch(4, 16, vocab_size, mask_token_id)
    mqar_batch = make_mqar_batch(4, 16, vocab_size, mask_token_id, pairs=4)
    model = TinyDualGooseDenoiser(
        vocab_size, 16, d_model=16, n_layers=1, scan_backend="reference"
    )
    logits = model(copy_batch.corrupted, copy_batch.t)
    assert logits.shape == (4, 16, vocab_size)
    assert model.scan_backend == "reference"
    assert copy_batch.target_mask.sum().item() == 4
    assert mqar_batch.target_mask.sum().item() == 4


def test_tiny_denoiser_training_updates_remain_finite():
    torch.manual_seed(0)
    vocab_size = 512
    mask_token_id = vocab_size - 1
    model = TinyDualGooseDenoiser(
        vocab_size,
        64,
        d_model=32,
        n_layers=1,
        direction="bidirectional",
        merge="static",
        scan_backend="reference",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    clean = torch.randint(0, mask_token_id, (2, 64))
    corrupted = clean.clone()
    corrupted[:, ::2] = mask_token_id
    t = torch.tensor([0.25, 0.75])

    for _ in range(3):
        logits = model(corrupted, t)
        loss = masked_nll_loss(logits, clean, corrupted, mask_token_id, t=t)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_mqar_query_does_not_overwrite_evidence():
    vocab_size = 64
    mask_token_id = vocab_size - 1
    pairs = 8
    batch_size = 128
    batch = make_mqar_batch(batch_size, 24, vocab_size, mask_token_id, pairs=pairs)
    query_pos = 12
    answer_pos = query_pos + 1
    pair_starts = torch.tensor([0, 2, 4, 6, 8, 10, 14, 16])
    rows = torch.arange(batch_size)
    selected = batch.clean[rows, query_pos] - 5
    value_pos = pair_starts[selected] + 1

    for pair_idx, start in enumerate(pair_starts.tolist()):
        assert batch.clean[:, start].eq(5 + pair_idx).all()
    assert batch.target_mask[:, answer_pos].all()
    assert batch.corrupted[rows, answer_pos].eq(mask_token_id).all()
    assert batch.corrupted[rows, value_pos].ne(mask_token_id).all()
    assert torch.equal(batch.clean[rows, answer_pos], batch.clean[rows, value_pos])


def test_mqar_after_table_layout_keeps_query_after_pairs():
    vocab_size = 64
    mask_token_id = vocab_size - 1
    pairs = 8
    batch_size = 128
    batch = make_mqar_batch(
        batch_size,
        24,
        vocab_size,
        mask_token_id,
        pairs=pairs,
        query_layout="after_table",
    )
    query_pos = 2 * pairs
    answer_pos = query_pos + 1
    rows = torch.arange(batch_size)
    selected = batch.clean[rows, query_pos] - 5
    value_pos = selected * 2 + 1

    for pair_idx in range(pairs):
        assert batch.clean[:, 2 * pair_idx].eq(5 + pair_idx).all()
    assert batch.target_mask[:, answer_pos].all()
    assert batch.corrupted[rows, answer_pos].eq(mask_token_id).all()
    assert batch.corrupted[rows, value_pos].ne(mask_token_id).all()
    assert torch.equal(batch.clean[rows, answer_pos], batch.clean[rows, value_pos])
