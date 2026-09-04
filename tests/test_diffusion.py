import torch

from dualgoose.diffusion import (
    apply_subs_sample,
    corrupt_absorbing,
    masked_nll_loss,
    suppress_mask_logit,
)


def test_corruption_endpoints():
    x = torch.arange(12, dtype=torch.long).view(2, 6)
    z0, m0 = corrupt_absorbing(x, 0.0, mask_token_id=99)
    z1, m1 = corrupt_absorbing(x, 1.0, mask_token_id=99)
    assert torch.equal(z0, x)
    assert not m0.any()
    assert m1.all()
    assert z1.eq(99).all()


def test_corruption_probability_matches_t():
    generator = torch.Generator().manual_seed(0)
    x = torch.ones(2000, 8, dtype=torch.long)
    _, masked = corrupt_absorbing(x, 0.25, mask_token_id=9, generator=generator)
    assert abs(masked.float().mean().item() - 0.25) < 0.025


def test_masked_nll_is_positive_and_masked_only():
    targets = torch.tensor([[1, 2, 3]])
    corrupted = torch.tensor([[1, 9, 3]])
    logits = torch.zeros(1, 3, 10)
    logits[:, :, 0] = 10.0
    logits[0, 1, 2] = 10.0
    loss = masked_nll_loss(logits, targets, corrupted, mask_token_id=9, t=torch.tensor([0.5]))
    assert loss.item() > 0.0
    corrupted_no_masks = targets.clone()
    zero = masked_nll_loss(logits, targets, corrupted_no_masks, mask_token_id=9)
    assert zero.item() == 0.0


def test_mask_logit_suppression_and_carryover():
    corrupted = torch.tensor([[4, 9]])
    logits = torch.zeros(1, 2, 10)
    logits[..., 9] = 100.0
    logits[0, 1, 3] = 10.0
    clean_logits = suppress_mask_logit(logits, 9)
    assert clean_logits[..., 9].lt(-1e30).all()
    sampled = apply_subs_sample(logits, corrupted, 9)
    assert sampled.tolist() == [[4, 3]]

