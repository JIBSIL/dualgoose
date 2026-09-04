"""Masked discrete diffusion utilities for toy validation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def alpha_linear(t: torch.Tensor) -> torch.Tensor:
    """Linear clean-token survival schedule alpha(t) = 1 - t."""

    return 1.0 - t


def _broadcast_t(t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=x.device, dtype=torch.float32)
    t = t.to(device=x.device, dtype=torch.float32)
    if t.ndim == 0:
        t = t.expand(x.shape[0])
    if t.ndim != 1 or t.shape[0] != x.shape[0]:
        raise ValueError("t must be scalar or have shape (batch,)")
    return t[:, None]


def corrupt_absorbing(
    x: torch.Tensor,
    t: torch.Tensor | float,
    mask_token_id: int,
    *,
    special_token_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply absorbing mask corruption.

    Returns `(z_t, mask_positions)`. `special_token_mask=True` positions are
    never corrupted.
    """

    if x.dtype != torch.long:
        raise TypeError("x must contain integer token ids")
    t_b = _broadcast_t(t, x).clamp(0.0, 1.0)
    mask_prob = 1.0 - alpha_linear(t_b)
    random = torch.rand(x.shape, device=x.device, generator=generator)
    mask_positions = random < mask_prob
    if special_token_mask is not None:
        mask_positions = mask_positions & ~special_token_mask.to(
            device=x.device, dtype=torch.bool
        )
    z_t = torch.where(mask_positions, torch.full_like(x, mask_token_id), x)
    return z_t, mask_positions


def suppress_mask_logit(logits: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """Return logits with the mask token impossible to sample."""

    logits = logits.clone()
    logits[..., mask_token_id] = torch.finfo(logits.dtype).min
    return logits


def masked_nll_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    corrupted: torch.Tensor,
    mask_token_id: int,
    *,
    t: torch.Tensor | None = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Positive MDLM-style masked-token NLL.

    If `t` is supplied, the loss is weighted by `1 / max(t, eps)`, matching the
    linear schedule where `-alpha'(t) / (1 - alpha(t)) = 1 / t`.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, length, vocab)")
    if targets.shape != corrupted.shape or targets.shape != logits.shape[:2]:
        raise ValueError("targets/corrupted must match logits batch and length")

    masked = corrupted.eq(mask_token_id)
    if not masked.any():
        return logits.sum() * 0.0

    clean_logits = suppress_mask_logit(logits, mask_token_id)
    nll = F.cross_entropy(
        clean_logits.view(-1, clean_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)

    if t is not None:
        weights = 1.0 / _broadcast_t(t, targets).clamp_min(eps)
        nll = nll * weights

    return nll[masked].mean()


def apply_subs_sample(
    logits: torch.Tensor,
    corrupted: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Argmax clean tokens at masked positions and carry observed tokens over."""

    predicted = suppress_mask_logit(logits, mask_token_id).argmax(dim=-1)
    return torch.where(corrupted.eq(mask_token_id), predicted, corrupted)

