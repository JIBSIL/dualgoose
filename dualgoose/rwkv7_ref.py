"""Sequential reference scan for the DualGoose toy RWKV-style recurrence."""

from __future__ import annotations

import torch


def rwkv7_scan(
    v: torch.Tensor,
    k: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    kappa: torch.Tensor,
    a: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    use_rank: bool = True,
    return_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the reference recurrence.

    Shapes:
    - `v`: `(B, L, Dv)`
    - `k`, `r`, `w`, `kappa`, `a`: `(B, L, Dk)`
    - state: `(B, Dv, Dk)`

    The update is `S_t = S_{t-1} A_t + v_t k_t^T` where
    `A_t = diag(w_t) - (a_t * kappa_t) kappa_t^T`.
    """

    if v.ndim != 3 or k.ndim != 3:
        raise ValueError("all inputs must have shape (batch, length, dim)")
    if not (k.shape == r.shape == w.shape == kappa.shape == a.shape):
        raise ValueError("key-axis tensors must have matching shapes")
    if v.shape[:2] != k.shape[:2]:
        raise ValueError("value and key tensors must share batch/length")

    batch, length, d_value = v.shape
    d_key = k.shape[-1]
    if initial_state is None:
        state = v.new_zeros(batch, d_value, d_key)
    else:
        state = initial_state
        if state.shape != (batch, d_value, d_key):
            raise ValueError("initial_state has incompatible shape")

    outputs = []
    states = []
    for idx in range(length):
        state = state * w[:, idx].unsqueeze(1)
        if use_rank:
            removal_key = a[:, idx] * kappa[:, idx]
            removed = torch.bmm(state, removal_key.unsqueeze(-1))
            state = state - removed * kappa[:, idx].unsqueeze(1)
        state = state + v[:, idx].unsqueeze(-1) * k[:, idx].unsqueeze(1)
        outputs.append(torch.bmm(state, r[:, idx].unsqueeze(-1)).squeeze(-1))
        if return_states:
            states.append(state)

    output = torch.stack(outputs, dim=1)
    if return_states:
        return output, torch.stack(states, dim=1)
    return output


def diagonal_scan(
    v: torch.Tensor,
    k: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference diagonal recurrence, equivalent to `rwkv7_scan(use_rank=False)`."""

    zeros = torch.zeros_like(k)
    return rwkv7_scan(
        v,
        k,
        r,
        w,
        zeros,
        zeros,
        initial_state=initial_state,
        use_rank=False,
    )


def reverse_rwkv7_scan(*args: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """Run the scan right-to-left via flip -> scan -> flip."""

    flipped = [arg.flip(1) for arg in args]
    out = rwkv7_scan(*flipped, **kwargs)
    if isinstance(out, tuple):
        return out[0].flip(1), out[1].flip(1)
    return out.flip(1)

