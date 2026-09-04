"""Triton kernels for the toy RWKV7 scan."""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only when Triton is absent.
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and tl is not None


if triton_available():

    @triton.jit
    def _rwkv7_scan_kernel(
        v_ptr,
        k_ptr,
        r_ptr,
        w_ptr,
        kappa_ptr,
        a_ptr,
        out_ptr,
        states_ptr,
        v_stride_b: tl.constexpr,
        v_stride_l: tl.constexpr,
        v_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_l: tl.constexpr,
        k_stride_d: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_l: tl.constexpr,
        out_stride_d: tl.constexpr,
        states_stride_b: tl.constexpr,
        states_stride_v: tl.constexpr,
        states_stride_l: tl.constexpr,
        states_stride_k: tl.constexpr,
        LENGTH: tl.constexpr,
        D_KEY: tl.constexpr,
        BLOCK_DK: tl.constexpr,
        USE_RANK: tl.constexpr,
        SAVE_STATES: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        value_idx = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_DK)
        mask = offsets < D_KEY
        state = tl.zeros((BLOCK_DK,), dtype=tl.float32)

        for step in range(0, LENGTH):
            value = tl.load(
                v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_idx * v_stride_d
            ).to(tl.float32)
            key_base = k_ptr + batch_idx * k_stride_b + step * k_stride_l
            r_base = r_ptr + batch_idx * k_stride_b + step * k_stride_l
            w_base = w_ptr + batch_idx * k_stride_b + step * k_stride_l
            kappa_base = kappa_ptr + batch_idx * k_stride_b + step * k_stride_l
            a_base = a_ptr + batch_idx * k_stride_b + step * k_stride_l

            key = tl.load(key_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )
            receptance = tl.load(
                r_base + offsets * k_stride_d, mask=mask, other=0.0
            ).to(tl.float32)
            decay = tl.load(w_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )
            kappa = tl.load(
                kappa_base + offsets * k_stride_d, mask=mask, other=0.0
            ).to(tl.float32)
            alpha = tl.load(a_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )

            state = state * decay
            if USE_RANK:
                removal_key = alpha * kappa
                removed = tl.sum(state * removal_key, axis=0)
                state = state - removed * kappa
            state = state + value * key
            if SAVE_STATES:
                tl.store(
                    states_ptr
                    + batch_idx * states_stride_b
                    + value_idx * states_stride_v
                    + step * states_stride_l
                    + offsets * states_stride_k,
                    state,
                    mask=mask,
                )
            output = tl.sum(state * receptance, axis=0)
            tl.store(
                out_ptr
                + batch_idx * out_stride_b
                + step * out_stride_l
                + value_idx * out_stride_d,
                output,
            )

    @triton.jit
    def _rwkv7_scan_checkpoint_kernel(
        v_ptr,
        k_ptr,
        r_ptr,
        w_ptr,
        kappa_ptr,
        a_ptr,
        out_ptr,
        checkpoints_ptr,
        v_stride_b: tl.constexpr,
        v_stride_l: tl.constexpr,
        v_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_l: tl.constexpr,
        k_stride_d: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_l: tl.constexpr,
        out_stride_d: tl.constexpr,
        checkpoints_stride_b: tl.constexpr,
        checkpoints_stride_v: tl.constexpr,
        checkpoints_stride_c: tl.constexpr,
        checkpoints_stride_k: tl.constexpr,
        LENGTH: tl.constexpr,
        D_KEY: tl.constexpr,
        BLOCK_DK: tl.constexpr,
        USE_RANK: tl.constexpr,
        CHECKPOINT_INTERVAL: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        value_idx = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_DK)
        mask = offsets < D_KEY
        state = tl.zeros((BLOCK_DK,), dtype=tl.float32)
        checkpoint_base = (
            checkpoints_ptr
            + batch_idx * checkpoints_stride_b
            + value_idx * checkpoints_stride_v
        )
        tl.store(
            checkpoint_base + offsets * checkpoints_stride_k,
            state,
            mask=mask,
        )

        for step in range(0, LENGTH):
            value = tl.load(
                v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_idx * v_stride_d
            ).to(tl.float32)
            key_base = k_ptr + batch_idx * k_stride_b + step * k_stride_l
            r_base = r_ptr + batch_idx * k_stride_b + step * k_stride_l
            w_base = w_ptr + batch_idx * k_stride_b + step * k_stride_l
            kappa_base = kappa_ptr + batch_idx * k_stride_b + step * k_stride_l
            a_base = a_ptr + batch_idx * k_stride_b + step * k_stride_l

            key = tl.load(key_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )
            receptance = tl.load(
                r_base + offsets * k_stride_d, mask=mask, other=0.0
            ).to(tl.float32)
            decay = tl.load(w_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )
            kappa = tl.load(
                kappa_base + offsets * k_stride_d, mask=mask, other=0.0
            ).to(tl.float32)
            alpha = tl.load(a_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )

            state = state * decay
            if USE_RANK:
                removal_key = alpha * kappa
                removed = tl.sum(state * removal_key, axis=0)
                state = state - removed * kappa
            state = state + value * key
            output = tl.sum(state * receptance, axis=0)
            tl.store(
                out_ptr
                + batch_idx * out_stride_b
                + step * out_stride_l
                + value_idx * out_stride_d,
                output,
            )
            if ((step + 1) % CHECKPOINT_INTERVAL == 0) or (step == LENGTH - 1):
                checkpoint_idx = (step + CHECKPOINT_INTERVAL) // CHECKPOINT_INTERVAL
                tl.store(
                    checkpoint_base
                    + checkpoint_idx * checkpoints_stride_c
                    + offsets * checkpoints_stride_k,
                    state,
                    mask=mask,
                )

    @triton.jit
    def _rwkv7_diag_backward_kernel(
        v_ptr,
        k_ptr,
        r_ptr,
        w_ptr,
        states_ptr,
        grad_out_ptr,
        grad_v_ptr,
        grad_k_ptr,
        grad_r_ptr,
        grad_w_ptr,
        v_stride_b: tl.constexpr,
        v_stride_l: tl.constexpr,
        v_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_l: tl.constexpr,
        k_stride_d: tl.constexpr,
        states_stride_b: tl.constexpr,
        states_stride_v: tl.constexpr,
        states_stride_l: tl.constexpr,
        states_stride_k: tl.constexpr,
        grad_out_stride_b: tl.constexpr,
        grad_out_stride_l: tl.constexpr,
        grad_out_stride_d: tl.constexpr,
        LENGTH: tl.constexpr,
        D_KEY: tl.constexpr,
        BLOCK_DK: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        value_idx = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_DK)
        mask = offsets < D_KEY
        grad_state = tl.zeros((BLOCK_DK,), dtype=tl.float32)

        for reverse_step in range(0, LENGTH):
            step = LENGTH - 1 - reverse_step
            key_base = k_ptr + batch_idx * k_stride_b + step * k_stride_l
            r_base = r_ptr + batch_idx * k_stride_b + step * k_stride_l
            w_base = w_ptr + batch_idx * k_stride_b + step * k_stride_l
            state_base = (
                states_ptr
                + batch_idx * states_stride_b
                + value_idx * states_stride_v
                + step * states_stride_l
            )

            key = tl.load(key_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )
            receptance = tl.load(
                r_base + offsets * k_stride_d, mask=mask, other=0.0
            ).to(tl.float32)
            decay = tl.load(w_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )
            state = tl.load(
                state_base + offsets * states_stride_k, mask=mask, other=0.0
            ).to(tl.float32)
            if step == 0:
                prev_state = tl.zeros((BLOCK_DK,), dtype=tl.float32)
            else:
                prev_state = tl.load(
                    state_base
                    - states_stride_l
                    + offsets * states_stride_k,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
            value = tl.load(
                v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_idx * v_stride_d
            ).to(tl.float32)
            grad_output = tl.load(
                grad_out_ptr
                + batch_idx * grad_out_stride_b
                + step * grad_out_stride_l
                + value_idx * grad_out_stride_d
            ).to(tl.float32)

            grad_state = grad_state + grad_output * receptance
            grad_v = tl.sum(grad_state * key, axis=0)
            grad_key_base = (
                batch_idx * k_stride_b + step * k_stride_l + offsets * k_stride_d
            )
            tl.store(
                grad_v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_idx * v_stride_d,
                grad_v,
            )
            tl.atomic_add(
                grad_k_ptr + grad_key_base,
                grad_state * value,
                mask=mask,
            )
            tl.atomic_add(
                grad_r_ptr + grad_key_base,
                grad_output * state,
                mask=mask,
            )
            tl.atomic_add(
                grad_w_ptr + grad_key_base,
                grad_state * prev_state,
                mask=mask,
            )
            grad_state = grad_state * decay

    @triton.jit
    def _rwkv7_rank_backward_kernel(
        v_ptr,
        k_ptr,
        r_ptr,
        w_ptr,
        kappa_ptr,
        a_ptr,
        states_ptr,
        grad_out_ptr,
        grad_v_ptr,
        grad_k_ptr,
        grad_r_ptr,
        grad_w_ptr,
        grad_kappa_ptr,
        grad_a_ptr,
        v_stride_b: tl.constexpr,
        v_stride_l: tl.constexpr,
        v_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_l: tl.constexpr,
        k_stride_d: tl.constexpr,
        states_stride_b: tl.constexpr,
        states_stride_v: tl.constexpr,
        states_stride_l: tl.constexpr,
        states_stride_k: tl.constexpr,
        grad_out_stride_b: tl.constexpr,
        grad_out_stride_l: tl.constexpr,
        grad_out_stride_d: tl.constexpr,
        LENGTH: tl.constexpr,
        D_KEY: tl.constexpr,
        BLOCK_DK: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        value_idx = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_DK)
        mask = offsets < D_KEY
        grad_state = tl.zeros((BLOCK_DK,), dtype=tl.float32)

        for reverse_step in range(0, LENGTH):
            step = LENGTH - 1 - reverse_step
            key_base = k_ptr + batch_idx * k_stride_b + step * k_stride_l
            r_base = r_ptr + batch_idx * k_stride_b + step * k_stride_l
            w_base = w_ptr + batch_idx * k_stride_b + step * k_stride_l
            kappa_base = kappa_ptr + batch_idx * k_stride_b + step * k_stride_l
            a_base = a_ptr + batch_idx * k_stride_b + step * k_stride_l
            state_base = (
                states_ptr
                + batch_idx * states_stride_b
                + value_idx * states_stride_v
                + step * states_stride_l
            )

            key = tl.load(key_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )
            receptance = tl.load(
                r_base + offsets * k_stride_d, mask=mask, other=0.0
            ).to(tl.float32)
            decay = tl.load(w_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )
            kappa = tl.load(
                kappa_base + offsets * k_stride_d, mask=mask, other=0.0
            ).to(tl.float32)
            alpha = tl.load(a_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )
            state = tl.load(
                state_base + offsets * states_stride_k, mask=mask, other=0.0
            ).to(tl.float32)
            if step == 0:
                prev_state = tl.zeros((BLOCK_DK,), dtype=tl.float32)
            else:
                prev_state = tl.load(
                    state_base
                    - states_stride_l
                    + offsets * states_stride_k,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
            value = tl.load(
                v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_idx * v_stride_d
            ).to(tl.float32)
            grad_output = tl.load(
                grad_out_ptr
                + batch_idx * grad_out_stride_b
                + step * grad_out_stride_l
                + value_idx * grad_out_stride_d
            ).to(tl.float32)

            grad_state = grad_state + grad_output * receptance
            grad_v = tl.sum(grad_state * key, axis=0)
            grad_key_base = (
                batch_idx * k_stride_b + step * k_stride_l + offsets * k_stride_d
            )
            state_before_rank = prev_state * decay
            removal_key = alpha * kappa
            removed = tl.sum(state_before_rank * removal_key, axis=0)
            grad_removed = -tl.sum(grad_state * kappa, axis=0)
            grad_state_before_rank = grad_state + grad_removed * removal_key

            tl.store(
                grad_v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_idx * v_stride_d,
                grad_v,
            )
            tl.atomic_add(
                grad_k_ptr + grad_key_base,
                grad_state * value,
                mask=mask,
            )
            tl.atomic_add(
                grad_r_ptr + grad_key_base,
                grad_output * state,
                mask=mask,
            )
            tl.atomic_add(
                grad_w_ptr + grad_key_base,
                grad_state_before_rank * prev_state,
                mask=mask,
            )
            tl.atomic_add(
                grad_kappa_ptr + grad_key_base,
                -removed * grad_state + grad_removed * state_before_rank * alpha,
                mask=mask,
            )
            tl.atomic_add(
                grad_a_ptr + grad_key_base,
                grad_removed * state_before_rank * kappa,
                mask=mask,
            )
            grad_state = grad_state_before_rank * decay

    @triton.jit
    def _rwkv7_diag_backward_grouped_kernel(
        v_ptr,
        k_ptr,
        r_ptr,
        w_ptr,
        states_ptr,
        grad_out_ptr,
        grad_v_ptr,
        grad_k_ptr,
        grad_r_ptr,
        grad_w_ptr,
        v_stride_b: tl.constexpr,
        v_stride_l: tl.constexpr,
        v_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_l: tl.constexpr,
        k_stride_d: tl.constexpr,
        states_stride_b: tl.constexpr,
        states_stride_v: tl.constexpr,
        states_stride_l: tl.constexpr,
        states_stride_k: tl.constexpr,
        grad_out_stride_b: tl.constexpr,
        grad_out_stride_l: tl.constexpr,
        grad_out_stride_d: tl.constexpr,
        LENGTH: tl.constexpr,
        D_VALUE: tl.constexpr,
        D_KEY: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        BLOCK_DK: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        value_block = tl.program_id(1)
        value_offsets = value_block * BLOCK_DV + tl.arange(0, BLOCK_DV)
        key_offsets = tl.arange(0, BLOCK_DK)
        value_mask = value_offsets < D_VALUE
        key_mask = key_offsets < D_KEY
        matrix_mask = value_mask[:, None] & key_mask[None, :]
        grad_state = tl.zeros((BLOCK_DV, BLOCK_DK), dtype=tl.float32)

        for reverse_step in range(0, LENGTH):
            step = LENGTH - 1 - reverse_step
            key_base = k_ptr + batch_idx * k_stride_b + step * k_stride_l
            state_base = (
                states_ptr
                + batch_idx * states_stride_b
                + step * states_stride_l
            )

            key = tl.load(
                key_base + key_offsets * k_stride_d, mask=key_mask, other=0.0
            ).to(tl.float32)
            receptance = tl.load(
                r_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + key_offsets * k_stride_d,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            decay = tl.load(
                w_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + key_offsets * k_stride_d,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            state = tl.load(
                state_base
                + value_offsets[:, None] * states_stride_v
                + key_offsets[None, :] * states_stride_k,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            if step == 0:
                prev_state = tl.zeros((BLOCK_DV, BLOCK_DK), dtype=tl.float32)
            else:
                prev_state = tl.load(
                    state_base
                    - states_stride_l
                    + value_offsets[:, None] * states_stride_v
                    + key_offsets[None, :] * states_stride_k,
                    mask=matrix_mask,
                    other=0.0,
                ).to(tl.float32)
            value = tl.load(
                v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_offsets * v_stride_d,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            grad_output = tl.load(
                grad_out_ptr
                + batch_idx * grad_out_stride_b
                + step * grad_out_stride_l
                + value_offsets * grad_out_stride_d,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)

            grad_state = grad_state + grad_output[:, None] * receptance[None, :]
            grad_v = tl.sum(grad_state * key[None, :], axis=1)
            grad_key_base = (
                batch_idx * k_stride_b + step * k_stride_l + key_offsets * k_stride_d
            )
            tl.store(
                grad_v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_offsets * v_stride_d,
                grad_v,
                mask=value_mask,
            )
            tl.atomic_add(
                grad_k_ptr + grad_key_base,
                tl.sum(grad_state * value[:, None], axis=0),
                mask=key_mask,
            )
            tl.atomic_add(
                grad_r_ptr + grad_key_base,
                tl.sum(grad_output[:, None] * state, axis=0),
                mask=key_mask,
            )
            tl.atomic_add(
                grad_w_ptr + grad_key_base,
                tl.sum(grad_state * prev_state, axis=0),
                mask=key_mask,
            )
            grad_state = grad_state * decay[None, :]

    @triton.jit
    def _rwkv7_rank_backward_grouped_kernel(
        v_ptr,
        k_ptr,
        r_ptr,
        w_ptr,
        kappa_ptr,
        a_ptr,
        states_ptr,
        grad_out_ptr,
        grad_v_ptr,
        grad_k_ptr,
        grad_r_ptr,
        grad_w_ptr,
        grad_kappa_ptr,
        grad_a_ptr,
        v_stride_b: tl.constexpr,
        v_stride_l: tl.constexpr,
        v_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_l: tl.constexpr,
        k_stride_d: tl.constexpr,
        states_stride_b: tl.constexpr,
        states_stride_v: tl.constexpr,
        states_stride_l: tl.constexpr,
        states_stride_k: tl.constexpr,
        grad_out_stride_b: tl.constexpr,
        grad_out_stride_l: tl.constexpr,
        grad_out_stride_d: tl.constexpr,
        LENGTH: tl.constexpr,
        D_VALUE: tl.constexpr,
        D_KEY: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        BLOCK_DK: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        value_block = tl.program_id(1)
        value_offsets = value_block * BLOCK_DV + tl.arange(0, BLOCK_DV)
        key_offsets = tl.arange(0, BLOCK_DK)
        value_mask = value_offsets < D_VALUE
        key_mask = key_offsets < D_KEY
        matrix_mask = value_mask[:, None] & key_mask[None, :]
        grad_state = tl.zeros((BLOCK_DV, BLOCK_DK), dtype=tl.float32)

        for reverse_step in range(0, LENGTH):
            step = LENGTH - 1 - reverse_step
            key_base = k_ptr + batch_idx * k_stride_b + step * k_stride_l
            state_base = (
                states_ptr
                + batch_idx * states_stride_b
                + step * states_stride_l
            )

            key = tl.load(
                key_base + key_offsets * k_stride_d, mask=key_mask, other=0.0
            ).to(tl.float32)
            receptance = tl.load(
                r_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + key_offsets * k_stride_d,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            decay = tl.load(
                w_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + key_offsets * k_stride_d,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            kappa = tl.load(
                kappa_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + key_offsets * k_stride_d,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            alpha = tl.load(
                a_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + key_offsets * k_stride_d,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            state = tl.load(
                state_base
                + value_offsets[:, None] * states_stride_v
                + key_offsets[None, :] * states_stride_k,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            if step == 0:
                prev_state = tl.zeros((BLOCK_DV, BLOCK_DK), dtype=tl.float32)
            else:
                prev_state = tl.load(
                    state_base
                    - states_stride_l
                    + value_offsets[:, None] * states_stride_v
                    + key_offsets[None, :] * states_stride_k,
                    mask=matrix_mask,
                    other=0.0,
                ).to(tl.float32)
            value = tl.load(
                v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_offsets * v_stride_d,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            grad_output = tl.load(
                grad_out_ptr
                + batch_idx * grad_out_stride_b
                + step * grad_out_stride_l
                + value_offsets * grad_out_stride_d,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)

            grad_state = grad_state + grad_output[:, None] * receptance[None, :]
            grad_v = tl.sum(grad_state * key[None, :], axis=1)
            grad_key_base = (
                batch_idx * k_stride_b + step * k_stride_l + key_offsets * k_stride_d
            )
            state_before_rank = prev_state * decay[None, :]
            removal_key = alpha * kappa
            removed = tl.sum(state_before_rank * removal_key[None, :], axis=1)
            grad_removed = -tl.sum(grad_state * kappa[None, :], axis=1)
            grad_state_before_rank = (
                grad_state + grad_removed[:, None] * removal_key[None, :]
            )

            tl.store(
                grad_v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_offsets * v_stride_d,
                grad_v,
                mask=value_mask,
            )
            tl.atomic_add(
                grad_k_ptr + grad_key_base,
                tl.sum(grad_state * value[:, None], axis=0),
                mask=key_mask,
            )
            tl.atomic_add(
                grad_r_ptr + grad_key_base,
                tl.sum(grad_output[:, None] * state, axis=0),
                mask=key_mask,
            )
            tl.atomic_add(
                grad_w_ptr + grad_key_base,
                tl.sum(grad_state_before_rank * prev_state, axis=0),
                mask=key_mask,
            )
            tl.atomic_add(
                grad_kappa_ptr + grad_key_base,
                tl.sum(
                    -removed[:, None] * grad_state
                    + grad_removed[:, None] * state_before_rank * alpha[None, :],
                    axis=0,
                ),
                mask=key_mask,
            )
            tl.atomic_add(
                grad_a_ptr + grad_key_base,
                tl.sum(
                    grad_removed[:, None] * state_before_rank * kappa[None, :],
                    axis=0,
                ),
                mask=key_mask,
            )
            grad_state = grad_state_before_rank * decay[None, :]

    @triton.jit
    def _rwkv7_diag_backward_partial_kernel(
        v_ptr,
        k_ptr,
        r_ptr,
        w_ptr,
        states_ptr,
        grad_out_ptr,
        grad_v_ptr,
        partial_grad_k_ptr,
        partial_grad_r_ptr,
        partial_grad_w_ptr,
        v_stride_b: tl.constexpr,
        v_stride_l: tl.constexpr,
        v_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_l: tl.constexpr,
        k_stride_d: tl.constexpr,
        states_stride_b: tl.constexpr,
        states_stride_v: tl.constexpr,
        states_stride_l: tl.constexpr,
        states_stride_k: tl.constexpr,
        grad_out_stride_b: tl.constexpr,
        grad_out_stride_l: tl.constexpr,
        grad_out_stride_d: tl.constexpr,
        partial_stride_b: tl.constexpr,
        partial_stride_vb: tl.constexpr,
        partial_stride_l: tl.constexpr,
        partial_stride_k: tl.constexpr,
        LENGTH: tl.constexpr,
        D_VALUE: tl.constexpr,
        D_KEY: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        BLOCK_DK: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        value_block = tl.program_id(1)
        value_offsets = value_block * BLOCK_DV + tl.arange(0, BLOCK_DV)
        key_offsets = tl.arange(0, BLOCK_DK)
        value_mask = value_offsets < D_VALUE
        key_mask = key_offsets < D_KEY
        matrix_mask = value_mask[:, None] & key_mask[None, :]
        grad_state = tl.zeros((BLOCK_DV, BLOCK_DK), dtype=tl.float32)

        for reverse_step in range(0, LENGTH):
            step = LENGTH - 1 - reverse_step
            key_base = k_ptr + batch_idx * k_stride_b + step * k_stride_l
            state_base = (
                states_ptr
                + batch_idx * states_stride_b
                + step * states_stride_l
            )

            key = tl.load(
                key_base + key_offsets * k_stride_d, mask=key_mask, other=0.0
            ).to(tl.float32)
            receptance = tl.load(
                r_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + key_offsets * k_stride_d,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            decay = tl.load(
                w_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + key_offsets * k_stride_d,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            state = tl.load(
                state_base
                + value_offsets[:, None] * states_stride_v
                + key_offsets[None, :] * states_stride_k,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            if step == 0:
                prev_state = tl.zeros((BLOCK_DV, BLOCK_DK), dtype=tl.float32)
            else:
                prev_state = tl.load(
                    state_base
                    - states_stride_l
                    + value_offsets[:, None] * states_stride_v
                    + key_offsets[None, :] * states_stride_k,
                    mask=matrix_mask,
                    other=0.0,
                ).to(tl.float32)
            value = tl.load(
                v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_offsets * v_stride_d,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            grad_output = tl.load(
                grad_out_ptr
                + batch_idx * grad_out_stride_b
                + step * grad_out_stride_l
                + value_offsets * grad_out_stride_d,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)

            grad_state = grad_state + grad_output[:, None] * receptance[None, :]
            grad_v = tl.sum(grad_state * key[None, :], axis=1)
            partial_base = (
                batch_idx * partial_stride_b
                + value_block * partial_stride_vb
                + step * partial_stride_l
                + key_offsets * partial_stride_k
            )
            tl.store(
                grad_v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_offsets * v_stride_d,
                grad_v,
                mask=value_mask,
            )
            tl.store(
                partial_grad_k_ptr + partial_base,
                tl.sum(grad_state * value[:, None], axis=0),
                mask=key_mask,
            )
            tl.store(
                partial_grad_r_ptr + partial_base,
                tl.sum(grad_output[:, None] * state, axis=0),
                mask=key_mask,
            )
            tl.store(
                partial_grad_w_ptr + partial_base,
                tl.sum(grad_state * prev_state, axis=0),
                mask=key_mask,
            )
            grad_state = grad_state * decay[None, :]

    @triton.jit
    def _rwkv7_rank_backward_partial_kernel(
        v_ptr,
        k_ptr,
        r_ptr,
        w_ptr,
        kappa_ptr,
        a_ptr,
        states_ptr,
        grad_out_ptr,
        grad_v_ptr,
        partial_grad_k_ptr,
        partial_grad_r_ptr,
        partial_grad_w_ptr,
        partial_grad_kappa_ptr,
        partial_grad_a_ptr,
        v_stride_b: tl.constexpr,
        v_stride_l: tl.constexpr,
        v_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_l: tl.constexpr,
        k_stride_d: tl.constexpr,
        states_stride_b: tl.constexpr,
        states_stride_v: tl.constexpr,
        states_stride_l: tl.constexpr,
        states_stride_k: tl.constexpr,
        grad_out_stride_b: tl.constexpr,
        grad_out_stride_l: tl.constexpr,
        grad_out_stride_d: tl.constexpr,
        partial_stride_b: tl.constexpr,
        partial_stride_vb: tl.constexpr,
        partial_stride_l: tl.constexpr,
        partial_stride_k: tl.constexpr,
        LENGTH: tl.constexpr,
        D_VALUE: tl.constexpr,
        D_KEY: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        BLOCK_DK: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        value_block = tl.program_id(1)
        value_offsets = value_block * BLOCK_DV + tl.arange(0, BLOCK_DV)
        key_offsets = tl.arange(0, BLOCK_DK)
        value_mask = value_offsets < D_VALUE
        key_mask = key_offsets < D_KEY
        matrix_mask = value_mask[:, None] & key_mask[None, :]
        grad_state = tl.zeros((BLOCK_DV, BLOCK_DK), dtype=tl.float32)

        for reverse_step in range(0, LENGTH):
            step = LENGTH - 1 - reverse_step
            key_base = k_ptr + batch_idx * k_stride_b + step * k_stride_l
            state_base = (
                states_ptr
                + batch_idx * states_stride_b
                + step * states_stride_l
            )

            key = tl.load(
                key_base + key_offsets * k_stride_d, mask=key_mask, other=0.0
            ).to(tl.float32)
            receptance = tl.load(
                r_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + key_offsets * k_stride_d,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            decay = tl.load(
                w_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + key_offsets * k_stride_d,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            kappa = tl.load(
                kappa_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + key_offsets * k_stride_d,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            alpha = tl.load(
                a_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + key_offsets * k_stride_d,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            state = tl.load(
                state_base
                + value_offsets[:, None] * states_stride_v
                + key_offsets[None, :] * states_stride_k,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            if step == 0:
                prev_state = tl.zeros((BLOCK_DV, BLOCK_DK), dtype=tl.float32)
            else:
                prev_state = tl.load(
                    state_base
                    - states_stride_l
                    + value_offsets[:, None] * states_stride_v
                    + key_offsets[None, :] * states_stride_k,
                    mask=matrix_mask,
                    other=0.0,
                ).to(tl.float32)
            value = tl.load(
                v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_offsets * v_stride_d,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            grad_output = tl.load(
                grad_out_ptr
                + batch_idx * grad_out_stride_b
                + step * grad_out_stride_l
                + value_offsets * grad_out_stride_d,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)

            grad_state = grad_state + grad_output[:, None] * receptance[None, :]
            grad_v = tl.sum(grad_state * key[None, :], axis=1)
            state_before_rank = prev_state * decay[None, :]
            removal_key = alpha * kappa
            removed = tl.sum(state_before_rank * removal_key[None, :], axis=1)
            grad_removed = -tl.sum(grad_state * kappa[None, :], axis=1)
            grad_state_before_rank = (
                grad_state + grad_removed[:, None] * removal_key[None, :]
            )
            partial_base = (
                batch_idx * partial_stride_b
                + value_block * partial_stride_vb
                + step * partial_stride_l
                + key_offsets * partial_stride_k
            )

            tl.store(
                grad_v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_offsets * v_stride_d,
                grad_v,
                mask=value_mask,
            )
            tl.store(
                partial_grad_k_ptr + partial_base,
                tl.sum(grad_state * value[:, None], axis=0),
                mask=key_mask,
            )
            tl.store(
                partial_grad_r_ptr + partial_base,
                tl.sum(grad_output[:, None] * state, axis=0),
                mask=key_mask,
            )
            tl.store(
                partial_grad_w_ptr + partial_base,
                tl.sum(grad_state_before_rank * prev_state, axis=0),
                mask=key_mask,
            )
            tl.store(
                partial_grad_kappa_ptr + partial_base,
                tl.sum(
                    -removed[:, None] * grad_state
                    + grad_removed[:, None] * state_before_rank * alpha[None, :],
                    axis=0,
                ),
                mask=key_mask,
            )
            tl.store(
                partial_grad_a_ptr + partial_base,
                tl.sum(
                    grad_removed[:, None] * state_before_rank * kappa[None, :],
                    axis=0,
                ),
                mask=key_mask,
            )
            grad_state = grad_state_before_rank * decay[None, :]

    @triton.jit
    def _rwkv7_partial_reduce_kernel(
        partial_grad_ptr,
        grad_ptr,
        partial_stride_b: tl.constexpr,
        partial_stride_vb: tl.constexpr,
        partial_stride_l: tl.constexpr,
        partial_stride_k: tl.constexpr,
        grad_stride_b: tl.constexpr,
        grad_stride_l: tl.constexpr,
        grad_stride_d: tl.constexpr,
        VALUE_BLOCKS: tl.constexpr,
        D_KEY: tl.constexpr,
        BLOCK_VB: tl.constexpr,
        BLOCK_DK: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        step = tl.program_id(1)
        value_block_offsets = tl.arange(0, BLOCK_VB)
        key_offsets = tl.arange(0, BLOCK_DK)
        mask = (value_block_offsets[:, None] < VALUE_BLOCKS) & (
            key_offsets[None, :] < D_KEY
        )
        partial = tl.load(
            partial_grad_ptr
            + batch_idx * partial_stride_b
            + value_block_offsets[:, None] * partial_stride_vb
            + step * partial_stride_l
            + key_offsets[None, :] * partial_stride_k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        reduced = tl.sum(partial, axis=0)
        tl.store(
            grad_ptr
            + batch_idx * grad_stride_b
            + step * grad_stride_l
            + key_offsets * grad_stride_d,
            reduced,
            mask=key_offsets < D_KEY,
        )

    @triton.jit
    def _rwkv7_diag_backward_recompute_kernel(
        v_ptr,
        k_ptr,
        r_ptr,
        w_ptr,
        grad_out_ptr,
        checkpoints_ptr,
        grad_v_ptr,
        grad_k_ptr,
        grad_r_ptr,
        grad_w_ptr,
        v_stride_b: tl.constexpr,
        v_stride_l: tl.constexpr,
        v_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_l: tl.constexpr,
        k_stride_d: tl.constexpr,
        grad_out_stride_b: tl.constexpr,
        grad_out_stride_l: tl.constexpr,
        grad_out_stride_d: tl.constexpr,
        checkpoints_stride_b: tl.constexpr,
        checkpoints_stride_v: tl.constexpr,
        checkpoints_stride_c: tl.constexpr,
        checkpoints_stride_k: tl.constexpr,
        LENGTH: tl.constexpr,
        D_KEY: tl.constexpr,
        BLOCK_DK: tl.constexpr,
        CHECKPOINT_INTERVAL: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        value_idx = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_DK)
        mask = offsets < D_KEY
        grad_state = tl.zeros((BLOCK_DK,), dtype=tl.float32)

        for reverse_step in range(0, LENGTH):
            step = LENGTH - 1 - reverse_step
            checkpoint_idx = step // CHECKPOINT_INTERVAL
            checkpoint_step = checkpoint_idx * CHECKPOINT_INTERVAL
            checkpoint_base = (
                checkpoints_ptr
                + batch_idx * checkpoints_stride_b
                + value_idx * checkpoints_stride_v
                + checkpoint_idx * checkpoints_stride_c
            )
            replay_state = tl.load(
                checkpoint_base + offsets * checkpoints_stride_k,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            prev_state = replay_state
            state = replay_state
            for replay_offset in range(0, CHECKPOINT_INTERVAL):
                replay_step = checkpoint_step + replay_offset
                if replay_step <= step:
                    replay_key_base = (
                        k_ptr + batch_idx * k_stride_b + replay_step * k_stride_l
                    )
                    replay_key = tl.load(
                        replay_key_base + offsets * k_stride_d,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                    replay_decay = tl.load(
                        w_ptr
                        + batch_idx * k_stride_b
                        + replay_step * k_stride_l
                        + offsets * k_stride_d,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                    replay_value = tl.load(
                        v_ptr
                        + batch_idx * v_stride_b
                        + replay_step * v_stride_l
                        + value_idx * v_stride_d
                    ).to(tl.float32)
                    if replay_step == step:
                        prev_state = replay_state
                    replay_state = (
                        replay_state * replay_decay + replay_value * replay_key
                    )
                    if replay_step == step:
                        state = replay_state

            key_base = k_ptr + batch_idx * k_stride_b + step * k_stride_l
            key = tl.load(key_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )
            receptance = tl.load(
                r_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + offsets * k_stride_d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            decay = tl.load(
                w_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + offsets * k_stride_d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_idx * v_stride_d
            ).to(tl.float32)
            grad_output = tl.load(
                grad_out_ptr
                + batch_idx * grad_out_stride_b
                + step * grad_out_stride_l
                + value_idx * grad_out_stride_d
            ).to(tl.float32)

            grad_state = grad_state + grad_output * receptance
            grad_v = tl.sum(grad_state * key, axis=0)
            grad_key_base = (
                batch_idx * k_stride_b + step * k_stride_l + offsets * k_stride_d
            )
            tl.store(
                grad_v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_idx * v_stride_d,
                grad_v,
            )
            tl.atomic_add(
                grad_k_ptr + grad_key_base,
                grad_state * value,
                mask=mask,
            )
            tl.atomic_add(
                grad_r_ptr + grad_key_base,
                grad_output * state,
                mask=mask,
            )
            tl.atomic_add(
                grad_w_ptr + grad_key_base,
                grad_state * prev_state,
                mask=mask,
            )
            grad_state = grad_state * decay

    @triton.jit
    def _rwkv7_rank_backward_recompute_kernel(
        v_ptr,
        k_ptr,
        r_ptr,
        w_ptr,
        kappa_ptr,
        a_ptr,
        grad_out_ptr,
        checkpoints_ptr,
        grad_v_ptr,
        grad_k_ptr,
        grad_r_ptr,
        grad_w_ptr,
        grad_kappa_ptr,
        grad_a_ptr,
        v_stride_b: tl.constexpr,
        v_stride_l: tl.constexpr,
        v_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_l: tl.constexpr,
        k_stride_d: tl.constexpr,
        grad_out_stride_b: tl.constexpr,
        grad_out_stride_l: tl.constexpr,
        grad_out_stride_d: tl.constexpr,
        checkpoints_stride_b: tl.constexpr,
        checkpoints_stride_v: tl.constexpr,
        checkpoints_stride_c: tl.constexpr,
        checkpoints_stride_k: tl.constexpr,
        LENGTH: tl.constexpr,
        D_KEY: tl.constexpr,
        BLOCK_DK: tl.constexpr,
        CHECKPOINT_INTERVAL: tl.constexpr,
    ) -> None:
        batch_idx = tl.program_id(0)
        value_idx = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_DK)
        mask = offsets < D_KEY
        grad_state = tl.zeros((BLOCK_DK,), dtype=tl.float32)

        for reverse_step in range(0, LENGTH):
            step = LENGTH - 1 - reverse_step
            checkpoint_idx = step // CHECKPOINT_INTERVAL
            checkpoint_step = checkpoint_idx * CHECKPOINT_INTERVAL
            checkpoint_base = (
                checkpoints_ptr
                + batch_idx * checkpoints_stride_b
                + value_idx * checkpoints_stride_v
                + checkpoint_idx * checkpoints_stride_c
            )
            replay_state = tl.load(
                checkpoint_base + offsets * checkpoints_stride_k,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            prev_state = replay_state
            state = replay_state
            for replay_offset in range(0, CHECKPOINT_INTERVAL):
                replay_step = checkpoint_step + replay_offset
                if replay_step <= step:
                    replay_key_base = (
                        k_ptr + batch_idx * k_stride_b + replay_step * k_stride_l
                    )
                    replay_key = tl.load(
                        replay_key_base + offsets * k_stride_d,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                    replay_decay = tl.load(
                        w_ptr
                        + batch_idx * k_stride_b
                        + replay_step * k_stride_l
                        + offsets * k_stride_d,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                    replay_kappa = tl.load(
                        kappa_ptr
                        + batch_idx * k_stride_b
                        + replay_step * k_stride_l
                        + offsets * k_stride_d,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                    replay_alpha = tl.load(
                        a_ptr
                        + batch_idx * k_stride_b
                        + replay_step * k_stride_l
                        + offsets * k_stride_d,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                    replay_value = tl.load(
                        v_ptr
                        + batch_idx * v_stride_b
                        + replay_step * v_stride_l
                        + value_idx * v_stride_d
                    ).to(tl.float32)
                    if replay_step == step:
                        prev_state = replay_state
                    replay_state = replay_state * replay_decay
                    replay_removal_key = replay_alpha * replay_kappa
                    replay_removed = tl.sum(replay_state * replay_removal_key, axis=0)
                    replay_state = (
                        replay_state
                        - replay_removed * replay_kappa
                        + replay_value * replay_key
                    )
                    if replay_step == step:
                        state = replay_state

            key_base = k_ptr + batch_idx * k_stride_b + step * k_stride_l
            key = tl.load(key_base + offsets * k_stride_d, mask=mask, other=0.0).to(
                tl.float32
            )
            receptance = tl.load(
                r_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + offsets * k_stride_d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            decay = tl.load(
                w_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + offsets * k_stride_d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            kappa = tl.load(
                kappa_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + offsets * k_stride_d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            alpha = tl.load(
                a_ptr
                + batch_idx * k_stride_b
                + step * k_stride_l
                + offsets * k_stride_d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_idx * v_stride_d
            ).to(tl.float32)
            grad_output = tl.load(
                grad_out_ptr
                + batch_idx * grad_out_stride_b
                + step * grad_out_stride_l
                + value_idx * grad_out_stride_d
            ).to(tl.float32)

            grad_state = grad_state + grad_output * receptance
            grad_v = tl.sum(grad_state * key, axis=0)
            grad_key_base = (
                batch_idx * k_stride_b + step * k_stride_l + offsets * k_stride_d
            )
            state_before_rank = prev_state * decay
            removal_key = alpha * kappa
            removed = tl.sum(state_before_rank * removal_key, axis=0)
            grad_removed = -tl.sum(grad_state * kappa, axis=0)
            grad_state_before_rank = grad_state + grad_removed * removal_key

            tl.store(
                grad_v_ptr
                + batch_idx * v_stride_b
                + step * v_stride_l
                + value_idx * v_stride_d,
                grad_v,
            )
            tl.atomic_add(
                grad_k_ptr + grad_key_base,
                grad_state * value,
                mask=mask,
            )
            tl.atomic_add(
                grad_r_ptr + grad_key_base,
                grad_output * state,
                mask=mask,
            )
            tl.atomic_add(
                grad_w_ptr + grad_key_base,
                grad_state_before_rank * prev_state,
                mask=mask,
            )
            tl.atomic_add(
                grad_kappa_ptr + grad_key_base,
                -removed * grad_state + grad_removed * state_before_rank * alpha,
                mask=mask,
            )
            tl.atomic_add(
                grad_a_ptr + grad_key_base,
                grad_removed * state_before_rank * kappa,
                mask=mask,
            )
            grad_state = grad_state_before_rank * decay


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _env_int(name: str, allowed: set[int] | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if allowed is not None and value not in allowed:
        allowed_values = ", ".join(str(item) for item in sorted(allowed))
        raise ValueError(f"{name} must be one of: {allowed_values}")
    return value


def _env_choice(name: str, allowed: set[str]) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {allowed_values}")
    return value


def _kernel_num_warps(block_dk: int) -> int:
    override = _env_int("DUALGOOSE_TRITON_NUM_WARPS", {1, 2, 4, 8})
    if override is not None:
        return override
    if block_dk <= 32:
        return 1
    if block_dk <= 64:
        return 2
    return 4


def _kernel_num_stages(block_dk: int) -> int:
    del block_dk
    override = _env_int("DUALGOOSE_TRITON_NUM_STAGES")
    if override is not None:
        return override
    return 1


def _backward_block_dv() -> int:
    return _env_int("DUALGOOSE_TRITON_BACKWARD_BLOCK_DV", {1, 2, 4, 8}) or 1


def _backward_mode() -> str:
    return _env_choice("DUALGOOSE_TRITON_BACKWARD_MODE", {"atomic", "two_pass"}) or "atomic"


def _recompute_checkpoint_interval() -> int:
    return _env_int("DUALGOOSE_TRITON_RECOMPUTE_CHECKPOINT_INTERVAL") or 64


def triton_kernel_config(d_key: int) -> dict[str, object]:
    block_dk = _next_power_of_2(d_key)
    return {
        "block_dk": block_dk,
        "num_warps": _kernel_num_warps(block_dk),
        "num_stages": _kernel_num_stages(block_dk),
        "backward_block_dv": _backward_block_dv(),
        "backward_mode": _backward_mode(),
        "recompute_checkpoint_interval": _recompute_checkpoint_interval(),
    }


def _rwkv7_scan_triton_forward(
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
    save_internal_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if not triton_available():
        raise RuntimeError("Triton is not installed")
    if not v.is_cuda:
        raise RuntimeError("rwkv7_scan_triton requires CUDA tensors")
    if initial_state is not None:
        raise NotImplementedError("rwkv7_scan_triton does not support initial_state")
    if return_states:
        raise NotImplementedError("rwkv7_scan_triton does not return states")
    if v.ndim != 3 or k.ndim != 3:
        raise ValueError("all inputs must have shape (batch, length, dim)")
    if not (k.shape == r.shape == w.shape == kappa.shape == a.shape):
        raise ValueError("key-axis tensors must have matching shapes")
    if v.shape[:2] != k.shape[:2]:
        raise ValueError("value and key tensors must share batch/length")
    if not all(tensor.is_cuda for tensor in (k, r, w, kappa, a)):
        raise RuntimeError("rwkv7_scan_triton requires all tensors on CUDA")
    if not all(tensor.dtype == v.dtype for tensor in (k, r, w, kappa, a)):
        raise ValueError("rwkv7_scan_triton requires matching dtypes")

    v = v.contiguous()
    k = k.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    kappa = kappa.contiguous()
    a = a.contiguous()
    batch, length, d_value = v.shape
    d_key = k.shape[-1]
    block_dk = _next_power_of_2(d_key)
    output = torch.empty_like(v)
    states = (
        torch.empty(
            batch,
            d_value,
            length,
            d_key,
            device=v.device,
            dtype=v.dtype,
        )
        if save_internal_states
        else torch.empty(1, device=v.device, dtype=v.dtype)
    )
    _rwkv7_scan_kernel[(batch, d_value)](
        v,
        k,
        r,
        w,
        kappa,
        a,
        output,
        states,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        states.stride(0),
        states.stride(1) if save_internal_states else 0,
        states.stride(2) if save_internal_states else 0,
        states.stride(3) if save_internal_states else 0,
        LENGTH=length,
        D_KEY=d_key,
        BLOCK_DK=block_dk,
        USE_RANK=use_rank,
        SAVE_STATES=save_internal_states,
        num_warps=_kernel_num_warps(block_dk),
        num_stages=_kernel_num_stages(block_dk),
    )
    if save_internal_states:
        return output, states
    return output


def _rwkv7_scan_triton_checkpoint_forward(
    v: torch.Tensor,
    k: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    kappa: torch.Tensor,
    a: torch.Tensor,
    *,
    use_rank: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not triton_available():
        raise RuntimeError("Triton is not installed")
    if not v.is_cuda:
        raise RuntimeError("rwkv7_scan_triton requires CUDA tensors")
    if v.ndim != 3 or k.ndim != 3:
        raise ValueError("all inputs must have shape (batch, length, dim)")
    if not (k.shape == r.shape == w.shape == kappa.shape == a.shape):
        raise ValueError("key-axis tensors must have matching shapes")
    if v.shape[:2] != k.shape[:2]:
        raise ValueError("value and key tensors must share batch/length")
    if not all(tensor.is_cuda for tensor in (k, r, w, kappa, a)):
        raise RuntimeError("rwkv7_scan_triton requires all tensors on CUDA")
    if not all(tensor.dtype == v.dtype for tensor in (k, r, w, kappa, a)):
        raise ValueError("rwkv7_scan_triton requires matching dtypes")

    v = v.contiguous()
    k = k.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    kappa = kappa.contiguous()
    a = a.contiguous()
    batch, length, d_value = v.shape
    if length <= 0:
        raise ValueError("rwkv7_scan_triton requires a positive sequence length")
    d_key = k.shape[-1]
    block_dk = _next_power_of_2(d_key)
    checkpoint_interval = min(_recompute_checkpoint_interval(), length)
    checkpoint_count = (length + checkpoint_interval - 1) // checkpoint_interval + 1
    output = torch.empty_like(v)
    checkpoints = torch.empty(
        batch,
        d_value,
        checkpoint_count,
        d_key,
        device=v.device,
        dtype=v.dtype,
    )
    _rwkv7_scan_checkpoint_kernel[(batch, d_value)](
        v,
        k,
        r,
        w,
        kappa,
        a,
        output,
        checkpoints,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        checkpoints.stride(0),
        checkpoints.stride(1),
        checkpoints.stride(2),
        checkpoints.stride(3),
        LENGTH=length,
        D_KEY=d_key,
        BLOCK_DK=block_dk,
        USE_RANK=use_rank,
        CHECKPOINT_INTERVAL=checkpoint_interval,
        num_warps=_kernel_num_warps(block_dk),
        num_stages=_kernel_num_stages(block_dk),
    )
    return output, checkpoints, checkpoint_interval


def _reduce_partial_grad_triton(partial_grad: torch.Tensor) -> torch.Tensor:
    batch, value_blocks, length, d_key = partial_grad.shape
    block_vb = _next_power_of_2(value_blocks)
    block_dk = _next_power_of_2(d_key)
    grad = torch.empty(
        batch,
        length,
        d_key,
        device=partial_grad.device,
        dtype=partial_grad.dtype,
    )
    _rwkv7_partial_reduce_kernel[(batch, length)](
        partial_grad,
        grad,
        partial_grad.stride(0),
        partial_grad.stride(1),
        partial_grad.stride(2),
        partial_grad.stride(3),
        grad.stride(0),
        grad.stride(1),
        grad.stride(2),
        VALUE_BLOCKS=value_blocks,
        D_KEY=d_key,
        BLOCK_VB=block_vb,
        BLOCK_DK=block_dk,
        num_warps=_kernel_num_warps(block_dk),
        num_stages=_kernel_num_stages(block_dk),
    )
    return grad


def _rwkv7_diag_backward_triton(
    v: torch.Tensor,
    k: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    states: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not triton_available():
        raise RuntimeError("Triton is not installed")
    v = v.contiguous()
    k = k.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    states = states.contiguous()
    grad_output = grad_output.contiguous()
    batch, length, d_value = v.shape
    d_key = k.shape[-1]
    block_dk = _next_power_of_2(d_key)
    block_dv = _backward_block_dv()
    value_blocks = (d_value + block_dv - 1) // block_dv
    grad_v = torch.empty_like(v)
    if _backward_mode() == "two_pass":
        partial_shape = (batch, value_blocks, length, d_key)
        partial_grad_k = torch.empty(partial_shape, device=k.device, dtype=k.dtype)
        partial_grad_r = torch.empty(partial_shape, device=r.device, dtype=r.dtype)
        partial_grad_w = torch.empty(partial_shape, device=w.device, dtype=w.dtype)
        _rwkv7_diag_backward_partial_kernel[(batch, value_blocks)](
            v,
            k,
            r,
            w,
            states,
            grad_output,
            grad_v,
            partial_grad_k,
            partial_grad_r,
            partial_grad_w,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            partial_grad_k.stride(0),
            partial_grad_k.stride(1),
            partial_grad_k.stride(2),
            partial_grad_k.stride(3),
            LENGTH=length,
            D_VALUE=d_value,
            D_KEY=d_key,
            BLOCK_DV=block_dv,
            BLOCK_DK=block_dk,
            num_warps=_kernel_num_warps(block_dk),
            num_stages=_kernel_num_stages(block_dk),
        )
        return (
            grad_v,
            _reduce_partial_grad_triton(partial_grad_k),
            _reduce_partial_grad_triton(partial_grad_r),
            _reduce_partial_grad_triton(partial_grad_w),
        )
    grad_k = torch.zeros_like(k)
    grad_r = torch.zeros_like(r)
    grad_w = torch.zeros_like(w)
    _rwkv7_diag_backward_grouped_kernel[(batch, value_blocks)](
        v,
        k,
        r,
        w,
        states,
        grad_output,
        grad_v,
        grad_k,
        grad_r,
        grad_w,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        LENGTH=length,
        D_VALUE=d_value,
        D_KEY=d_key,
        BLOCK_DV=block_dv,
        BLOCK_DK=block_dk,
        num_warps=_kernel_num_warps(block_dk),
        num_stages=_kernel_num_stages(block_dk),
    )
    return grad_v, grad_k, grad_r, grad_w


def _rwkv7_rank_backward_triton(
    v: torch.Tensor,
    k: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    kappa: torch.Tensor,
    a: torch.Tensor,
    states: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if not triton_available():
        raise RuntimeError("Triton is not installed")
    v = v.contiguous()
    k = k.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    kappa = kappa.contiguous()
    a = a.contiguous()
    states = states.contiguous()
    grad_output = grad_output.contiguous()
    batch, length, d_value = v.shape
    d_key = k.shape[-1]
    block_dk = _next_power_of_2(d_key)
    block_dv = _backward_block_dv()
    value_blocks = (d_value + block_dv - 1) // block_dv
    grad_v = torch.empty_like(v)
    if _backward_mode() == "two_pass":
        partial_shape = (batch, value_blocks, length, d_key)
        partial_grad_k = torch.empty(partial_shape, device=k.device, dtype=k.dtype)
        partial_grad_r = torch.empty(partial_shape, device=r.device, dtype=r.dtype)
        partial_grad_w = torch.empty(partial_shape, device=w.device, dtype=w.dtype)
        partial_grad_kappa = torch.empty(
            partial_shape, device=kappa.device, dtype=kappa.dtype
        )
        partial_grad_a = torch.empty(partial_shape, device=a.device, dtype=a.dtype)
        _rwkv7_rank_backward_partial_kernel[(batch, value_blocks)](
            v,
            k,
            r,
            w,
            kappa,
            a,
            states,
            grad_output,
            grad_v,
            partial_grad_k,
            partial_grad_r,
            partial_grad_w,
            partial_grad_kappa,
            partial_grad_a,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            partial_grad_k.stride(0),
            partial_grad_k.stride(1),
            partial_grad_k.stride(2),
            partial_grad_k.stride(3),
            LENGTH=length,
            D_VALUE=d_value,
            D_KEY=d_key,
            BLOCK_DV=block_dv,
            BLOCK_DK=block_dk,
            num_warps=_kernel_num_warps(block_dk),
            num_stages=_kernel_num_stages(block_dk),
        )
        return (
            grad_v,
            _reduce_partial_grad_triton(partial_grad_k),
            _reduce_partial_grad_triton(partial_grad_r),
            _reduce_partial_grad_triton(partial_grad_w),
            _reduce_partial_grad_triton(partial_grad_kappa),
            _reduce_partial_grad_triton(partial_grad_a),
        )
    grad_k = torch.zeros_like(k)
    grad_r = torch.zeros_like(r)
    grad_w = torch.zeros_like(w)
    grad_kappa = torch.zeros_like(kappa)
    grad_a = torch.zeros_like(a)
    _rwkv7_rank_backward_grouped_kernel[(batch, value_blocks)](
        v,
        k,
        r,
        w,
        kappa,
        a,
        states,
        grad_output,
        grad_v,
        grad_k,
        grad_r,
        grad_w,
        grad_kappa,
        grad_a,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        LENGTH=length,
        D_VALUE=d_value,
        D_KEY=d_key,
        BLOCK_DV=block_dv,
        BLOCK_DK=block_dk,
        num_warps=_kernel_num_warps(block_dk),
        num_stages=_kernel_num_stages(block_dk),
    )
    return (
        grad_v,
        grad_k,
        grad_r,
        grad_w,
        grad_kappa,
        grad_a,
    )


def _rwkv7_diag_backward_recompute_triton(
    v: torch.Tensor,
    k: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    checkpoints: torch.Tensor,
    grad_output: torch.Tensor,
    checkpoint_interval: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not triton_available():
        raise RuntimeError("Triton is not installed")
    v = v.contiguous()
    k = k.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    checkpoints = checkpoints.contiguous()
    grad_output = grad_output.contiguous()
    batch, length, d_value = v.shape
    d_key = k.shape[-1]
    block_dk = _next_power_of_2(d_key)
    grad_v = torch.empty_like(v)
    grad_k = torch.zeros_like(k)
    grad_r = torch.zeros_like(r)
    grad_w = torch.zeros_like(w)
    _rwkv7_diag_backward_recompute_kernel[(batch, d_value)](
        v,
        k,
        r,
        w,
        grad_output,
        checkpoints,
        grad_v,
        grad_k,
        grad_r,
        grad_w,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        checkpoints.stride(0),
        checkpoints.stride(1),
        checkpoints.stride(2),
        checkpoints.stride(3),
        LENGTH=length,
        D_KEY=d_key,
        BLOCK_DK=block_dk,
        CHECKPOINT_INTERVAL=checkpoint_interval,
        num_warps=_kernel_num_warps(block_dk),
        num_stages=_kernel_num_stages(block_dk),
    )
    return grad_v, grad_k, grad_r, grad_w


def _rwkv7_rank_backward_recompute_triton(
    v: torch.Tensor,
    k: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    kappa: torch.Tensor,
    a: torch.Tensor,
    checkpoints: torch.Tensor,
    grad_output: torch.Tensor,
    checkpoint_interval: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if not triton_available():
        raise RuntimeError("Triton is not installed")
    v = v.contiguous()
    k = k.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    kappa = kappa.contiguous()
    a = a.contiguous()
    checkpoints = checkpoints.contiguous()
    grad_output = grad_output.contiguous()
    batch, length, d_value = v.shape
    d_key = k.shape[-1]
    block_dk = _next_power_of_2(d_key)
    grad_v = torch.empty_like(v)
    grad_k = torch.zeros_like(k)
    grad_r = torch.zeros_like(r)
    grad_w = torch.zeros_like(w)
    grad_kappa = torch.zeros_like(kappa)
    grad_a = torch.zeros_like(a)
    _rwkv7_rank_backward_recompute_kernel[(batch, d_value)](
        v,
        k,
        r,
        w,
        kappa,
        a,
        grad_output,
        checkpoints,
        grad_v,
        grad_k,
        grad_r,
        grad_w,
        grad_kappa,
        grad_a,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        checkpoints.stride(0),
        checkpoints.stride(1),
        checkpoints.stride(2),
        checkpoints.stride(3),
        LENGTH=length,
        D_KEY=d_key,
        BLOCK_DK=block_dk,
        CHECKPOINT_INTERVAL=checkpoint_interval,
        num_warps=_kernel_num_warps(block_dk),
        num_stages=_kernel_num_stages(block_dk),
    )
    return (
        grad_v,
        grad_k,
        grad_r,
        grad_w,
        grad_kappa,
        grad_a,
    )


class _RWKV7TritonScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v: torch.Tensor,
        k: torch.Tensor,
        r: torch.Tensor,
        w: torch.Tensor,
        kappa: torch.Tensor,
        a: torch.Tensor,
        use_rank: bool,
    ) -> torch.Tensor:
        ctx.use_rank = use_rank
        output, states = _rwkv7_scan_triton_forward(
            v,
            k,
            r,
            w,
            kappa,
            a,
            use_rank=use_rank,
            save_internal_states=True,
        )
        if use_rank:
            ctx.save_for_backward(v, k, r, w, kappa, a, states)
        else:
            ctx.save_for_backward(v, k, r, w, states)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        if ctx.use_rank:
            v, k, r, w, kappa, a, states = saved
            return (
                *_rwkv7_rank_backward_triton(
                    v, k, r, w, kappa, a, states, grad_output
                ),
                None,
            )
        if not ctx.use_rank:
            v, k, r, w, states = saved
            grad_v, grad_k, grad_r, grad_w = _rwkv7_diag_backward_triton(
                v, k, r, w, states, grad_output
            )
            return grad_v, grad_k, grad_r, grad_w, None, None, None
        raise AssertionError("unreachable scan backend state")


class _RWKV7TritonRecomputeScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v: torch.Tensor,
        k: torch.Tensor,
        r: torch.Tensor,
        w: torch.Tensor,
        kappa: torch.Tensor,
        a: torch.Tensor,
        use_rank: bool,
    ) -> torch.Tensor:
        ctx.use_rank = use_rank
        output, checkpoints, checkpoint_interval = _rwkv7_scan_triton_checkpoint_forward(
            v,
            k,
            r,
            w,
            kappa,
            a,
            use_rank=use_rank,
        )
        ctx.recompute_checkpoint_interval = checkpoint_interval
        if use_rank:
            ctx.save_for_backward(v, k, r, w, kappa, a, checkpoints)
        else:
            ctx.save_for_backward(v, k, r, w, checkpoints)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.use_rank:
            v, k, r, w, kappa, a, checkpoints = ctx.saved_tensors
            return (
                *_rwkv7_rank_backward_recompute_triton(
                    v,
                    k,
                    r,
                    w,
                    kappa,
                    a,
                    checkpoints,
                    grad_output,
                    ctx.recompute_checkpoint_interval,
                ),
                None,
            )
        v, k, r, w, checkpoints = ctx.saved_tensors
        grad_v, grad_k, grad_r, grad_w = _rwkv7_diag_backward_recompute_triton(
            v, k, r, w, checkpoints, grad_output, ctx.recompute_checkpoint_interval
        )
        return grad_v, grad_k, grad_r, grad_w, None, None, None


def rwkv7_scan_triton(
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
) -> torch.Tensor:
    """Run the RWKV7 scan with Triton forward and backward kernels."""

    if initial_state is not None:
        raise NotImplementedError("rwkv7_scan_triton does not support initial_state")
    if return_states:
        raise NotImplementedError("rwkv7_scan_triton does not return states")
    needs_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (v, k, r, w, kappa, a)
    )
    if not needs_grad:
        return _rwkv7_scan_triton_forward(v, k, r, w, kappa, a, use_rank=use_rank)
    return _RWKV7TritonScan.apply(v, k, r, w, kappa, a, use_rank)


def rwkv7_scan_triton_recompute(
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
) -> torch.Tensor:
    """Run Triton forward with checkpointed Triton recomputation in backward.

    This backend avoids saving the full `(batch, value, length, key)` state
    tensor from the forward pass. Backward stores sparse recurrent-state
    checkpoints and replays at most one checkpoint chunk per backward step,
    providing a bounded saved-state-memory tradeoff for validation and small
    training runs.
    """

    if initial_state is not None:
        raise NotImplementedError(
            "rwkv7_scan_triton_recompute does not support initial_state"
        )
    if return_states:
        raise NotImplementedError("rwkv7_scan_triton_recompute does not return states")
    needs_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (v, k, r, w, kappa, a)
    )
    if not needs_grad:
        return _rwkv7_scan_triton_forward(v, k, r, w, kappa, a, use_rank=use_rank)
    return _RWKV7TritonRecomputeScan.apply(v, k, r, w, kappa, a, use_rank)
