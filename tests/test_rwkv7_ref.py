import torch
import pytest

from dualgoose.rwkv7_backends import make_scan_fn
from dualgoose.rwkv7_ref import diagonal_scan, reverse_rwkv7_scan, rwkv7_scan
from dualgoose.rwkv7_triton import triton_available, triton_kernel_config


def _inputs():
    torch.manual_seed(0)
    batch, length, d_value, d_key = 2, 5, 3, 4
    v = torch.randn(batch, length, d_value)
    k = torch.randn(batch, length, d_key).tanh()
    r = torch.randn(batch, length, d_key).tanh()
    w = torch.rand(batch, length, d_key)
    kappa = torch.randn(batch, length, d_key).tanh()
    a = torch.rand(batch, length, d_key)
    return v, k, r, w, kappa, a


def test_scan_shapes_and_state_orientation():
    out, states = rwkv7_scan(*_inputs(), return_states=True)
    assert out.shape == (2, 5, 3)
    assert states.shape == (2, 5, 3, 4)


def test_rank_disabled_matches_diagonal_reference():
    v, k, r, w, kappa, a = _inputs()
    rank_off = rwkv7_scan(v, k, r, w, kappa, a, use_rank=False)
    diagonal = diagonal_scan(v, k, r, w)
    assert torch.allclose(rank_off, diagonal)


def test_reference_backend_matches_direct_scan():
    v, k, r, w, kappa, a = _inputs()
    backend_scan = make_scan_fn("reference")
    assert torch.allclose(
        backend_scan(v, k, r, w, kappa, a),
        rwkv7_scan(v, k, r, w, kappa, a),
    )


def test_unknown_scan_backend_is_rejected():
    try:
        make_scan_fn("unknown")
    except ValueError as exc:
        assert "scan_backend" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_triton_kernel_config_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("DUALGOOSE_TRITON_NUM_WARPS", raising=False)
    monkeypatch.delenv("DUALGOOSE_TRITON_NUM_STAGES", raising=False)
    monkeypatch.delenv("DUALGOOSE_TRITON_BACKWARD_BLOCK_DV", raising=False)
    monkeypatch.delenv("DUALGOOSE_TRITON_BACKWARD_MODE", raising=False)
    monkeypatch.delenv("DUALGOOSE_TRITON_RECOMPUTE_CHECKPOINT_INTERVAL", raising=False)
    assert triton_kernel_config(32) == {
        "block_dk": 32,
        "num_warps": 1,
        "num_stages": 1,
        "backward_block_dv": 1,
        "backward_mode": "atomic",
        "recompute_checkpoint_interval": 64,
    }
    assert triton_kernel_config(64) == {
        "block_dk": 64,
        "num_warps": 2,
        "num_stages": 1,
        "backward_block_dv": 1,
        "backward_mode": "atomic",
        "recompute_checkpoint_interval": 64,
    }
    assert triton_kernel_config(128) == {
        "block_dk": 128,
        "num_warps": 4,
        "num_stages": 1,
        "backward_block_dv": 1,
        "backward_mode": "atomic",
        "recompute_checkpoint_interval": 64,
    }
    monkeypatch.setenv("DUALGOOSE_TRITON_NUM_WARPS", "4")
    monkeypatch.setenv("DUALGOOSE_TRITON_NUM_STAGES", "2")
    monkeypatch.setenv("DUALGOOSE_TRITON_BACKWARD_BLOCK_DV", "8")
    monkeypatch.setenv("DUALGOOSE_TRITON_BACKWARD_MODE", "two_pass")
    monkeypatch.setenv("DUALGOOSE_TRITON_RECOMPUTE_CHECKPOINT_INTERVAL", "2")
    assert triton_kernel_config(64)["num_warps"] == 4
    assert triton_kernel_config(64)["num_stages"] == 2
    assert triton_kernel_config(64)["backward_block_dv"] == 8
    assert triton_kernel_config(64)["backward_mode"] == "two_pass"
    assert triton_kernel_config(64)["recompute_checkpoint_interval"] == 2


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="Triton CUDA backend requires CUDA and Triton",
)
def test_triton_backend_matches_reference_on_cuda():
    inputs = tuple(tensor.cuda() for tensor in _inputs())
    for backend in ["triton", "triton_recompute"]:
        triton_scan = make_scan_fn(backend)
        for use_rank in [True, False]:
            expected = rwkv7_scan(*inputs, use_rank=use_rank)
            actual = triton_scan(*inputs, use_rank=use_rank)
            assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="Triton CUDA backend requires CUDA and Triton",
)
def test_triton_backend_gradients_match_reference_on_cuda(monkeypatch):
    inputs = tuple(tensor.cuda() for tensor in _inputs())
    for backend in ["triton", "triton_recompute"]:
        triton_scan = make_scan_fn(backend)
        backward_modes = ["atomic", "two_pass"] if backend == "triton" else ["atomic"]
        for backward_mode in backward_modes:
            monkeypatch.setenv("DUALGOOSE_TRITON_BACKWARD_MODE", backward_mode)
            monkeypatch.setenv("DUALGOOSE_TRITON_RECOMPUTE_CHECKPOINT_INTERVAL", "2")
            for use_rank in [True, False]:
                reference_inputs = [
                    tensor.detach().clone().requires_grad_(True) for tensor in inputs
                ]
                triton_inputs = [
                    tensor.detach().clone().requires_grad_(True) for tensor in inputs
                ]
                reference_loss = (
                    rwkv7_scan(*reference_inputs, use_rank=use_rank).square().mean()
                )
                triton_loss = triton_scan(
                    *triton_inputs, use_rank=use_rank
                ).square().mean()
                reference_loss.backward()
                triton_loss.backward()
                for reference_tensor, triton_tensor in zip(
                    reference_inputs, triton_inputs
                ):
                    if reference_tensor.grad is None:
                        assert triton_tensor.grad is None
                    else:
                        assert triton_tensor.grad is not None
                        assert torch.allclose(
                            triton_tensor.grad,
                            reference_tensor.grad,
                            rtol=1e-4,
                            atol=1e-4,
                        )
    monkeypatch.delenv("DUALGOOSE_TRITON_BACKWARD_MODE", raising=False)
    monkeypatch.delenv("DUALGOOSE_TRITON_RECOMPUTE_CHECKPOINT_INTERVAL", raising=False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="Triton CUDA backend requires CUDA and Triton",
)
def test_triton_two_pass_backward_grouping_matches_reference_on_cuda(monkeypatch):
    inputs = tuple(tensor.cuda() for tensor in _inputs())
    triton_scan = make_scan_fn("triton")
    monkeypatch.setenv("DUALGOOSE_TRITON_BACKWARD_MODE", "two_pass")
    monkeypatch.setenv("DUALGOOSE_TRITON_BACKWARD_BLOCK_DV", "2")
    for use_rank in [True, False]:
        reference_inputs = [
            tensor.detach().clone().requires_grad_(True) for tensor in inputs
        ]
        triton_inputs = [
            tensor.detach().clone().requires_grad_(True) for tensor in inputs
        ]
        reference_loss = rwkv7_scan(*reference_inputs, use_rank=use_rank).square().mean()
        triton_loss = triton_scan(*triton_inputs, use_rank=use_rank).square().mean()
        reference_loss.backward()
        triton_loss.backward()
        for reference_tensor, triton_tensor in zip(reference_inputs, triton_inputs):
            if reference_tensor.grad is None:
                assert triton_tensor.grad is None
            else:
                assert triton_tensor.grad is not None
                assert torch.allclose(
                    triton_tensor.grad,
                    reference_tensor.grad,
                    rtol=1e-4,
                    atol=1e-4,
                )


def test_forward_scan_has_no_future_leakage():
    v, k, r, w, kappa, a = _inputs()
    out = rwkv7_scan(v, k, r, w, kappa, a)
    v_changed = v.clone()
    v_changed[:, -1] = v_changed[:, -1] + 10.0
    out_changed = rwkv7_scan(v_changed, k, r, w, kappa, a)
    assert torch.allclose(out[:, :-1], out_changed[:, :-1])
    assert not torch.allclose(out[:, -1], out_changed[:, -1])


def test_reverse_scan_has_no_left_leakage_in_original_frame():
    v, k, r, w, kappa, a = _inputs()
    out = reverse_rwkv7_scan(v, k, r, w, kappa, a)
    v_changed = v.clone()
    v_changed[:, 0] = v_changed[:, 0] + 10.0
    out_changed = reverse_rwkv7_scan(v_changed, k, r, w, kappa, a)
    assert torch.allclose(out[:, 1:], out_changed[:, 1:])
    assert not torch.allclose(out[:, 0], out_changed[:, 0])
