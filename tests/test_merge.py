import torch

from dualgoose.mixers import StaticMerge, TimestepFiLMMerge


def test_static_merge_shape():
    merge = StaticMerge(8)
    u_fwd = torch.randn(2, 4, 8)
    u_bwd = torch.randn(2, 4, 8)
    out = merge(u_fwd, u_bwd)
    assert out.shape == (2, 4, 8)
    assert torch.allclose(out, 0.5 * (u_fwd + u_bwd))


def test_timestep_merge_shape_small_initial_value_and_gradients():
    merge = TimestepFiLMMerge(8)
    u_fwd = torch.randn(2, 4, 8, requires_grad=True)
    u_bwd = torch.randn(2, 4, 8, requires_grad=True)
    t = torch.tensor([0.2, 0.8], requires_grad=True)
    out = merge(u_fwd, u_bwd, t)
    assert out.shape == (2, 4, 8)
    assert out.detach().abs().mean().item() < 1e-3
    out.square().sum().backward()
    assert u_fwd.grad is not None and u_fwd.grad.abs().sum() > 0
    assert u_bwd.grad is not None and u_bwd.grad.abs().sum() > 0
    assert t.grad is not None
