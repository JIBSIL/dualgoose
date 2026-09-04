"""Short-conv arming of the recurrent cells (docs/armed_gdn_experiment.md).

Verifies the optional Mamba-2/GDN-style short causal conv added to DeltaNetRefLayer and
RWKV7RefLayer: (1) construction + output shape are unchanged, (2) the conv is STRICTLY CAUSAL
(a perturbation at position t never changes outputs at positions < t), (3) it is off by default,
and (4) BidirectionalRWKVMixer threads it into both streams. Pure CPU — no GPU / mamba_ssm / fla.
"""

import torch

from dualgoose.mixers import BidirectionalRWKVMixer, DeltaNetRefLayer, RWKV7RefLayer
from dualgoose.model import TinyMamba2RefSSM, TinySelectiveSSM, build_tiny_denoiser


def _assert_causal_and_shaped(layer, d_model, seq_len=12):
    torch.manual_seed(0)
    x = torch.randn(2, seq_len, d_model)
    y = layer(x)
    assert y.shape == (2, seq_len, d_model)
    # Perturb the last position; strictly-causal mixing must leave every earlier output identical.
    t = seq_len - 1
    x2 = x.clone()
    x2[:, t] += 5.0
    y2 = layer(x2)
    assert torch.allclose(y[:, :t], y2[:, :t], atol=1e-5), "short conv leaked future context"


def test_deltanet_short_conv_causal_and_shape():
    layer = DeltaNetRefLayer(16, gated=True, short_conv=4).eval()
    assert hasattr(layer, "qkv_conv")
    _assert_causal_and_shaped(layer, 16)


def test_rwkv7_short_conv_causal_and_shape():
    layer = RWKV7RefLayer(16, scan_backend="reference", short_conv=4).eval()
    assert hasattr(layer, "vkr_conv")
    _assert_causal_and_shaped(layer, 16)


def test_short_conv_off_by_default():
    assert not hasattr(DeltaNetRefLayer(16, gated=True), "qkv_conv")
    assert not hasattr(RWKV7RefLayer(16, scan_backend="reference"), "vkr_conv")


def test_mixer_threads_short_conv_into_both_streams():
    m = BidirectionalRWKVMixer(16, recurrence="gated_deltanet", short_conv=4, merge="static")
    assert m.fwd.short_conv == 4 and m.bwd.short_conv == 4
    # armed and unarmed differ only by the conv params -> armed has strictly more params
    unarmed = BidirectionalRWKVMixer(16, recurrence="gated_deltanet", short_conv=0, merge="static")
    assert sum(p.numel() for p in m.parameters()) > sum(p.numel() for p in unarmed.parameters())


def test_local_mamba_conv_toggle_causal_and_builds():
    # fallback decomposition cell (pure PyTorch, no mamba_ssm): the conv kernel toggles between
    # k=3 (mamba_local) and k=1 (mamba_local_noconv); both must stay strictly causal.
    for k in (1, 3):
        _assert_causal_and_shaped(TinySelectiveSSM(16, kernel_size=k).eval(), 16)
    # end-to-end build via build_tiny_denoiser (the fallback arm path), CPU forward
    for k in (1, 3):
        m = build_tiny_denoiser(32, 16, model_type="mamba", d_model=16, n_layers=1,
                                mamba_kernel_size=k)
        y = m(torch.randint(0, 32, (2, 16)), torch.full((2,), 0.5))
        assert y.shape == (2, 16, 32)


def test_mamba2_ref_state_size_causal_and_builds():
    # state-matched diagonal SSM comparator (Path B): state = d_inner * d_state, runs w/o mamba_ssm.
    cell = TinyMamba2RefSSM(16, d_state=16, kernel_size=4).eval()
    assert cell.nheads * cell.headdim * cell.d_state == cell.d_inner * cell.d_state == 32 * 16
    _assert_causal_and_shaped(cell, 16)
    # conv toggle (kernel=1 => neutralized) stays causal
    _assert_causal_and_shaped(TinyMamba2RefSSM(16, d_state=16, kernel_size=1).eval(), 16)
    # end-to-end factory build (mamba2_ref arm), incl. the noconv variant; CPU forward
    for k in (1, 4):
        m = build_tiny_denoiser(32, 16, model_type="mamba2_ref", d_model=16, n_layers=1,
                                mamba2_d_state=16, mamba2_d_conv=k)
        y = m(torch.randint(0, 32, (2, 16)), torch.full((2,), 0.5))
        assert y.shape == (2, 16, 32)


def test_mamba2_ref_state_matches_rank1_at_d32():
    # the P1 parity point: at d_model=32, d_state=16 => 64*16 = 1024 state elements, matching a
    # d_key=32 delta cell's 32*32 = 1024. (documents the state-match the experiment relies on)
    cell = TinyMamba2RefSSM(32, d_state=16)
    assert cell.d_inner * cell.d_state == 1024
    assert DeltaNetRefLayer(32).d_key ** 2 == 1024
