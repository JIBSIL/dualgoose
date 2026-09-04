import importlib.util
import os

import pytest
import torch

from dualgoose.model import (
    TinyAttentionDenoiser,
    TinyDiffuMambaHDenoiser,
    TinyMamba2Denoiser,
    TinyMambaDenoiser,
)


def test_tiny_attention_denoiser_shape_and_head_validation():
    model = TinyAttentionDenoiser(32, 12, d_model=16, n_layers=1, n_heads=4)
    tokens = torch.randint(0, 32, (2, 12))
    t = torch.tensor([0.25, 0.75])
    logits = model(tokens, t)
    assert logits.shape == (2, 12, 32)

    with pytest.raises(ValueError, match="d_model must be divisible"):
        TinyAttentionDenoiser(32, 12, d_model=15, n_layers=1, n_heads=4)


def test_tiny_mamba_denoiser_shape():
    model = TinyMambaDenoiser(32, 12, d_model=16, n_layers=1)
    tokens = torch.randint(0, 32, (2, 12))
    t = torch.tensor([0.25, 0.75])
    logits = model(tokens, t)
    assert logits.shape == (2, 12, 32)


def test_tiny_diffumamba_h_denoiser_sparse_attention_shape_and_head_validation():
    model = TinyDiffuMambaHDenoiser(
        32,
        12,
        d_model=16,
        n_layers=1,
        n_heads=4,
        attention_window=2,
    )
    tokens = torch.randint(0, 32, (2, 12))
    t = torch.tensor([0.25, 0.75])
    logits = model(tokens, t)
    assert logits.shape == (2, 12, 32)

    with pytest.raises(ValueError, match="d_model must be divisible"):
        TinyDiffuMambaHDenoiser(32, 12, d_model=15, n_layers=1, n_heads=4)

    with pytest.raises(ValueError, match="window_size must be non-negative"):
        TinyDiffuMambaHDenoiser(32, 12, d_model=16, n_layers=1, attention_window=-1)


@pytest.mark.skipif(
    importlib.util.find_spec("mamba_ssm") is None
    or not torch.cuda.is_available()
    or os.getenv("DUALGOOSE_RUN_MAMBA2_CUDA_TEST") != "1",
    reason="official Mamba2 CUDA smoke is optional and JIT-compiles kernels",
)
def test_tiny_mamba2_denoiser_shape_on_cuda():
    model = TinyMamba2Denoiser(32, 12, d_model=64, n_layers=1).cuda().eval()
    tokens = torch.randint(0, 32, (2, 12), device="cuda")
    t = torch.tensor([0.25, 0.75], device="cuda")
    with torch.no_grad():
        logits = model(tokens, t)
    torch.cuda.synchronize()
    assert logits.shape == (2, 12, 32)
