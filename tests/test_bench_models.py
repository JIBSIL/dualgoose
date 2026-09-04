import importlib.util

import torch

from dualgoose.bench_models import benchmark_model_case, build_model, make_batch
from dualgoose.model import (
    TinyAttentionDenoiser,
    TinyDiffuMambaHDenoiser,
    TinyDualGooseDenoiser,
    TinyMamba2Denoiser,
    TinyMambaDenoiser,
)


def test_model_benchmark_helpers_run_on_cpu():
    device = torch.device("cpu")
    tokens, t = make_batch(batch_size=2, seq_len=6, vocab_size=32, device=device)
    assert tokens.shape == (2, 6)
    assert t.shape == (2,)
    assert isinstance(
        build_model("dualgoose", vocab_size=32, seq_len=6, d_model=8, layers=1, heads=2),
        TinyDualGooseDenoiser,
    )
    assert isinstance(
        build_model("attention", vocab_size=32, seq_len=6, d_model=8, layers=1, heads=2),
        TinyAttentionDenoiser,
    )
    assert isinstance(
        build_model("mamba", vocab_size=32, seq_len=6, d_model=8, layers=1, heads=2),
        TinyMambaDenoiser,
    )
    assert isinstance(
        build_model(
            "diffumamba_h",
            vocab_size=32,
            seq_len=6,
            d_model=8,
            layers=1,
            heads=2,
            attention_window=2,
        ),
        TinyDiffuMambaHDenoiser,
    )
    if importlib.util.find_spec("mamba_ssm") is not None:
        assert isinstance(
            build_model(
                "mamba2", vocab_size=32, seq_len=6, d_model=8, layers=1, heads=2
            ),
            TinyMamba2Denoiser,
        )
    row = benchmark_model_case(
        model_type="attention",
        batch_size=2,
        seq_len=6,
        vocab_size=32,
        d_model=8,
        layers=1,
        heads=2,
        device=device,
        warmup=0,
        iters=1,
    )
    assert row["model_type"] == "attention"
    assert row["scan_backend"] == ""
    assert row["include_backward"] is False
    assert row["optimizer_step"] is False
    assert row["tokens_per_s"] > 0.0

    dualgoose_row = benchmark_model_case(
        model_type="dualgoose",
        batch_size=2,
        seq_len=6,
        vocab_size=32,
        d_model=8,
        layers=1,
        heads=2,
        device=device,
        warmup=0,
        iters=1,
        scan_backend="reference",
    )
    assert dualgoose_row["scan_backend"] == "reference"
    assert dualgoose_row["tokens_per_s"] > 0.0

    backward_row = benchmark_model_case(
        model_type="attention",
        batch_size=2,
        seq_len=6,
        vocab_size=32,
        d_model=8,
        layers=1,
        heads=2,
        device=device,
        warmup=0,
        iters=1,
        include_backward=True,
    )
    assert backward_row["include_backward"] is True
    assert backward_row["loss"] is not None
    assert backward_row["tokens_per_s"] > 0.0

    hybrid_row = benchmark_model_case(
        model_type="diffumamba_h",
        batch_size=2,
        seq_len=6,
        vocab_size=32,
        d_model=8,
        layers=1,
        heads=2,
        attention_window=2,
        device=device,
        warmup=0,
        iters=1,
    )
    assert hybrid_row["model_type"] == "diffumamba_h"
    assert hybrid_row["heads"] == 2
    assert hybrid_row["attention_window"] == 2
    assert hybrid_row["tokens_per_s"] > 0.0

    optimizer_row = benchmark_model_case(
        model_type="attention",
        batch_size=2,
        seq_len=6,
        vocab_size=32,
        d_model=8,
        layers=1,
        heads=2,
        device=device,
        warmup=0,
        iters=1,
        include_backward=True,
        optimizer_step=True,
    )
    assert optimizer_row["optimizer_step"] is True
    assert optimizer_row["loss"] is not None
