import torch

from dualgoose.bench_scan import benchmark_case, make_inputs


def test_bench_scan_inputs_and_case_run_on_cpu():
    device = torch.device("cpu")
    inputs = make_inputs(batch_size=2, seq_len=4, d_model=3, device=device)
    assert [tensor.shape for tensor in inputs] == [(2, 4, 3)] * 6
    _, k, r, w, kappa, a = inputs
    scale = 3**-0.5
    assert k.abs().max() <= scale
    assert r.abs().max() <= scale
    assert kappa.abs().max() <= scale
    assert a.le(w).all()
    row = benchmark_case(
        batch_size=2,
        seq_len=4,
        d_model=3,
        device=device,
        use_rank=True,
        warmup=0,
        iters=1,
    )
    assert row["seq_len"] == 4
    assert row["use_rank"] is True
    assert row["scan_backend"] == "reference"
    assert row["triton_num_warps"] == ""
    assert row["triton_backward_block_dv"] == ""
    assert row["triton_backward_mode"] == ""
    assert row["triton_recompute_checkpoint_interval"] == ""
    assert row["compile_scan"] is False
    assert row["include_backward"] is False
    assert row["tokens_per_s"] > 0.0

    backward_row = benchmark_case(
        batch_size=2,
        seq_len=4,
        d_model=3,
        device=device,
        use_rank=True,
        warmup=0,
        iters=1,
        include_backward=True,
    )
    assert backward_row["include_backward"] is True
    assert backward_row["loss"] is not None
    assert backward_row["grad_abs_mean"] is not None
    assert backward_row["tokens_per_s"] > 0.0
