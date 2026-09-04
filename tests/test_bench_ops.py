import torch

from dualgoose.bench_ops import (
    benchmark_attention_case,
    benchmark_scan_case,
    make_attention_inputs,
)


def test_attention_inputs_and_operator_benchmarks_run_on_cpu():
    device = torch.device("cpu")
    q, k, v = make_attention_inputs(
        batch_size=2,
        seq_len=4,
        d_model=8,
        heads=2,
        device=device,
    )
    assert q.shape == k.shape == v.shape == (2, 2, 4, 4)

    scan_row = benchmark_scan_case(
        batch_size=2,
        seq_len=4,
        d_model=4,
        device=device,
        warmup=0,
        iters=1,
        compile_scan=False,
    )
    attention_row = benchmark_attention_case(
        batch_size=2,
        seq_len=4,
        d_model=8,
        heads=2,
        device=device,
        warmup=0,
        iters=1,
    )
    assert scan_row["op"] == "rwkv7_rank_scan"
    assert attention_row["op"] == "scaled_dot_product_attention"
    assert scan_row["tokens_per_s"] > 0.0
    assert attention_row["tokens_per_s"] > 0.0
