import torch

from dualgoose.eval_bucketed import evaluate_buckets
from dualgoose.model import build_tiny_denoiser
from dualgoose.text_data import make_text_batcher


def _make_text(tmp_path):
    text_file = tmp_path / "corpus.txt"
    text_file.write_text("DualGoose buckets validate masked diffusion denoisers. " * 32)
    return text_file


def test_evaluate_buckets_reports_one_row_per_ratio(tmp_path):
    vocab_size = 32
    mask_token_id = vocab_size - 1
    text_file = _make_text(tmp_path)
    model = build_tiny_denoiser(
        vocab_size,
        16,
        model_type="dualgoose",
        d_model=16,
        n_layers=1,
        direction="bidirectional",
        merge="film",
        scan_backend="reference",
        n_heads=2,
    )
    batcher = make_text_batcher(text_file, vocab_size, mask_token_id, tokenizer="byte")

    ratios = [0.2, 0.5, 0.8]
    buckets = evaluate_buckets(
        model,
        batcher,
        mask_ratios=ratios,
        batch_size=4,
        seq_len=16,
        mask_token_id=mask_token_id,
        device=torch.device("cpu"),
        eval_batches=3,
        seed=7,
    )

    assert [b["mask_ratio"] for b in buckets] == ratios
    for b in buckets:
        assert 0.0 <= b["masked_accuracy"] <= 1.0
        assert b["masked_tokens"] > 0
        # Observed mask fraction should track the requested ratio.
        assert abs(b["observed_mask_fraction"] - b["mask_ratio"]) < 0.12


def test_evaluate_buckets_is_deterministic(tmp_path):
    vocab_size = 32
    mask_token_id = vocab_size - 1
    text_file = _make_text(tmp_path)
    model = build_tiny_denoiser(
        vocab_size,
        16,
        model_type="attention",
        d_model=16,
        n_layers=1,
        n_heads=2,
    )
    batcher = make_text_batcher(text_file, vocab_size, mask_token_id, tokenizer="byte")

    kwargs = dict(
        mask_ratios=[0.3, 0.6],
        batch_size=4,
        seq_len=16,
        mask_token_id=mask_token_id,
        device=torch.device("cpu"),
        eval_batches=2,
        seed=11,
    )
    first = evaluate_buckets(model, batcher, **kwargs)
    second = evaluate_buckets(model, batcher, **kwargs)
    for a, b in zip(first, second):
        assert a["masked_nll"] == b["masked_nll"]
        assert a["masked_accuracy"] == b["masked_accuracy"]
