"""Run small multi-seed toy sweeps for validation reports."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .diffusion import apply_subs_sample, masked_nll_loss
from .model import TinyDualGooseDenoiser
from .toy_data import make_bidirectional_copy_batch, make_mqar_batch
from .train_toy import git_metadata


@dataclass(frozen=True)
class TrialSpec:
    name: str
    task: str
    seed: int
    steps: int
    batch_size: int
    eval_batch_size: int
    seq_len: int
    vocab_size: int
    d_model: int
    layers: int
    lr: float
    direction: str = "bidirectional"
    merge: str = "static"
    diagonal_only: bool = False
    copy_direction: str = "mixed"
    mqar_pairs: int = 4
    mqar_layout: str = "middle"


def quick_suite() -> list[TrialSpec]:
    """A CPU-friendly suite that exercises copy and MQAR paths."""

    specs: list[TrialSpec] = []
    for seed in [0, 1]:
        specs.extend(
            [
                TrialSpec(
                    name="copy_right_bidi",
                    task="copy",
                    seed=seed,
                    steps=220,
                    batch_size=64,
                    eval_batch_size=256,
                    seq_len=16,
                    vocab_size=32,
                    d_model=24,
                    layers=1,
                    lr=0.003,
                    direction="bidirectional",
                    copy_direction="right",
                ),
                TrialSpec(
                    name="copy_right_forward",
                    task="copy",
                    seed=seed,
                    steps=220,
                    batch_size=64,
                    eval_batch_size=256,
                    seq_len=16,
                    vocab_size=32,
                    d_model=24,
                    layers=1,
                    lr=0.003,
                    direction="forward",
                    copy_direction="right",
                ),
                TrialSpec(
                    name="mqar_bidi_rank",
                    task="mqar",
                    seed=seed,
                    steps=180,
                    batch_size=64,
                    eval_batch_size=256,
                    seq_len=16,
                    vocab_size=32,
                    d_model=32,
                    layers=1,
                    lr=0.003,
                    direction="bidirectional",
                    mqar_pairs=4,
                ),
                TrialSpec(
                    name="mqar_bidi_diagonal",
                    task="mqar",
                    seed=seed,
                    steps=180,
                    batch_size=64,
                    eval_batch_size=256,
                    seq_len=16,
                    vocab_size=32,
                    d_model=32,
                    layers=1,
                    lr=0.003,
                    direction="bidirectional",
                    diagonal_only=True,
                    mqar_pairs=4,
                ),
            ]
        )
    return specs


def mqar_grid_suite() -> list[TrialSpec]:
    """Small L/K MQAR grid for CPU reference validation."""

    specs: list[TrialSpec] = []
    for seed in [0, 1]:
        for pairs, seq_len, d_model, steps in [
            (4, 16, 32, 180),
            (8, 24, 48, 240),
        ]:
            for diagonal_only in [False, True]:
                specs.append(
                    TrialSpec(
                        name=f"mqar_k{pairs}_{'diag' if diagonal_only else 'rank'}",
                        task="mqar",
                        seed=seed,
                        steps=steps,
                        batch_size=64,
                        eval_batch_size=256,
                        seq_len=seq_len,
                        vocab_size=64,
                        d_model=d_model,
                        layers=1,
                        lr=0.003,
                        direction="bidirectional",
                        diagonal_only=diagonal_only,
                        mqar_pairs=pairs,
                    )
                )
    return specs


def mqar_k8_long_suite() -> list[TrialSpec]:
    """Longer K=8 MQAR stability sweep before attempting K=16."""

    specs: list[TrialSpec] = []
    for seed in [0, 1]:
        for diagonal_only in [False, True]:
            specs.append(
                TrialSpec(
                    name=f"mqar_k8_long_{'diag' if diagonal_only else 'rank'}",
                    task="mqar",
                    seed=seed,
                    steps=480,
                    batch_size=64,
                    eval_batch_size=256,
                    seq_len=24,
                    vocab_size=64,
                    d_model=64,
                    layers=1,
                    lr=0.003,
                    direction="bidirectional",
                    diagonal_only=diagonal_only,
                    mqar_pairs=8,
                )
            )
    return specs


def mqar_k8_solve_suite() -> list[TrialSpec]:
    """Standard after-table K=8 MQAR solve check for recurrent memory capacity."""

    specs: list[TrialSpec] = []
    for diagonal_only, seeds in [(False, [0, 1, 2, 3]), (True, [0, 1])]:
        for seed in seeds:
            specs.append(
                TrialSpec(
                    name=f"mqar_k8_solve_{'diag' if diagonal_only else 'rank'}",
                    task="mqar",
                    seed=seed,
                    steps=480,
                    batch_size=64,
                    eval_batch_size=512,
                    seq_len=24,
                    vocab_size=64,
                    d_model=64,
                    layers=1,
                    lr=0.003,
                    direction="forward",
                    diagonal_only=diagonal_only,
                    mqar_pairs=8,
                    mqar_layout="after_table",
                )
            )
    return specs


def mqar_k16_probe_suite() -> list[TrialSpec]:
    """First K=16 after-table MQAR scaling probe on the V100 reference path."""

    common = {
        "task": "mqar",
        "seed": 0,
        "steps": 960,
        "batch_size": 64,
        "eval_batch_size": 512,
        "seq_len": 40,
        "vocab_size": 64,
        "lr": 0.003,
        "direction": "forward",
        "mqar_pairs": 16,
        "mqar_layout": "after_table",
    }
    return [
        TrialSpec(
            name="mqar_k16_probe_rank_d64_l1",
            d_model=64,
            layers=1,
            diagonal_only=False,
            **common,
        ),
        TrialSpec(
            name="mqar_k16_probe_rank_d96_l1",
            d_model=96,
            layers=1,
            diagonal_only=False,
            lr=0.002,
            **{key: value for key, value in common.items() if key != "lr"},
        ),
        TrialSpec(
            name="mqar_k16_probe_rank_d64_l2",
            d_model=64,
            layers=2,
            diagonal_only=False,
            **common,
        ),
        TrialSpec(
            name="mqar_k16_probe_diag_d64_l1",
            d_model=64,
            layers=1,
            diagonal_only=True,
            **common,
        ),
    ]


def mqar_k16_solve_suite() -> list[TrialSpec]:
    """Extended-budget K=16 MQAR solve check on the V100 reference path."""

    specs: list[TrialSpec] = []
    for seed in [0, 1]:
        specs.append(
            TrialSpec(
                name="mqar_k16_solve_rank_d64_l2",
                task="mqar",
                seed=seed,
                steps=3000,
                batch_size=64,
                eval_batch_size=512,
                seq_len=40,
                vocab_size=64,
                d_model=64,
                layers=2,
                lr=0.002,
                direction="forward",
                diagonal_only=False,
                mqar_pairs=16,
                mqar_layout="after_table",
            )
        )
    return specs


def make_batch(
    spec: TrialSpec,
    batch_size: int,
    mask_token_id: int,
    device: torch.device,
):
    if spec.task == "copy":
        return make_bidirectional_copy_batch(
            batch_size,
            spec.seq_len,
            spec.vocab_size,
            mask_token_id,
            direction=spec.copy_direction,
            device=device,
        )
    if spec.task == "mqar":
        return make_mqar_batch(
            batch_size,
            spec.seq_len,
            spec.vocab_size,
            mask_token_id,
            pairs=spec.mqar_pairs,
            query_layout=spec.mqar_layout,
            device=device,
        )
    raise ValueError(f"unsupported task: {spec.task}")


def run_trial(spec: TrialSpec, device: torch.device) -> dict[str, object]:
    torch.manual_seed(spec.seed)
    mask_token_id = spec.vocab_size - 1
    model = TinyDualGooseDenoiser(
        spec.vocab_size,
        spec.seq_len,
        d_model=spec.d_model,
        n_layers=spec.layers,
        direction=spec.direction,
        merge=spec.merge,
        use_rank=not spec.diagonal_only,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.lr)
    losses: list[float] = []
    for _ in range(spec.steps):
        batch = make_batch(spec, spec.batch_size, mask_token_id, device)
        logits = model(batch.corrupted, batch.t)
        loss = masked_nll_loss(logits, batch.clean, batch.corrupted, mask_token_id, t=batch.t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    torch.manual_seed(spec.seed + 10_000)
    model.eval()
    with torch.no_grad():
        batch = make_batch(spec, spec.eval_batch_size, mask_token_id, device)
        logits = model(batch.corrupted, batch.t)
        pred = apply_subs_sample(logits, batch.corrupted, mask_token_id)
        accuracy = pred[batch.target_mask].eq(batch.clean[batch.target_mask]).float().mean()

    return {
        **asdict(spec),
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "eval_accuracy": float(accuracy.cpu()),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["name"]), []).append(row)
    summary = []
    for name, group in sorted(grouped.items()):
        acc = torch.tensor([float(row["eval_accuracy"]) for row in group])
        loss = torch.tensor([float(row["final_loss"]) for row in group])
        summary.append(
            {
                "name": name,
                "n": len(group),
                "mean_eval_accuracy": float(acc.mean()),
                "std_eval_accuracy": float(acc.std(unbiased=False)),
                "mean_final_loss": float(loss.mean()),
                "std_final_loss": float(loss.std(unbiased=False)),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=[
            "quick",
            "mqar_grid",
            "mqar_k8_long",
            "mqar_k8_solve",
            "mqar_k16_probe",
            "mqar_k16_solve",
        ],
        default="quick",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default="results/toy_sweep_quick.json")
    parser.add_argument("--output-csv", default="results/toy_sweep_quick.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.suite == "quick":
        specs = quick_suite()
    elif args.suite == "mqar_grid":
        specs = mqar_grid_suite()
    elif args.suite == "mqar_k8_long":
        specs = mqar_k8_long_suite()
    elif args.suite == "mqar_k8_solve":
        specs = mqar_k8_solve_suite()
    elif args.suite == "mqar_k16_probe":
        specs = mqar_k16_probe_suite()
    else:
        specs = mqar_k16_solve_suite()
    device = torch.device(args.device)
    rows = [run_trial(spec, device) for spec in specs]
    result = {
        "suite": args.suite,
        "device": str(device),
        "git": git_metadata(),
        "trials": rows,
        "summary": summarize(rows),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(Path(args.output_csv), rows)


if __name__ == "__main__":
    main()
