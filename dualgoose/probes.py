"""Small deterministic probes for recurrence-side validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .mixers import StaticMerge, TimestepFiLMMerge
from .rwkv7_ref import diagonal_scan, rwkv7_scan


def last_write_replacement_probe() -> dict[str, object]:
    """Probe whether a recurrence can implement last-write-wins replacement.

    Two histories write the same key with opposite value order:

    - A then B, target B
    - B then A, target A

    With `w=1`, `k=1`, and no rank replacement, the diagonal recurrence sees
    the same final state `[1, 1]` for both histories. With rank replacement
    enabled and `a=kappa=1`, the state is erased on the key axis before each
    write, so the final output is exactly the last value.
    """

    values = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    batch, length, _ = values.shape
    ones = torch.ones(batch, length, 1)
    rank_out = rwkv7_scan(values, ones, ones, ones, ones, ones, use_rank=True)
    diagonal_out = diagonal_scan(values, ones, ones, ones)
    final_rank = rank_out[:, -1]
    final_diagonal = diagonal_out[:, -1]
    targets = torch.tensor([1, 0])

    return {
        "task": "last_write_replacement",
        "rank_final": final_rank.tolist(),
        "diagonal_final": final_diagonal.tolist(),
        "rank_predictions": final_rank.argmax(dim=-1).tolist(),
        "diagonal_predictions": final_diagonal.argmax(dim=-1).tolist(),
        "targets": targets.tolist(),
        "rank_accuracy": float(final_rank.argmax(dim=-1).eq(targets).float().mean()),
        "diagonal_accuracy": float(
            final_diagonal.argmax(dim=-1).eq(targets).float().mean()
        ),
        "diagonal_collision": bool(torch.allclose(final_diagonal[0], final_diagonal[1])),
        "rank_separates_histories": bool(not torch.allclose(final_rank[0], final_rank[1])),
    }


def _fit_merge(
    merge: torch.nn.Module,
    u_fwd: torch.Tensor,
    u_bwd: torch.Tensor,
    t: torch.Tensor,
    target: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(merge.parameters(), lr=lr)
    history = []
    for step in range(steps):
        pred = merge(u_fwd, u_bwd, t)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            history.append({"step": step + 1, "mse": float(loss.detach())})
    with torch.no_grad():
        pred = merge(u_fwd, u_bwd, t)
        low = t < 0.5
        high = ~low
        return {
            "history": history,
            "final_mse": float(F.mse_loss(pred, target)),
            "low_mse": float(F.mse_loss(pred[low], target[low])),
            "high_mse": float(F.mse_loss(pred[high], target[high])),
        }


def timestep_merge_routing_probe(
    *,
    steps: int = 300,
    d_model: int = 8,
    batch_size: int = 64,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, object]:
    """Compare static and timestep-conditioned merge on a routing task.

    The stream inputs are identical for low- and high-timestep samples. The
    target is `0` for low `t` and `1` for high `t`, so a static merge cannot
    solve both buckets from stream values alone. The FiLM merge gets `t` and
    can route the output by timestep.
    """

    torch.manual_seed(seed)
    t = torch.cat(
        [
            torch.full((batch_size // 2,), 0.2),
            torch.full((batch_size - batch_size // 2,), 0.8),
        ]
    )
    u_fwd = torch.zeros(batch_size, 1, d_model)
    u_bwd = torch.zeros(batch_size, 1, d_model)
    target = torch.where(
        t[:, None, None] < 0.5,
        torch.zeros_like(u_fwd),
        torch.ones_like(u_fwd),
    )

    static = StaticMerge(d_model)
    film = TimestepFiLMMerge(d_model, initial_scale=1.0, gate_bias_init=0.0)
    static_result = _fit_merge(static, u_fwd, u_bwd, t, target, steps=steps, lr=lr)
    film_result = _fit_merge(film, u_fwd, u_bwd, t, target, steps=steps, lr=lr)
    return {
        "task": "timestep_merge_routing",
        "steps": steps,
        "d_model": d_model,
        "batch_size": batch_size,
        "lr": lr,
        "static": static_result,
        "film": film_result,
        "film_beats_static": film_result["final_mse"] < static_result["final_mse"],
        "film_bucket_max_mse": max(film_result["low_mse"], film_result["high_mse"]),
        "static_bucket_min_mse": min(static_result["low_mse"], static_result["high_mse"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe", choices=["last_write", "merge_routing"], default="last_write"
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.probe == "last_write":
        result = last_write_replacement_probe()
        default_output = "results/last_write_probe.json"
    else:
        result = timestep_merge_routing_probe()
        default_output = "results/timestep_merge_probe.json"
    output = Path(args.output or default_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
