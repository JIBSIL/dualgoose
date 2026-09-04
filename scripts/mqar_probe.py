"""MQAR capacity probe (C8/C9): does the rank-1 RWKV-7 recurrence beat a diagonal-only
ablation and official Mamba-2 on masked associative recall, as K (number of key/value pairs)
grows? This is the task where rank-1 transitions *should* help — unlike LM, where Mamba-2 wins.

Self-contained GPU trainer+evaluator (mirrors dualgoose.sweep_toy.run_trial) extended with a
Mamba-2 arm. Per-trial JSON outputs under results/mqar_probe/ make it idempotent/resumable.
Run in .venv-mamba (torch 2.7.1) so the mamba2 arm works; DualGoose-triton parity there is
confirmed.
"""

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch

from dualgoose.diffusion import apply_subs_sample, masked_nll_loss
from dualgoose.model import build_tiny_denoiser
from dualgoose.toy_data import make_mqar_batch
from dualgoose.train_toy import git_metadata

OUTDIR = Path("results/mqar_probe")


def amp_ctx(fp32):
    """bf16 autocast, or a no-op for fp32 (GPUs without hardware bf16, e.g. Pascal / GTX 1070)."""
    return contextlib.nullcontext() if fp32 else torch.autocast("cuda", dtype=torch.bfloat16)

# (label, model_type, use_rank, extra build kwargs)
MODELS = {
    "dg_rank": dict(model_type="dualgoose", use_rank=True, scan_backend="triton",
                    merge="static", direction="bidirectional"),
    "dg_diag": dict(model_type="dualgoose", use_rank=False, scan_backend="triton",
                    merge="static", direction="bidirectional"),
    "mamba2": dict(model_type="mamba2"),
    "attention": dict(model_type="attention"),  # recall ceiling (full self-attention)
    # rotary variant — with --heads 2 (head_dim 16 at d=32) these give the factorial its valid
    # attention row (the original d32/8-head arm was head-dim-degenerate and excluded)
    "attention_rope": dict(model_type="attention_rope"),
    "deltanet": dict(model_type="deltanet", merge="static", direction="bidirectional"),
    # --- armed-cell fair-fight arms (docs/armed_gdn_experiment.md) ---
    # gated_deltanet = rank-1 delta + diagonal decay (no conv); armed_gdn adds Mamba-2's
    # short causal conv at ZERO extra recurrent state, so vs mamba2(ds16) it is state-matched.
    "gated_deltanet": dict(model_type="gated_deltanet", merge="static", direction="bidirectional"),
    "armed_gdn": dict(model_type="gated_deltanet", short_conv=4,
                      merge="static", direction="bidirectional"),
    "dg_rank_conv": dict(model_type="dualgoose", use_rank=True, scan_backend="triton",
                         short_conv=4, merge="static", direction="bidirectional"),
    # ablation-grade WITH-CONV quadrant (post-mortem hardening): same RWKV cell, same conv,
    # only use_rank toggles — the class-free version of the armed rank-1-vs-diagonal read.
    # Reference scan so the pair runs on Pascal.
    "dg_rank_conv_ref": dict(model_type="dualgoose", use_rank=True, scan_backend="reference",
                             short_conv=4, merge="static", direction="bidirectional"),
    "dg_diag_conv": dict(model_type="dualgoose", use_rank=False, scan_backend="reference",
                         short_conv=4, merge="static", direction="bidirectional"),
    # decomposition control: full Mamba-2 with its short conv neutralized (isolates conv's share).
    # PREFERRED (state-matched, official cell) but needs mamba_ssm to accept d_conv=1.
    "mamba2_noconv": dict(model_type="mamba2", mamba2_d_conv=1),
    # FALLBACK decomposition on a pure-PyTorch diagonal selective SSM (always builds, incl. V100
    # sm_70 / no mamba_ssm): mamba_local (conv k=3) vs mamba_local_noconv (k=1) isolates the conv's
    # share on a diagonal cell. NB: state = d_model here, so this is the conv-share read, NOT the
    # state-matched headline parity (that stays official mamba2 ds16). See docs/armed_gdn_experiment.md.
    "mamba_local": dict(model_type="mamba", mamba_kernel_size=3),
    "mamba_local_noconv": dict(model_type="mamba", mamba_kernel_size=1),
    # --- Pascal / GTX 1070-native (no Triton / no mamba_ssm): reference RWKV scan + pure-PyTorch
    # STATE-MATCHED Mamba-2. Lets the full fair fight (incl. the P1 parity call) run on sm_61. ---
    "dg_rank_ref": dict(model_type="dualgoose", use_rank=True, scan_backend="reference",
                        merge="static", direction="bidirectional"),
    "dg_diag_ref": dict(model_type="dualgoose", use_rank=False, scan_backend="reference",
                        merge="static", direction="bidirectional"),
    "mamba2_ref": dict(model_type="mamba2_ref"),               # state-matched via --mamba2-d-state
    "mamba2_ref_noconv": dict(model_type="mamba2_ref", mamba2_d_conv=1),
}

# model_types whose recurrent state is matched via --mamba2-d-state (d_inner * d_state)
_STATE_MATCHED = {"mamba2", "mamba2_ref"}


def build(label, vocab, seq_len, d_model, layers, heads, mamba2_d_state=None):
    kw = dict(MODELS[label])
    if kw.get("model_type") in _STATE_MATCHED and mamba2_d_state is not None:
        kw["mamba2_d_state"] = mamba2_d_state
    return build_tiny_denoiser(vocab, seq_len, d_model=d_model, n_layers=layers,
                               n_heads=heads, **kw)


def run_trial(label, K, seed, *, d_model, layers, heads, vocab, steps, lr, batch,
              eval_batch, eval_iters, device, mamba2_d_state=None, fp32=False, rebind_eval=False):
    seq_len = 2 * K + 8  # after_table: table (2K) + query/answer (2) + distractor slack
    mask_id = vocab - 1
    torch.manual_seed(seed)
    model = build(label, vocab, seq_len, d_model, layers, heads, mamba2_d_state).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    last_loss = float("nan")
    for _ in range(steps):
        b = make_mqar_batch(batch, seq_len, vocab, mask_id, pairs=K,
                            query_layout="after_table", device=device, generator=gen)
        with amp_ctx(fp32):
            logits = model(b.corrupted, b.t)
            loss = masked_nll_loss(logits, b.clean, b.corrupted, mask_id, t=b.t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last_loss = float(loss.detach().cpu())
    # eval recall accuracy on fresh batches (held-out by generator seed)
    model.eval()
    egen = torch.Generator(device=device).manual_seed(seed + 10_000)
    correct = total = rebind_correct = 0
    # after_table layout: table at positions 0..2K-1, value slots at odd offsets
    vpos = torch.tensor([2 * p + 1 for p in range(K)], device=device)
    with torch.no_grad():
        for _ in range(eval_iters):
            b = make_mqar_batch(eval_batch, seq_len, vocab, mask_id, pairs=K,
                                query_layout="after_table", device=device, generator=egen)
            with amp_ctx(fp32):
                logits = model(b.corrupted, b.t)
            pred = apply_subs_sample(logits, b.corrupted, mask_id)
            m = b.target_mask
            correct += int(pred[m].eq(b.clean[m]).sum().cpu())
            total += int(m.sum().cpu())
            if rebind_eval:
                # rebinding control (post-mortem hardening): cyclically derange the table VALUES in
                # the model input, score vs the ORIGINAL answer — a binding-reader collapses to
                # ~floor; a prior-exploiter keeps its accuracy.
                rc = b.corrupted.clone()
                rc[:, vpos] = b.corrupted[:, vpos.roll(-1)]
                with amp_ctx(fp32):
                    rlogits = model(rc, b.t)
                rpred = apply_subs_sample(rlogits, rc, mask_id)
                rebind_correct += int(rpred[m].eq(b.clean[m]).sum().cpu())
    acc = correct / max(1, total)
    rebind_acc = (rebind_correct / max(1, total)) if rebind_eval else None
    return {
        "label": label, "K": K, "seed": seed, "seq_len": seq_len,
        "d_model": d_model, "layers": layers, "params": nparams,
        "mamba2_d_state": mamba2_d_state if MODELS[label].get("model_type") in _STATE_MATCHED else None,
        "amp": "fp32" if fp32 else "bf16",
        "rebind_accuracy": rebind_acc,
        # post-mortem floors: value-marginal, uniform-over-table, and the strongest no-binding
        # strategy ("emit a random table value" — the answer value may recur in other pairs)
        "value_marginal_floor": round(1.0 / (vocab - 1 - (5 + K)), 5),
        "table_floor": round(1.0 / K, 5),
        "no_binding_floor": round(1.0 / K + (K - 1) / (K * (vocab - 1 - (5 + K))), 5),
        "steps": steps, "lr": lr, "batch": batch,
        "final_loss": last_loss, "eval_accuracy": acc,
        "eval_samples": total, "tok_s": round(batch * seq_len * steps / (time.time() - t0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--ks", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--eval-iters", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=None,
                    help="override the per-K step schedule (for sample-efficiency cuts)")
    ap.add_argument("--tag", default="", help="suffix added to output filenames")
    ap.add_argument("--mamba2-d-state", type=int, default=None,
                    help="override Mamba-2 d_state for state-matched comparisons (default 64)")
    ap.add_argument("--fp32", action="store_true",
                    help="disable bf16 autocast (for GPUs without hardware bf16, e.g. Pascal / GTX 1070)")
    ap.add_argument("--rebind-eval", action="store_true",
                    help="post-mortem rebinding control: also eval with table values cyclically "
                         "deranged in the input, scored vs the original answer")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    for K in args.ks:
        steps = args.steps if args.steps else 2000 + 125 * K  # K8=3000, K16=4000, K32=6000
        for label in args.models:
            for seed in args.seeds:
                out = OUTDIR / f"{label}_k{K}_d{args.d_model}l{args.layers}{args.tag}_seed{seed}.json"
                if out.exists():
                    print(f"SKIP {label} K{K} seed{seed}", flush=True)
                    continue
                print(f"RUN {label} K{K} seed{seed} steps{steps}", flush=True)
                try:
                    r = run_trial(label, K, seed, d_model=args.d_model, layers=args.layers,
                                  heads=args.heads, vocab=args.vocab, steps=steps, lr=args.lr,
                                  batch=args.batch, eval_batch=args.eval_batch,
                                  eval_iters=args.eval_iters, device=device,
                                  mamba2_d_state=args.mamba2_d_state, fp32=args.fp32,
                                  rebind_eval=args.rebind_eval)
                    r["git"] = git_metadata()
                    out.write_text(json.dumps(r, indent=2) + "\n")
                    print(f"DONE {label} K{K} seed{seed} acc={r['eval_accuracy']:.4f} "
                          f"loss={r['final_loss']:.4f} tok/s={r['tok_s']}", flush=True)
                except RuntimeError as e:
                    print(f"FAIL {label} K{K} seed{seed} -> {str(e)[:100]}", flush=True)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
