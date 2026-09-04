"""Long-context retrieval probe: recall vs CONTEXT LENGTH.

This is the regime where attention shines and fixed-state recurrences dip — the state-size recall
bound (Jelassi et al. "Repeat After Me"; Arora et al. "Zoology"), and the reason pure-recurrent LMs
have lagged on long retrieval. `pairs` key/value pairs are packed at the start, a long distractor
haystack follows, and the query sits at the end, so retrieval distance = seq_len. We sweep seq_len
for GDN (armed / unarmed), a state-matched Mamba-2 reference, RWKV-7 (reference scan), and full
attention (the ceiling), and read how each degrades with length.

Pascal-native (GTX 1070 / sm_61): use --fp32 and the reference cells (no Triton / mamba_ssm), exactly
like scripts/mqar_probe.py. Per-trial JSON under results/long_retrieval/ (idempotent / resumable).
"""

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch

from dualgoose.diffusion import apply_subs_sample, masked_nll_loss
from dualgoose.model import build_tiny_denoiser
from dualgoose.toy_data import make_long_retrieval_batch
from dualgoose.train_toy import git_metadata

OUTDIR = Path("results/long_retrieval")

MODELS = {
    "armed_gdn": dict(model_type="gated_deltanet", short_conv=4,
                      merge="static", direction="bidirectional"),
    "gated_deltanet": dict(model_type="gated_deltanet", merge="static", direction="bidirectional"),
    "mamba2_ref": dict(model_type="mamba2_ref"),            # state-matched via --mamba2-d-state
    "dg_rank_ref": dict(model_type="dualgoose", use_rank=True, scan_backend="reference",
                        merge="static", direction="bidirectional"),
    "attention": dict(model_type="attention"),             # long-context ceiling
    # position-generalizing attention (rotary; no absolute pos table) — the valid ceiling for
    # shifting-layout curriculum cells, where the absolute-pos toy attention collapses 7/7
    "attention_rope": dict(model_type="attention_rope"),
    # armed RWKV-7 (rank-1 + vector decay + conv; == mqar_probe's dg_rank_conv_ref cell) —
    # cross-family check of the GDN-only retrieval-dynamics results (lock-in, mark-and-route)
    "armed_rwkv": dict(model_type="dualgoose", use_rank=True, scan_backend="reference",
                       short_conv=4, merge="static", direction="bidirectional"),
    # JRT directional ablation: forward-only NEVER sees the query before the haystack, so if dense
    # supervision + curriculum lift the bidirectional arm but not this one, the win is query-first
    # reading (the built-in Just-Read-Twice of a bidirectional denoiser), not just denser gradient.
    "armed_gdn_fwd": dict(model_type="gated_deltanet", short_conv=4,
                          merge="static", direction="forward"),
}
# model_types whose recurrent state is matched via --mamba2-d-state (d_inner * d_state)
_STATE_MATCHED = {"mamba2", "mamba2_ref"}


def amp_ctx(fp32):
    """bf16 autocast, or a no-op for fp32 (GPUs without hardware bf16, e.g. Pascal / GTX 1070)."""
    return contextlib.nullcontext() if fp32 else torch.autocast("cuda", dtype=torch.bfloat16)


def build(label, vocab, seq_len, d_model, layers, heads, mamba2_d_state=None):
    kw = dict(MODELS[label])
    if kw.get("model_type") in _STATE_MATCHED and mamba2_d_state is not None:
        kw["mamba2_d_state"] = mamba2_d_state
    return build_tiny_denoiser(vocab, seq_len, d_model=d_model, n_layers=layers,
                               n_heads=heads, **kw)


def run_trial(label, seq_len, seed, *, pairs, d_model, layers, heads, vocab, steps, lr, batch,
              eval_batch, eval_iters, device, mamba2_d_state=None, fp32=False,
              n_queries=1, curriculum=False, collision=False, curriculum_shape="uniform",
              rebind_eval=False):
    mask_id = vocab - 1
    torch.manual_seed(seed)
    model = build(label, vocab, seq_len, d_model, layers, heads, mamba2_d_state).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)
    gen = torch.Generator(device=device).manual_seed(seed)
    gcpu = torch.Generator().manual_seed(seed + 555)  # cpu gen for per-batch curriculum gap
    max_gap = seq_len - 2 * pairs - 2 * n_queries
    t0 = time.time()
    last = float("nan")
    for step in range(steps):
        # curriculum: sample the retrieval gap each batch; eval is always at max.
        #   uniform: gap ~ U[0, max] (the original recipe — fails at L256, see GDN_PAPER scale note)
        #   ramp:    gap ~ U[0, cap] with cap growing linearly to max by 80% of training — the
        #            shaped curriculum that preserves an easy-first learning path at long L
        g = None
        if curriculum:
            cap = max_gap
            if curriculum_shape == "ramp":
                cap = max(1, min(max_gap, int(max_gap * (step + 1) / (0.8 * steps))))
            g = int(torch.randint(0, cap + 1, (1,), generator=gcpu))
        b = make_long_retrieval_batch(batch, seq_len, vocab, mask_id, pairs=pairs,
                                      n_queries=n_queries, gap=g, collision=collision,
                                      device=device, generator=gen)
        with amp_ctx(fp32):
            logits = model(b.corrupted, b.t)
            loss = masked_nll_loss(logits, b.clean, b.corrupted, mask_id, t=b.t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last = float(loss.detach().cpu())
    model.eval()
    egen = torch.Generator(device=device).manual_seed(seed + 10_000)
    correct = total = rebind_correct = 0
    # eval builds with gap=None (max) => table sits at position 0; value slots are odd offsets
    vpos = torch.tensor([2 * p + 1 for p in range(pairs)], device=device)
    with torch.no_grad():
        for _ in range(eval_iters):
            b = make_long_retrieval_batch(eval_batch, seq_len, vocab, mask_id, pairs=pairs,
                                          n_queries=n_queries, collision=collision,
                                          device=device, generator=egen)
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
        "label": label, "seq_len": seq_len, "pairs": pairs, "seed": seed,
        "n_queries": n_queries, "curriculum": bool(curriculum), "collision": bool(collision),
        "curriculum_shape": curriculum_shape if curriculum else None,
        "rebind_accuracy": rebind_acc,
        "value_marginal_floor": round(1.0 / (vocab - 1 - (5 + pairs)), 5),
        "d_model": d_model, "layers": layers, "params": nparams,
        "mamba2_d_state": mamba2_d_state if MODELS[label].get("model_type") in _STATE_MATCHED else None,
        "amp": "fp32" if fp32 else "bf16", "steps": steps, "lr": lr, "batch": batch,
        "final_loss": last, "eval_accuracy": acc, "eval_samples": total,
        "tok_s": round(batch * seq_len * steps / (time.time() - t0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seq-lens", nargs="+", type=int, default=[64, 128, 256, 512],
                    help="context lengths to sweep (retrieval distance)")
    ap.add_argument("--pairs", type=int, default=4, help="KV pairs (kept small: axis is DISTANCE)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--eval-iters", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--mamba2-d-state", type=int, default=16,
                    help="Mamba-2 d_state; 16 => 1024 state at d=32, matched to the rank-1 cells")
    ap.add_argument("--fp32", action="store_true",
                    help="disable bf16 autocast (Pascal / GTX 1070 has no hardware bf16)")
    ap.add_argument("--dense", type=int, default=1,
                    help="JRT dense supervision: number of distinct queried pairs per sequence "
                         "(all answers masked + supervised); 1 = original sparse probe")
    ap.add_argument("--curriculum", action="store_true",
                    help="JRT curriculum: sample the table->query gap uniformly per batch during "
                         "training (eval always at max gap)")
    ap.add_argument("--collision", action="store_true",
                    help="airtight JRT discriminator: haystack = decoy pairs reusing the table's "
                         "keys with garbage values (first binding wins); kills the lexical gate")
    ap.add_argument("--curriculum-shape", default="uniform", choices=["uniform", "ramp"],
                    help="ramp = max gap grows linearly to full by 80%% of training (shaped "
                         "curriculum; tests whether the L256 boundary is a curriculum-shape artifact)")
    ap.add_argument("--rebind-eval", action="store_true",
                    help="post-mortem rebinding control: also eval with table values cyclically "
                         "deranged in the input, scored vs the original answer")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    # seq_lens outer => short (cheap) lengths finish first, giving the near end of the curve early
    for L in args.seq_lens:
        for label in args.models:
            for seed in args.seeds:
                out = OUTDIR / f"{label}_L{L}p{args.pairs}_d{args.d_model}{args.tag}_seed{seed}.json"
                if out.exists():
                    print(f"SKIP {label} L{L} seed{seed}", flush=True)
                    continue
                print(f"RUN {label} L{L} p{args.pairs} seed{seed} steps{args.steps}", flush=True)
                try:
                    r = run_trial(label, L, seed, pairs=args.pairs, d_model=args.d_model,
                                  layers=args.layers, heads=args.heads, vocab=args.vocab,
                                  steps=args.steps, lr=args.lr, batch=args.batch,
                                  eval_batch=args.eval_batch, eval_iters=args.eval_iters,
                                  device=device, mamba2_d_state=args.mamba2_d_state, fp32=args.fp32,
                                  n_queries=args.dense, curriculum=args.curriculum,
                                  collision=args.collision, curriculum_shape=args.curriculum_shape,
                                  rebind_eval=args.rebind_eval)
                    r["git"] = git_metadata()
                    out.write_text(json.dumps(r, indent=2) + "\n")
                    print(f"DONE {label} L{L} seed{seed} acc={r['eval_accuracy']:.4f} "
                          f"loss={r['final_loss']:.4f} tok/s={r['tok_s']}", flush=True)
                except RuntimeError as e:
                    print(f"FAIL {label} L{L} seed{seed} -> {str(e)[:120]}", flush=True)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
