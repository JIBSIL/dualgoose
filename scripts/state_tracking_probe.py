"""State-tracking probe (C7): can the rank-1 RWKV-7 recurrence track a non-commutative group
product where a diagonal SSM / Mamba-2 / attention cannot?

Task: the S_n word problem. The input is a sequence of L random permutations g_1..g_L from S_n
(each a token); the masked answer slot must be filled with the running product g_1∘...∘g_L.
Tracking that product requires representing non-abelian state. S_3 is solvable (its word problem
is in TC^0, so all architectures should handle it) — a sanity control. S_5 is the smallest
*non-solvable* group: its word problem is NC^1-complete, provably outside TC^0, so Transformers
and diagonal SSMs cannot compute it for growing L, while RWKV-7's diagonal-plus-rank-1 transition
(eigenvalues outside [0,1]) can. The discriminating signal is S_5 accuracy vs L.

Same masked-diffusion harness as the MQAR probe (only the answer slot masked, t=0.5). Self-contained;
reuses build_tiny_denoiser / masked_nll_loss / apply_subs_sample. Run in .venv-mamba. Per-trial
JSON outputs under results/state_tracking/ (resumable).
"""

import argparse
import contextlib
import itertools
import json
import time
from pathlib import Path

import torch

from dualgoose.diffusion import apply_subs_sample, masked_nll_loss
from dualgoose.model import build_tiny_denoiser

OUTDIR = Path("results/state_tracking")

MODELS = {
    # state-tracking (running prefix product) is causal, so the DualGoose arms use the
    # forward scan — a bidirectional merge would average in the backward scan's suffix
    # products, which are the wrong quantity and dilute the signal.
    "dg_rank": dict(model_type="dualgoose", use_rank=True, scan_backend="triton",
                    merge="static", direction="forward"),
    "dg_diag": dict(model_type="dualgoose", use_rank=False, scan_backend="triton",
                    merge="static", direction="forward"),
    "mamba2": dict(model_type="mamba2"),
    "attention": dict(model_type="attention"),
    # the other beyond-TC^0 rank-1 recurrence; forward (causal) to match the dg arms.
    "deltanet": dict(model_type="deltanet", merge="static", direction="forward"),
    "deltanet_linq": dict(model_type="deltanet_linq", merge="static", direction="forward"),
    "gated_deltanet": dict(model_type="gated_deltanet", merge="static", direction="forward"),
    # armed-cell guardrail (docs/armed_gdn_experiment.md): does adding Mamba-2's short causal
    # conv to the rank-1 cell COST its beyond-TC^0 depth-robustness? Compare armed_gdn vs
    # gated_deltanet at S5 L={2,4,8,16}. mamba2_noconv isolates the conv's share on the diagonal.
    "armed_gdn": dict(model_type="gated_deltanet", short_conv=4,
                      merge="static", direction="forward"),
    "mamba2_noconv": dict(model_type="mamba2", mamba2_d_conv=1),
    # pure-PyTorch fallback (always builds, incl. V100 / no mamba_ssm) — see docs/armed_gdn_experiment.md
    "mamba_local": dict(model_type="mamba", mamba_kernel_size=3),
    "mamba_local_noconv": dict(model_type="mamba", mamba_kernel_size=1),
    # Pascal / GTX 1070-native (no Triton / mamba_ssm): reference RWKV scan + pure-PyTorch
    # state-matched Mamba-2 (at the default d=128, d_state=64 matches the rank-1 cells' 128^2).
    "dg_rank_ref": dict(model_type="dualgoose", use_rank=True, scan_backend="reference",
                        merge="static", direction="forward"),
    "dg_diag_ref": dict(model_type="dualgoose", use_rank=False, scan_backend="reference",
                        merge="static", direction="forward"),
    "mamba2_ref": dict(model_type="mamba2_ref"),
    "mamba2_ref_noconv": dict(model_type="mamba2_ref", mamba2_d_conv=1),
    # armed RWKV-7 (rank-1 + vector decay + conv), forward like the other S5 arms — cross-family
    # check of the GDN-only "arming helps state-tracking" guardrail result
    "armed_rwkv": dict(model_type="dualgoose", use_rank=True, scan_backend="reference",
                       short_conv=4, merge="static", direction="forward"),
}

# model_types whose recurrent state is matched via --mamba2-d-state (d_inner * d_state)
_STATE_MATCHED = {"mamba2", "mamba2_ref"}


def amp_ctx(fp32):
    """bf16 autocast, or a no-op for fp32 (GPUs without hardware bf16, e.g. Pascal / GTX 1070)."""
    return contextlib.nullcontext() if fp32 else torch.autocast("cuda", dtype=torch.bfloat16)


def group_table(n):
    """Return (perms, mul, identity_index) for S_n. mul[i,j] = index of perm_i ∘ perm_j."""
    perms = list(itertools.permutations(range(n)))
    index = {p: i for i, p in enumerate(perms)}
    G = len(perms)
    mul = [[0] * G for _ in range(G)]
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            comp = tuple(a[b[k]] for k in range(n))  # apply b then a
            mul[i][j] = index[comp]
    identity = index[tuple(range(n))]
    return perms, torch.tensor(mul, dtype=torch.long), identity


def generator_indices(n):
    """Standard generating set of S_n: a transposition (0 1) and the n-cycle (0 1 ... n-1),
    plus the identity (a no-op step). Returns their indices in the perms list."""
    perms = list(itertools.permutations(range(n)))
    index = {p: i for i, p in enumerate(perms)}
    ident = tuple(range(n))
    transposition = (1, 0) + tuple(range(2, n))
    cycle = tuple((k + 1) % n for k in range(n))
    return [index[ident], index[transposition], index[cycle]]


def make_batch(batch, L, mul, identity, mask_id, device, gen, gens=None):
    """Interleaved dense-supervision layout: positions [g_1, p_1, g_2, p_2, ..., g_L, p_L],
    seq_len = 2L. Generators g_i are visible; every running-product slot p_i = g_1∘...∘g_i is
    masked and supervised. The model must compose the prefix to fill each p_i (it cannot see
    neighbouring products, all of which are masked), so this is genuine running state-tracking
    with L targets per sequence rather than one."""
    G = mul.shape[0]
    seq_len = 2 * L
    mul = mul.to(device)
    if gens is not None:
        pick = torch.randint(0, len(gens), (batch, L), device=device, generator=gen)
        g = gens.to(device)[pick]  # sample from the generating set
    else:
        g = torch.randint(0, G, (batch, L), device=device, generator=gen)
    clean = torch.empty(batch, seq_len, dtype=torch.long, device=device)
    cur = torch.full((batch,), identity, dtype=torch.long, device=device)
    for i in range(L):
        cur = mul[cur, g[:, i]]
        clean[:, 2 * i] = g[:, i]
        clean[:, 2 * i + 1] = cur
    corrupted = clean.clone()
    target = torch.zeros_like(clean, dtype=torch.bool)
    target[:, 1::2] = True  # every running-product slot
    corrupted[target] = mask_id
    t = torch.full((batch,), 0.5, device=device)

    class B:  # ToyBatch-like
        pass
    b = B()
    b.clean, b.corrupted, b.target_mask, b.t = clean, corrupted, target, t
    return b


def run_trial(label, group_n, L, seed, *, d_model, layers, heads, steps, lr, batch,
              eval_batch, eval_iters, device, use_generators=False, mamba2_d_state=None,
              fp32=False):
    perms, mul, identity = group_table(group_n)
    G = len(perms)
    gens = torch.tensor(generator_indices(group_n)) if use_generators else None
    vocab = G + 1  # group elements 0..G-1 plus a mask token
    mask_id = vocab - 1
    seq_len = 2 * L
    torch.manual_seed(seed)
    kw = dict(MODELS[label])
    if kw.get("model_type") in _STATE_MATCHED and mamba2_d_state is not None:
        kw["mamba2_d_state"] = mamba2_d_state
    model = build_tiny_denoiser(vocab, seq_len, d_model=d_model, n_layers=layers,
                                n_heads=heads, **kw).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    last = float("nan")
    for _ in range(steps):
        b = make_batch(batch, L, mul, identity, mask_id, device, gen, gens)
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
    correct = total = 0
    depth_correct = torch.zeros(L)  # accuracy at each running-product depth p_1..p_L
    depth_total = torch.zeros(L)
    with torch.no_grad():
        for _ in range(eval_iters):
            b = make_batch(eval_batch, L, mul, identity, mask_id, device, egen, gens)
            with amp_ctx(fp32):
                logits = model(b.corrupted, b.t)
            pred = apply_subs_sample(logits, b.corrupted, mask_id)
            m = b.target_mask
            correct += int(pred[m].eq(b.clean[m]).sum().cpu())
            total += int(m.sum().cpu())
            ok = pred.eq(b.clean)  # [batch, 2L]
            for i in range(L):
                col = ok[:, 2 * i + 1]
                depth_correct[i] += float(col.sum().cpu())
                depth_total[i] += col.numel()
    depth_acc = (depth_correct / depth_total.clamp_min(1)).tolist()
    return {
        "label": label, "group": f"S{group_n}", "G": G, "L": L, "seed": seed,
        "d_model": d_model, "layers": layers, "params": nparams, "steps": steps,
        "amp": "fp32" if fp32 else "bf16",
        "mamba2_d_state": mamba2_d_state if MODELS[label].get("model_type") in _STATE_MATCHED else None,
        "chance": 1.0 / G, "generators": bool(use_generators), "final_loss": last, "eval_accuracy": correct / max(1, total),
        "depth_accuracy": [round(a, 4) for a in depth_acc],  # acc at product p_1..p_L
        "last_depth_accuracy": round(depth_acc[-1], 4),  # hardest: full product p_L
        "tok_s": round(batch * seq_len * steps / (time.time() - t0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--group-n", type=int, default=5)
    ap.add_argument("--ls", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--eval-iters", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--generators", action="store_true",
                    help="sample inputs from a small generating set instead of full S_n")
    ap.add_argument("--mamba2-d-state", type=int, default=None,
                    help="override Mamba-2/mamba2_ref d_state for state-matched comparisons")
    ap.add_argument("--fp32", action="store_true",
                    help="disable bf16 autocast (for GPUs without hardware bf16, e.g. Pascal / GTX 1070)")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    for L in args.ls:
        for label in args.models:
            for seed in args.seeds:
                out = OUTDIR / f"{label}_S{args.group_n}_L{L}{args.tag}_seed{seed}.json"
                if out.exists():
                    print(f"SKIP {label} S{args.group_n} L{L} seed{seed}", flush=True)
                    continue
                print(f"RUN {label} S{args.group_n} L{L} seed{seed} steps{args.steps}", flush=True)
                try:
                    r = run_trial(label, args.group_n, L, seed, d_model=args.d_model,
                                  layers=args.layers, heads=args.heads, steps=args.steps,
                                  lr=args.lr, batch=args.batch, eval_batch=args.eval_batch,
                                  eval_iters=args.eval_iters, device=device,
                                  use_generators=args.generators,
                                  mamba2_d_state=args.mamba2_d_state, fp32=args.fp32)
                    out.write_text(json.dumps(r, indent=2) + "\n")
                    print(f"DONE {label} S{args.group_n} L{L} seed{seed} acc={r['eval_accuracy']:.4f} "
                          f"(chance={r['chance']:.4f}) loss={r['final_loss']:.4f} tok/s={r['tok_s']}",
                          flush=True)
                except RuntimeError as e:
                    print(f"FAIL {label} S{args.group_n} L{L} seed{seed} -> {str(e)[:100]}", flush=True)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
