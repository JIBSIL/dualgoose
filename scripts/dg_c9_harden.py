"""C9 HARDENED — is "rank-1 recurrence class > diagonal recurrence class" on masked MQAR real?

Motivation. C7's mechanism claim ("rank-1 beats the diagonal ablation on S_n state-tracking") evaporated when
re-run at 10 seeds: the published 3-seed `mean ± std` hid a bimodal lock-in with std ±0.40 on a [0,1] metric
(`docs/c7_hardened_result.md`). C9 is the *other* rank-1-vs-diagonal claim and it now carries the mechanism
story alone, so it gets the identical treatment before anyone leans on it:

  1. 10 seeds, not 3          -- the published dg_diag K=16 cell is 0.764 ± 0.143, i.e. already seed-unstable.
  2. lock-in rate + a permutation test on the mean gap, not mean±std over a possibly-bimodal outcome.
  3. a BASELINE-FREE control  -- the one thing C7's original run lacked.
  4. the CORRECT floors       -- not "1/vocab".

Config is matched to the published C9 command (`docs/mqar_capacity_rented.md`):
    python scripts/mqar_probe.py --d-model 32 --layers 1     # constrained state, d=32, N=1
i.e. the *constrained-state* regime, which is the only one where the arms separate (at d=128 everything solves).

## The control: REBINDING (why, and why not "mismatch")
Train on `true`. Then evaluate the SAME trained model on:

  true      : table shows (k_i -> v_i); query k_sel; target v_sel.
  shuffled  : table shows (k_i -> v_{perm(i)}) for a DERANGEMENT perm; query k_sel; target is still the
              ORIGINAL v_sel, which now sits next to a different key.
              -> a model that genuinely binds key->value must collapse to ~0 (it will emit v_{perm(sel)}).
              -> a model that merely emits "some salient value from the table" is unaffected and stays at 1/K.
              This dissociates BINDING from SALIENCE, and needs no baseline at all.

There is no `mismatch` (absent-key) condition here, and that is a property of the toy builder, not an
oversight: `make_mqar_batch` puts *every* key 5..5+K-1 in the table by construction, so no key is ever absent
to query. Rebinding subsumes it -- it is the stronger control, because it holds the token multiset fixed and
perturbs only the binding.

## Floors (the published probe compared against none of these)
  table_floor = 1/K            -- guess uniformly among the K values actually present. THE relevant floor.
  marginal    = 1/|values|     -- values are drawn uniformly from [5+K, vocab-1), so this is exact.
A model can sit at table_floor with zero binding ability, so `true` alone proves nothing; `true` high AND
`shuffled` at/below table_floor is what proves retrieval.

Usage:
  python scripts/dg_c9_harden.py --ks 16 32 --seeds 0 1 2 3 4 5 6 7 8 9        # matched: d=32, layers=1
  python scripts/dg_c9_harden.py --summarize-only
"""
import argparse, json, time, itertools, random
from pathlib import Path

import torch

from dualgoose.diffusion import apply_subs_sample, masked_nll_loss
from dualgoose.model import build_tiny_denoiser

OUTDIR = Path("results/c9_hardened")

ARMS = {
    "dg_rank":     dict(model_type="dualgoose", use_rank=True,  scan_backend="triton",
                        merge="static", direction="bidirectional"),
    "dg_diag":     dict(model_type="dualgoose", use_rank=False, scan_backend="triton",
                        merge="static", direction="bidirectional"),
    "deltanet":    dict(model_type="deltanet", merge="static", direction="bidirectional"),
    # state-matched Mamba-2: d_inner(64) x d_state(16) = 1024 elems ~= DualGoose head state at d=32
    "mamba2_ds16": dict(model_type="mamba2", mamba2_d_state=16),
}
LOCKIN = 0.90   # a seed "solves" MQAR if recall >= 0.90


def build(arm, vocab, seq_len, d_model, layers, heads):
    kw = dict(ARMS[arm])
    return build_tiny_denoiser(vocab, seq_len, d_model=d_model, n_layers=layers, n_heads=heads, **kw)


def _derangement(B, K, device, gen):
    """Random derangement per row (no fixed points), by rejection then a cyclic-shift fallback."""
    perm = torch.argsort(torch.rand(B, K, device=device, generator=gen), dim=1)
    ar = torch.arange(K, device=device).unsqueeze(0)
    for _ in range(64):
        bad = perm.eq(ar).any(dim=1)
        if not bool(bad.any()):
            return perm
        n = int(bad.sum())
        perm[bad] = torch.argsort(torch.rand(n, K, device=device, generator=gen), dim=1)
    bad = perm.eq(ar).any(dim=1)                       # fallback: guaranteed-derangement cyclic shift
    if bool(bad.any()):
        n = int(bad.sum())
        shift = torch.randint(1, K, (n, 1), device=device, generator=gen)
        perm[bad] = (ar.expand(n, K) + shift) % K
    return perm


def make_batch(B, seq_len, vocab, mask_id, K, device, gen, mode):
    """`after_table` layout, matched to mqar_probe.run_trial, plus the rebinding control.

    layout:  [k0 v0 k1 v1 ... k_{K-1} v_{K-1}]  k_sel  <MASK>  <random filler>
    keys are the fixed tokens 5..5+K-1; values are drawn from [5+K, vocab-1).
    """
    clean = torch.randint(5, vocab - 1, (B, seq_len), device=device, generator=gen)
    rows = torch.arange(B, device=device)
    vals = torch.randint(5 + K, vocab - 1, (B, K), device=device, generator=gen)   # v_i, the TRUE bindings

    perm = torch.arange(K, device=device).unsqueeze(0).expand(B, K) if mode == "true" \
        else _derangement(B, K, device, gen)
    shown = torch.gather(vals, 1, perm)               # slot i displays v_{perm(i)}

    for i in range(K):
        clean[:, 2 * i] = 5 + i                       # key token
        clean[:, 2 * i + 1] = shown[:, i]             # displayed value

    sel = torch.randint(0, K, (B,), device=device, generator=gen)
    q, a = 2 * K, 2 * K + 1
    clean[rows, q] = 5 + sel
    clean[rows, a] = vals[rows, sel]                  # target = ORIGINAL binding of the queried key

    corrupted = clean.clone()
    corrupted[rows, a] = mask_id
    tmask = torch.zeros_like(clean, dtype=torch.bool)
    tmask[rows, a] = True
    return clean, corrupted, tmask, torch.full((B,), 0.5, device=device)


@torch.no_grad()
def evaluate(model, mode, *, K, seq_len, vocab, mask_id, device, seed, batch, iters):
    gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    hit = tot = 0
    for _ in range(iters):
        clean, corrupted, tmask, t = make_batch(batch, seq_len, vocab, mask_id, K, device, gen, mode)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(corrupted, t)
        pred = apply_subs_sample(logits, corrupted, mask_id)
        hit += int(pred[tmask].eq(clean[tmask]).sum().cpu())
        tot += int(tmask.sum().cpu())
    return hit / max(1, tot)


def run_trial(arm, K, seed, *, d_model, layers, heads, vocab, steps, lr, batch,
              eval_batch, eval_iters, device):
    seq_len = 2 * K + 8
    mask_id = vocab - 1
    torch.manual_seed(seed)
    model = build(arm, vocab, seq_len, d_model, layers, heads).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)
    gen = torch.Generator(device=device).manual_seed(seed)
    t0, last = time.time(), float("nan")
    for _ in range(steps):
        clean, corrupted, _, t = make_batch(batch, seq_len, vocab, mask_id, K, device, gen, "true")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(corrupted, t)
            loss = masked_nll_loss(logits, clean, corrupted, mask_id, t=t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        last = float(loss.detach().cpu())
    model.eval()
    ev = dict(K=K, seq_len=seq_len, vocab=vocab, mask_id=mask_id, device=device,
              seed=seed, batch=eval_batch, iters=eval_iters)
    acc_true = evaluate(model, "true", **ev)
    acc_shuf = evaluate(model, "shuffled", **ev)
    n_values = (vocab - 1) - (5 + K)
    return {
        "arm": arm, "K": K, "seed": seed, "seq_len": seq_len, "params": nparams,
        "d_model": d_model, "layers": layers, "steps": steps, "lr": lr, "batch": batch,
        "final_loss": last, "acc_true": acc_true, "acc_shuffled": acc_shuf,
        "shuffle_drop": acc_true - acc_shuf,
        "table_floor": 1.0 / K, "marginal": 1.0 / n_values,
        "locked_in": acc_true >= LOCKIN,
        "secs": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------- stats
def perm_test(a, b, iters=20000, seed=0):
    """Two-sided permutation test on the difference of means. No scipy needed."""
    rng = random.Random(seed)
    obs = abs(sum(a) / len(a) - sum(b) / len(b))
    pool, na, ge = list(a) + list(b), len(a), 0
    for _ in range(iters):
        rng.shuffle(pool)
        if abs(sum(pool[:na]) / na - sum(pool[na:]) / (len(pool) - na)) >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (iters + 1)


def boot_ci(x, iters=10000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choices(x, k=len(x))) / len(x) for _ in range(iters))
    return m[int(0.025 * iters)], m[int(0.975 * iters)]


def summarize(rows):
    out = []
    ks = sorted({r["K"] for r in rows})
    print(f"\n{'='*104}\nC9 HARDENED — masked MQAR at constrained state (d=32, N=1), "
          f"{len({r['seed'] for r in rows})} seeds\n{'='*104}")
    for K in ks:
        sub = [r for r in rows if r["K"] == K]
        floor = 1.0 / K
        marg = sub[0]["marginal"]
        print(f"\n--- K={K}   table_floor=1/K={floor:.3f}   marginal=1/|values|={marg:.3f} ---")
        print(f"{'arm':<12} {'true mean [95% CI]':<28} {'lock-in':<9} {'shuffled':<10} {'drop':<8} {'min..max'}")
        per = {}
        for arm in ARMS:
            a = [r for r in sub if r["arm"] == arm]
            if not a:
                continue
            t = [r["acc_true"] for r in a]
            s = [r["acc_shuffled"] for r in a]
            per[arm] = t
            lo, hi = boot_ci(t)
            li = sum(r["locked_in"] for r in a)
            print(f"{arm:<12} {sum(t)/len(t):.3f} [{lo:.3f},{hi:.3f}]{'':<8} "
                  f"{li:>2}/{len(a):<6} {sum(s)/len(s):.3f}{'':<5} "
                  f"{sum(t)/len(t)-sum(s)/len(s):+.3f}   {min(t):.3f}..{max(t):.3f}")
            out.append(dict(K=K, arm=arm, n=len(a), mean_true=sum(t)/len(t), ci=[lo, hi],
                            lockin=li, mean_shuffled=sum(s)/len(s),
                            drop=sum(t)/len(t)-sum(s)/len(s), min=min(t), max=max(t)))
        print("  pairwise (permutation test on mean gap, two-sided):")
        for x, y in itertools.combinations([a for a in ARMS if a in per], 2):
            p = perm_test(per[x], per[y])
            star = "  <-- significant" if p < 0.05 else ""
            print(f"    {x:<12} vs {y:<12} gap={sum(per[x])/len(per[x]) - sum(per[y])/len(per[y]):+.3f}  p={p:.4f}{star}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--ks", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--d-model", type=int, default=32)     # matched to the published C9 command
    ap.add_argument("--layers", type=int, default=1)       # matched
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--eval-iters", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.selftest:
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(0)
        K, V, SL = 8, 64, 24
        for mode in ("true", "shuffled"):
            clean, corr, tm, _ = make_batch(64, SL, V, V - 1, K, dev, g, mode)
            rows = torch.arange(64, device=dev)
            q = clean[rows, 2 * K] - 5                      # queried key index
            shown_at_q = clean[rows, 2 * q + 1]             # value displayed next to the queried key
            tgt = clean[tm]
            same = int(shown_at_q.eq(tgt).sum())
            print(f"{mode:>9}: target == value shown beside queried key in {same}/64 rows")
            if mode == "true":
                assert same == 64, "true mode must display the target next to its key"
            else:
                # derangement => the displayed value differs, EXCEPT when two values happened to be equal
                assert same < 64 * 0.25, f"shuffled mode still shows the target too often ({same}/64)"
        print("selftest PASS — rebinding control genuinely moves the answer away from the queried key")
        return

    rows = []
    if not args.summarize_only:
        device = torch.device("cuda")
        for K in args.ks:
            steps = args.steps or (2000 + 125 * K)     # published schedule: K16=4000, K32=6000
            for arm in args.arms:
                for seed in args.seeds:
                    out = OUTDIR / f"{arm}_k{K}_d{args.d_model}l{args.layers}{args.tag}_seed{seed}.json"
                    if out.exists():
                        rows.append(json.loads(out.read_text())); continue
                    r = run_trial(arm, K, seed, d_model=args.d_model, layers=args.layers,
                                  heads=args.heads, vocab=args.vocab, steps=steps, lr=args.lr,
                                  batch=args.batch, eval_batch=args.eval_batch,
                                  eval_iters=args.eval_iters, device=device)
                    out.write_text(json.dumps(r, indent=2) + "\n")
                    rows.append(r)
                    print(f"{arm:<12} K={K:<3} seed={seed}  true={r['acc_true']:.3f} "
                          f"shuf={r['acc_shuffled']:.3f} drop={r['shuffle_drop']:+.3f} "
                          f"({r['secs']}s)", flush=True)
    else:
        rows = [json.loads(p.read_text()) for p in sorted(OUTDIR.glob(f"*{args.tag}_seed*.json"))]

    rows = [r for r in rows if r["K"] in args.ks]
    summ = summarize(rows)
    (OUTDIR / f"summary{args.tag}.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(f"\nwrote {OUTDIR}/summary{args.tag}.json")


if __name__ == "__main__":
    main()
