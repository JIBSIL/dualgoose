# DualGoose

Code and data for **"Anatomy of Associative Recall in Fixed-State Recurrences:
A Matched-State Decomposition, an Interference Wall, and a Curriculum That
Breaks It"** (J. Boesch & A. Wee, Obit Development — preprint of preliminary
results) and its companion, **"DreamingGoose: Staged Distillation from
Autoregressive Transformers to Bidirectional Recurrent Diffusion Language
Models."** Both PDFs are in [`paper/`](paper/).

> **Why "DualGoose"?** Lineage, not description: the original design was a
> paired (dual) forward/backward RWKV-7 — "Goose" is RWKV-7's codename —
> denoiser. The ablations (§4 of the paper) settled the headline cell on armed
> Gated DeltaNet — the armed delta-rule families are statistically
> indistinguishable there — and the companion DreamingGoose conversions are
> gated delta-rule students. The name marks where this started.

## What the paper does

Modern recurrent cells (Gated DeltaNet, RWKV-7, Mamba-2) each bundle several
mechanisms. The paper toggles them **one knob at a time at a fixed recurrent-state
budget** — short causal convolution × transition structure (rank-1 delta rule vs.
diagonal) × decay — and measures what each contributes to associative recall.
Findings: the convolution is the dominant lever; the rank-1 margin is conditional
on the convolution's absence; no rank-1-vs-diagonal *class* claim survives; a
distance wall that looks architectural is a training-coverage gap a distance
curriculum removes; and training in this regime is a **lock-in lottery** whose
rate the curriculum's shape controls.

## The harness

- **Kernel-free**: every cell is pure PyTorch; the Mamba-2 comparator implements
  the SSD recurrence explicitly and needs neither Triton nor `mamba_ssm`.
- **Runs anywhere**: the headline factorial was run on a single GTX 1070
  (pre-Volta support via an atomic-free two-pass Triton backward and `--fp32`).
- **Floors and controls built in**: every task ships floor calculators
  (value-marginal, no-binding, table, positional) and a **rebinding control**
  that deranges key–value bindings after training to verify genuine binding.

## Reproduce

```bash
pip install -e .            # torch is the only heavy dependency

# Constrained-state factorial (paper Tables 1–2)
python scripts/mqar_probe.py --models armed_gdn gated_deltanet deltanet \
    dg_rank_ref dg_diag_ref mamba2_ref mamba2_ref_noconv attention \
    --d-model 32 --layers 1 --mamba2-d-state 16 --ks 8 16 32 --seeds 0 1 2 --fp32

# Hardened 10-seed transition ablation with rebinding control (Table 3)
python scripts/dg_c9_harden.py --ks 16 32 --seeds 0 1 2 3 4 5 6 7 8 9

# Haystack wall / curriculum / collision-key discriminator (Tables 4–5)
python scripts/long_retrieval_probe.py --models armed_gdn armed_gdn_fwd attention \
    --seq-lens 64 --pairs 4 --seeds 0 --steps 3000 --batch 256 --layers 2 --heads 2 \
    --dense 4 --curriculum --fp32          # add --collision for the discriminator

# S5 state-tracking guardrail (Table 7)
python scripts/state_tracking_probe.py --models armed_gdn gated_deltanet \
    --group-n 5 --ls 8 16 --d-model 128 --seeds 0 1 2 3 4 5 6 7 8 9 --fp32
```

Drop `--fp32` on Ampere or newer. Exact commands for every table are in the
paper's Appendix B.

## Results

`results/` holds the JSON behind every table: `mqar_probe/` (factorial and
hardened ablations), `long_retrieval/` (wall, curriculum, collision), and
`state_tracking/` (S5 guardrail). Each file records the full config, per-run
metrics, floors, and the rebinding-control accuracy.

## Repo layout

```
dualgoose/   the package: cells, tasks, training loop, Triton scan
scripts/     the four probe entry points used in the paper
tests/       pytest suite for the package
results/     result JSONs for every table
paper/       LaTeX sources, style files, figures, and both PDFs
```

## Citation

```bibtex
@article{boesch2026anatomy,
  title   = {Anatomy of Associative Recall in Fixed-State Recurrences:
             A Matched-State Decomposition, an Interference Wall, and a
             Curriculum That Breaks It},
  author  = {Boesch, Julian and Wee, Andrew},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT (code and results). Papers are preprints of preliminary results, shared for
discussion.
