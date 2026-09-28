# DualGoose

Code, results and papers for two preprints about attention-free language-model layers:
recurrent cells such as Gated DeltaNet, RWKV-7 and Mamba-2, which keep a fixed-size
memory ("state") instead of attending over the whole context.

| Paper | Question | PDF |
|---|---|---|
| Anatomy of Associative Recall in Fixed-State Recurrences | Why are recurrent layers worse than attention at recalling key-value pairs from their context, and what fixes it? | [arXiv:2609.16183](https://arxiv.org/abs/2609.16183), [`paper/main.pdf`](paper/main.pdf) |
| DreamingGoose | Can a pretrained Transformer be converted into an attention-free, bidirectional diffusion language model, and what survives the conversion? | [`paper/dreaminggoose.pdf`](paper/dreaminggoose.pdf) |

By Julian Boesch and Andrew Wee (Purdue University and Obit Research); DreamingGoose is
also by Alexander Stranzl (State University of New York at Stony Brook). Both papers are
preprints of preliminary results, shared for discussion.

The code here reproduces the Anatomy paper. The DreamingGoose conversion code is not yet
released.

## Anatomy of Associative Recall

Associative recall means storing key-value pairs from the context and returning the right
value when a key is asked for. Recurrent layers are known to do this worse than attention,
but most comparisons pit whole architectures against each other, so it is unclear which
ingredient causes the gap. The paper holds the state size fixed and changes one ingredient
at a time. What it finds:

- The short convolution in front of the recurrence is the biggest lever, worth about +0.5
  recall. Comparisons between convolution-free cells and Mamba, which has one, mostly
  measure the convolution.
- The type of state update matters much less. A rank-1 "delta rule" update beats a
  diagonal one by +0.19 and +0.32 recall at 16 and 32 pairs, but only by +0.03 once both
  have the convolution, and a state-matched Mamba-2 ties the plain rank-1 cell. No
  architecture family is simply better at recall.
- Cells that recall 32 pairs still fail, at chance, to retrieve 4 pairs across a long
  stretch of distractor text. This is a training problem, not a capacity limit: a distance
  curriculum (training on gradually longer gaps) takes the same model from 0.021 to 1.000.
- Whether a training run learns to retrieve is close to a coin flip, and the curriculum's
  shape changes the odds: from 1/10 to 7/10 seeds, and to 6/6 at sequence length 512 when
  the curriculum only advances while measured accuracy holds.
- Equipping a cell for recall costs nothing on state tracking: on an S5 permutation task
  the recall-equipped cell is significantly better at every depth.

## DreamingGoose

DreamingGoose converts Qwen3 Transformers (1.7B and 8B) into attention-free, bidirectional
diffusion language models in three stages, so that each capability can be traced to the
stage where it was kept or lost. Language modeling transfers; in-context retrieval does
not (0.000 on a recall probe where the teacher scores about 0.34). Porting the Anatomy
paper's curriculum, advanced only while measured accuracy holds, restores retrieval in 3 of
3 seeds at 1.7B and 2 of 3 at 8B. One limit survives every intervention: the learned
retrieval does not extend to tokens that never appeared in retrieval training. The paper
also reports a 7B block-diffusion conversion and two negative results from training it.

## What's in this repository

| Path | Contents |
|---|---|
| `dualgoose/` | The Python package: recurrent cells, synthetic tasks, training loop, optional Triton scan |
| `scripts/` | The four experiment scripts used in the Anatomy paper |
| `results/` | One JSON file per run, behind every table in the Anatomy paper |
| `tests/` | pytest suite for the package |
| `paper/` | LaTeX sources, figures, PDFs and arXiv source bundles for both papers |

## Setup

Python 3.10 or newer. PyTorch is the only heavy dependency.

```bash
pip install -e .
pytest              # optional: run the test suite
```

Every cell has a pure-PyTorch reference implementation, including a state-matched Mamba-2
that needs neither `mamba_ssm` nor custom kernels. An optional Triton scan for the
RWKV-7-style cells also runs on GPUs older than Volta, about 26x faster than the reference
on a Pascal card; the paper's main experiments ran on a single GTX 1070.

## Reproducing the Anatomy paper

| Paper table | Experiment | Script |
|---|---|---|
| Tables 1-2 | Recall at a fixed state size, across ingredient combinations | `scripts/mqar_probe.py` |
| Table 3 | 10-seed ablation of the state update, with the rebinding control | `scripts/dg_c9_harden.py` |
| Tables 4-6 | Retrieval across a distractor haystack, the curriculum, collision keys | `scripts/long_retrieval_probe.py` |
| Table 7 | S5 state tracking | `scripts/state_tracking_probe.py` |

```bash
# Tables 1-2
python scripts/mqar_probe.py --models armed_gdn gated_deltanet deltanet \
    dg_rank_ref dg_diag_ref mamba2_ref mamba2_ref_noconv attention \
    --d-model 32 --layers 1 --mamba2-d-state 16 --ks 8 16 32 --seeds 0 1 2 --fp32

# Table 3
python scripts/dg_c9_harden.py --ks 16 32 --seeds 0 1 2 3 4 5 6 7 8 9

# Tables 4-6 (add --collision for the collision-key discriminator)
python scripts/long_retrieval_probe.py --models armed_gdn armed_gdn_fwd attention \
    --seq-lens 64 --pairs 4 --seeds 0 --steps 3000 --batch 256 --layers 2 --heads 2 \
    --dense 4 --curriculum --fp32

# Table 7
python scripts/state_tracking_probe.py --models armed_gdn gated_deltanet \
    --group-n 5 --ls 8 16 --d-model 128 --seeds 0 1 2 3 4 5 6 7 8 9 --fp32
```

`--fp32` is for GPUs without bf16 support; drop it on Ampere or newer. Appendix B of the
paper gives the exact command for every table.

### Result files

`results/` holds one JSON file per run: `mqar_probe/` for Tables 1-3, `long_retrieval/`
for Tables 4-6 and `state_tracking/` for Table 7. Each file records the full configuration,
the per-run metrics, the chance-level floors, and the rebinding-control score (a check that
the model really binds keys to values instead of guessing likely values).

## Why "DualGoose"?

The project began as a denoiser built from two RWKV-7 layers ("Goose" is RWKV-7's
codename), one reading the sequence forward and one backward. The experiments later settled
on Gated DeltaNet cells (the delta-rule variants tie in the paper's tests), and the
DreamingGoose students use them too. The name records where the project started.

## Citation

```bibtex
@article{boesch2026anatomy,
  title   = {Anatomy of Associative Recall in Fixed-State Recurrences:
             A Matched-State Decomposition, an Interference Wall, and a
             Curriculum That Breaks It},
  author  = {Boesch, Julian and Wee, Andrew},
  journal = {arXiv preprint arXiv:2609.16183},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.16183}
}
```

## License

MIT for code and results. The papers are preprints of preliminary results, shared for
discussion.
