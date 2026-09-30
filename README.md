# With the Question Fixed

**Hint uptake and disclosure when a reasoning model is forced to think in another language**

Emrehan Dalaman. Term paper for *Advanced Topics in Computational Text and Media Sciences #2* (topic: Explainable AI),
Universität Trier, SoSe 2026. Lecturer: Raghvi Baloni, M.Sc.

**Paper:** [`paper/with-the-question-fixed.pdf`](paper/with-the-question-fixed.pdf)

## The question

A chain of thought (CoT) is only worth monitoring if it admits what shaped the answer. The usual test plants a hint in
a multiple-choice question and checks whether the CoT mentions it. Earlier cross-language studies changed the
question's language together with the thinking language. This study keeps 100 GPQA Diamond questions and three of
Chen et al.'s (2025) hints in English and changes only the language Qwen3.5-9B thinks in (English, Turkish or
Chinese), forced with a sentence prefilled at the start of its reasoning. In all, it analyses 4,800 traces, some of them
under a Turkish prompt.

## What it finds

- With the language named in the prefilled sentence, the thinking language barely changes whether the model follows
  the hint or mentions it. English and Turkish traces that follow the hint almost always mention it; Chinese ones
  mention it slightly less often.
- How the language is forced matters more. Naming the language in the prefix makes English traces over three times
  longer than a prefix that does not name it.
- The model drifts out of the forced language mostly after its answer is settled. A plain instruction to think in the
  language shortens the traces mostly in that phase, without changing whether the hint is mentioned.
- In a single-sample comparison, the hint moves English answers 17 points more often than Turkish ones when the
  language is not named, but about equally often when it is named.

So comparisons between thinking languages need to fix and report how the language is forced and how the traces are
sampled.

## What is in this repository

| path | what |
|---|---|
| `paper/with-the-question-fixed.pdf` | the paper |
| `cotlang/` | the experiment harness: sampling the items (`items.py`), the prompts, hints and prefilled sentences (`prompts.py`), generation against a vLLM server (`generate.py`, `server.py`), answer extraction (`extract.py`), the registered gates and language compliance (`gate.py`), the first-settlement split and the length measures (`phase.py`), the keyword mention regex (`mention.py`), the DeepSeek judge and its schema (`judge.py`), the analysis (`analysis.py`), and a guard that keeps GPQA text out of commits (`gpqa_guard.py`). `run.py` is the entry point |
| `configs/` | the configuration of every run in the paper: Stage A (`stageA_core`, `stageA_extras`, `stageA_instruct`), Stage B (`stageB_tr`), the presence-penalty probe (`probe_pp0`, `probe_pp15`) and the judge passes (`*_judge`) |
| `scripts/` | the analyses behind the paper's tables and figures: uptake (`uptake_cells.py`, `uptake_by_forcing.py`), disclosure (`disclosure_readout.py`, `keyword_mention.py`), the length checks (`length_defence.py`, `length_defence_extra.py`), the phase analysis (`phase_report.py`), Stage B against Stage A (`cross_stage.py`), the penalty probe (`probe_report.py`), the judge's test-retest (`judge_retest.py`) and the figures (`figures.py`) |
| `stageA/`, `stageB/`, `probes/` | the per-cell results those scripts write, and the per-trace gate records (ids, answers, lengths, compliance; no question text) |

Run with [uv](https://docs.astral.sh/uv/): `uv run python -m cotlang.run --config configs/<run>.yaml --stage <stage>`.
`scripts/figures.py`, `scripts/uptake_cells.py` and `scripts/uptake_by_forcing.py` run from the files in this repository
alone. The other scripts read the harness's own run folders (`runs/<workdir>/`), which hold the traces and are not
published here; the same traces and judge labels are in the dataset below, in a joined format. Comments in the configs
and scripts refer to planning documents of the working repository that are not included.

## Data

The traces, the judge's labels and the checked Turkish translation of the GPQA Diamond questions are a gated dataset on
the Hugging Face Hub: <https://huggingface.co/datasets/emrehannn/cotlang-gpqa-diamond-cot-traces>. Access is granted on
request, because GPQA's authors ask that its questions stay off the open web and the traces restate them. For the same
reason, no GPQA question text is in this repository.

## Citation

```bibtex
@misc{dalaman2026questionfixed,
  author       = {Dalaman, Emrehan},
  title        = {With the Question Fixed: Hint Uptake and Disclosure When a Reasoning Model Is Forced to Think in Another Language},
  year         = {2026},
  howpublished = {Term paper, Universit{\"a}t Trier. \url{https://github.com/emrehannn/cotlang}}
}
```
