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
- In a single-sample comparison, the hint moves English answers 19 points more often than Turkish ones when the
  language is not named, but about equally often when it is named.

So comparisons between thinking languages need to fix and report how the language is forced and how the traces are
sampled.

## What is in this repository

| path | what |
|---|---|
| `paper/with-the-question-fixed.pdf` | the paper |
| `configs/` | the configuration of every run, from the smoke tests and pilots to Stage A, Stage B, the presence-penalty probe and the judge passes |
| `cotlang/prompts.py` | every string the model sees: the task instruction, the hints, the prefilled sentences and the instruction arm, in the study's languages |

The experiment harness and the analysis scripts are not published yet. Comments in the configs refer to files of the
working repository that are not included here.

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
