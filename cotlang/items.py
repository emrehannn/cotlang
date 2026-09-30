from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from .prompts import LETTERS, QWEN3_USER, longest_user_message, pick_hint_letter, stable_seed

FALLBACK_CHARS_PER_TOKEN = 2.5


def _load_lang(dataset: str, lang: str) -> dict[str, dict]:
    """Download one language of Global-MMLU and index its rows by sample_id."""
    ds = load_dataset(dataset, lang, split="test")
    out = {}
    for r in ds:
        out[r["sample_id"]] = {
            "question": r["question"],
            "options": {"A": r["option_a"], "B": r["option_b"], "C": r["option_c"], "D": r["option_d"]},
            "answer": r["answer"],
            "subject": r["subject"],
            "subject_category": r["subject_category"],
            "cultural_sensitivity_label": r.get("cultural_sensitivity_label", "-"),
            "is_annotated": bool(r.get("is_annotated", False)),
        }
    return out


def load_exclusions(path: Path | None) -> set[str]:
    """Item ids listed one per line in the exclusion file ("#" starts a comment)."""
    if not path or not Path(path).exists():
        return set()
    out = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def load_tokenizer(name: str | None):
    """The model's tokenizer (for the length cap), or None if it cannot be loaded."""
    if not name:
        return None
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_pretrained(name)
    except Exception as e:
        print(f"[items] WARNING: could not load tokenizer {name} ({e!r}); estimating {FALLBACK_CHARS_PER_TOKEN} chars/token", file=sys.stderr)
        return None


def prompt_tokens(tok, item: dict, lang: str) -> int:
    """Token length of the longest prompt this item can produce in `lang`."""
    raw = QWEN3_USER.format(content=longest_user_message(item, lang), prefix="")
    if tok is None:
        return int(len(raw) / FALLBACK_CHARS_PER_TOKEN)
    return len(tok.encode(raw).ids)


def _make_item(sid: str, per_lang: dict, langs: list[str], seed: int) -> dict:
    """Assemble one item record from its rows in every language."""
    base = per_lang["en"][sid]
    return {
        "item_id": sid,
        "subject": base["subject"],
        "subject_category": base["subject_category"],
        "answer": base["answer"],
        "hint_letter": pick_hint_letter(sid, base["answer"], seed),
        "cultural_sensitivity_label": base["cultural_sensitivity_label"],
        "is_annotated": {l: per_lang[l][sid]["is_annotated"] for l in langs},
        "text": {l: {"question": per_lang[l][sid]["question"], "options": per_lang[l][sid]["options"]} for l in langs},
    }


def _load_gpqa(dataset: str, config: str, seed: int) -> dict[str, dict]:
    """GPQA (English only), shaped like a _load_lang() result. GPQA stores the correct answer and three wrong ones in
    separate columns; they are shuffled into A-D with a fixed seed per question, so the correct letter is deterministic.
    GPQA is gated: accept its terms on Hugging Face and log in with a valid token first. Its authors ask that questions
    not be published online, so GPQA item files and traces are kept out of git (data/*.jsonl and runs/ are ignored)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(dataset, f"{config}.csv", repo_type="dataset")
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid = r["Record ID"].strip()
            texts = [r["Correct Answer"].strip()] + [r[f"Incorrect Answer {k}"].strip() for k in (1, 2, 3)]
            order = list(range(4))
            random.Random(stable_seed(seed, rid, "gpqa-options")).shuffle(order)
            out[f"gpqa/{config}/{rid}"] = {
                "question": r["Question"].strip(),
                "options": {LETTERS[i]: texts[order[i]] for i in range(4)},
                "answer": LETTERS[order.index(0)],
                "subject": r.get("Subdomain", "").strip() or "gpqa",
                "subject_category": r.get("High-level domain", "").strip() or "gpqa",
                "cultural_sensitivity_label": "n/a",
                "is_annotated": True,
            }
    return out


def proportional_order(items: list[dict], key: str) -> list[dict]:
    """Reorder so that every prefix holds each group's share of the whole list to within one item (largest-deficit
    interleave). The order inside a group is kept (it is already a seeded shuffle); ties go to the alphabetically first
    group. Deterministic; the set of items is unchanged."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        groups[it[key]].append(it)
    names, n = sorted(groups), len(items)
    taken = {g: 0 for g in names}
    out = []
    for i in range(1, n + 1):
        g = max((g for g in names if taken[g] < len(groups[g])),
                key=lambda g: (i * len(groups[g]) / n - taken[g], -names.index(g)))
        out.append(groups[g][taken[g]])
        taken[g] += 1
    return out


def build_items(cfg: dict, out_path: Path, seed: int, exclude_path: Path | None = None) -> list[dict]:
    """Select the items, write them as jsonl, return them."""
    icfg = cfg["items"]
    langs = icfg["langs"]
    if "en" not in langs:
        raise ValueError("items.langs must include 'en' (the labels live on the English rows)")
    if icfg.get("source", "global_mmlu") == "gpqa":
        if langs != ["en"]:
            raise ValueError("GPQA exists only in English: set items.langs to [en]")
        if icfg.get("require_label") not in (None, "none"):
            raise ValueError("GPQA has no cultural-sensitivity label: set items.require_label to null")
        per_lang = {"en": _load_gpqa(icfg["dataset"], icfg.get("config", "gpqa_diamond"), seed)}
    else:
        per_lang = {l: _load_lang(icfg["dataset"], l) for l in langs}
    common = set.intersection(*(set(d) for d in per_lang.values()))
    excluded = load_exclusions(exclude_path)
    require_label = icfg.get("require_label", "CA")
    max_tokens = int(icfg.get("max_prompt_tokens", 400))
    tok = load_tokenizer(icfg.get("tokenizer", "Qwen/Qwen3-4B"))

    en = per_lang["en"]
    candidates = []
    for sid in sorted(common):
        base = en[sid]
        label = base["cultural_sensitivity_label"]
        if require_label == "CA" and label != "CA":
            continue
        if require_label == "not-CS" and label == "CS":
            continue
        if base["answer"] not in LETTERS:
            continue
        if any(per_lang[l][sid]["answer"] != base["answer"] for l in langs):
            continue
        candidates.append(sid)
    print(f"[items] {len(candidates)} candidates after label/answer filters (require_label={require_label})", file=sys.stderr)

    key = icfg.get("stratify_by", "subject_category")
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in candidates:
        groups[en[sid][key] if key else "all"].append(sid)
    rng = random.Random(seed)
    n = icfg["n"]
    per_group = {g: n // len(groups) for g in groups}
    for g in sorted(groups)[: n - sum(per_group.values())]:
        per_group[g] += 1

    chosen, dropped_long, dropped_excl = [], 0, 0
    for g in sorted(groups):
        pool = sorted(groups[g])
        rng.shuffle(pool)
        taken = 0
        for sid in pool:
            if taken >= per_group[g]:
                break
            if sid in excluded:
                dropped_excl += 1
                continue
            item = _make_item(sid, per_lang, langs, seed)
            lengths = {l: prompt_tokens(tok, item, l) for l in langs}
            if max(lengths.values()) > max_tokens:
                dropped_long += 1
                continue
            item["prompt_tokens_max"] = lengths
            chosen.append(item)
            taken += 1
        if taken < per_group[g]:
            print(f"[items] WARNING: category {g} has only {taken}/{per_group[g]} usable items", file=sys.stderr)
    rng.shuffle(chosen)
    if icfg.get("order_by"):
        chosen = proportional_order(chosen, icfg["order_by"])
    print(f"[items] chose {len(chosen)} items; skipped {dropped_long} over {max_tokens} tokens, {dropped_excl} excluded", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in chosen:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return chosen


def load_items(path: Path) -> list[dict]:
    """Read the jsonl item file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def export_review_csv(items: list[dict], lang: str, out_csv: Path) -> None:
    """Side-by-side English / `lang` sheet for a native speaker. Put rejected item_ids into the exclusion file."""
    rows = []
    for it in items:
        en, other = it["text"]["en"], it["text"][lang]
        rows.append({
            "item_id": it["item_id"], "subject": it["subject"], "answer": it["answer"],
            f"is_annotated_{lang}": it["is_annotated"].get(lang),
            "question_en": en["question"], f"question_{lang}": other["question"],
            **{f"{L}_en": en["options"][L] for L in LETTERS},
            **{f"{L}_{lang}": other["options"][L] for L in LETTERS},
            "review_ok": "", "notes": "",
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[items] wrote {len(rows)} rows to {out_csv}", file=sys.stderr)
