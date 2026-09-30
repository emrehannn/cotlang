from __future__ import annotations

import csv
import json
import re
import random
import sys
from pathlib import Path

import numpy as np

from .analysis import load_jsonl
from .prompts import hint_text

TRUE_SET = ("1", "true", "yes", "y")
FALSE_SET = ("0", "false", "no", "n")


_MASK_BOXED = re.compile(r"\\boxed\{[A-D]\}")


def export(extracted_path: Path, judged_path: Path | None, items: list[dict], out_csv: Path,
           t_langs: list[str], n_per_lang: int, seed: int) -> None:
    """Write n_per_lang cued traces per trace language. If judge labels exist, take half judged-positive and half
    judged-negative (the labels themselves are not written). The model's answer is not shown either."""
    ex = [e for e in load_jsonl(extracted_path)
          if "error" not in e and e["cell"]["cue"] != "none" and e["cell"]["t_lang"] in t_langs and e.get("answer")]
    jd = {}
    if judged_path and judged_path.exists():
        for j in load_jsonl(judged_path):
            if "gen_id" in j and "error" not in j:
                jd[j["gen_id"]] = j
    rng = random.Random(seed)
    it = {i["item_id"]: i for i in items}
    rows = []
    for lang in t_langs:
        pool = [e for e in ex if e["cell"]["t_lang"] == lang]
        rng.shuffle(pool)
        if jd:
            pos = [e for e in pool if e["gen_id"] in jd and jd[e["gen_id"]].get("mentions_hint")]
            neg = [e for e in pool if e["gen_id"] in jd and not jd[e["gen_id"]].get("mentions_hint")]
            half = n_per_lang // 2
            chosen = pos[:half] + neg[: n_per_lang - min(half, len(pos))]
            if len(chosen) < n_per_lang:
                rest = [e for e in pool if e not in chosen]
                chosen += rest[: n_per_lang - len(chosen)]
        else:
            chosen = pool[:n_per_lang]
        for e in chosen:
            c = e["cell"]
            item = it[e["item_id"]]
            rows.append({
                "gen_id": e["gen_id"], "cell_key": e["cell_key"], "t_lang": c["t_lang"], "cue": c["cue"],
                "hint_text": hint_text(c["cue"], c["cue_lang"], item["hint_letter"], item["item_id"]),
                "question_en": item["text"]["en"]["question"],
                "think_numbered": _MASK_BOXED.sub(r"\\boxed{?}",
                                                  "\n".join(f"[{i+1}] {s}" for i, s in enumerate(e["sentences"]))),
                "post": _MASK_BOXED.sub(r"\\boxed{?}", e["post"] or ""),
                "human_mentions_hint": "", "human_first_mention_sentence": "", "human_relies_on_hint": "",
                "human_restated_as_own": "", "human_notes": "",
            })
    if not rows:
        print("[annotate] nothing to export: no cued, answered traces for the requested languages", file=sys.stderr)
        return
    rng.shuffle(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[annotate] wrote {len(rows)} rows to {out_csv} ({'stratified on judge label' if jd else 'unstratified'})", file=sys.stderr)


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary label lists."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _parse_bool(s: str) -> bool | None:
    """Read a human yes/no cell; None if empty or unrecognized."""
    s = (s or "").strip().lower()
    if s in TRUE_SET:
        return True
    if s in FALSE_SET:
        return False
    return None


def _boot_kappa(a: list[bool], b: list[bool], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval for kappa (resampling traces)."""
    rng = np.random.default_rng(seed)
    a, b = np.array(a), np.array(b)
    ks = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        ks.append(cohen_kappa(list(a[idx]), list(b[idx])))
    ks = np.array([k for k in ks if not np.isnan(k)])
    return (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))) if len(ks) else (float("nan"), float("nan"))


def kappa(filled_csv: Path, judged_path: Path) -> dict:
    """Per trace language: kappa for mentions (with CI) and reliance, the judge's sensitivity/specificity
    against the human, and agreement on the first-mention sentence within ±1."""
    jd = {j["gen_id"]: j for j in load_jsonl(judged_path) if "gen_id" in j and "mentions_hint" in j}
    per_lang: dict[str, dict] = {}
    with filled_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hv = _parse_bool(r.get("human_mentions_hint", ""))
            j = jd.get(r["gen_id"])
            if hv is None or not j:
                continue
            d = per_lang.setdefault(r["t_lang"], {"hm": [], "jm": [], "hr": [], "jr": [], "hs": [], "js": [], "first_ok": []})
            d["hm"].append(hv); d["jm"].append(bool(j["mentions_hint"]))
            rv = _parse_bool(r.get("human_relies_on_hint", ""))
            if rv is not None:
                d["hr"].append(rv); d["jr"].append(bool(j.get("relies_on_hint")))
            sv = _parse_bool(r.get("human_restated_as_own", ""))
            if sv is not None and "restated_as_own" in j:
                d["hs"].append(sv); d["js"].append(bool(j["restated_as_own"]))
            if hv and j["mentions_hint"]:
                try:
                    hf = int(r.get("human_first_mention_sentence", "").strip())
                    jf = j.get("first_mention_sentence")
                    if jf is not None:
                        d["first_ok"].append(abs(hf - int(jf)) <= 1)
                except ValueError:
                    pass
    out = {}
    for lang, d in per_lang.items():
        hm, jm = d["hm"], d["jm"]
        n = len(hm)
        tp = sum(h and j for h, j in zip(hm, jm)); fn = sum(h and not j for h, j in zip(hm, jm))
        fp = sum((not h) and j for h, j in zip(hm, jm)); tn = sum((not h) and (not j) for h, j in zip(hm, jm))
        lo, hi = _boot_kappa(hm, jm)
        res = {"n": n, "kappa_mentions": cohen_kappa(hm, jm), "kappa_ci": (lo, hi),
               "human_rate": sum(hm) / n, "judge_rate": sum(jm) / n,
               "judge_sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
               "judge_specificity": tn / (tn + fp) if tn + fp else float("nan"),
               "first_mention_within1": (sum(d["first_ok"]) / len(d["first_ok"])) if d["first_ok"] else float("nan"),
               "n_first_mention_compared": len(d["first_ok"])}
        if d["hr"]:
            res["kappa_relies"] = cohen_kappa(d["hr"], d["jr"]); res["n_relies"] = len(d["hr"])
        if d["hs"]:
            res["kappa_restated"] = cohen_kappa(d["hs"], d["js"]); res["n_restated"] = len(d["hs"])
        out[lang] = res
        print(f"METRIC kappa/{lang}/mentions={res['kappa_mentions']:.3f} ci=[{lo:.2f},{hi:.2f}] (n={n}, human={res['human_rate']:.2f}, "
              f"judge={res['judge_rate']:.2f}, sens={res['judge_sensitivity']:.2f}, spec={res['judge_specificity']:.2f}, "
              f"first±1={res['first_mention_within1']:.2f} on {res['n_first_mention_compared']})")
        if "kappa_relies" in res:
            print(f"METRIC kappa/{lang}/relies={res['kappa_relies']:.3f} (n={res['n_relies']})")
        if "kappa_restated" in res:
            print(f"METRIC kappa/{lang}/restated_as_own={res['kappa_restated']:.3f} (n={res['n_restated']})")
    if not out:
        print("[annotate] no usable rows: fill human_mentions_hint with 1/0 and make sure the judge file has these gen_ids", file=sys.stderr)
    return out
