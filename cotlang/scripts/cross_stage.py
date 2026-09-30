#!/usr/bin/env python3
"""The cross-stage paired table of the twenty-fifth addendum: Stage B (the Turkish prompt) against Stage A (the English
prompt), paired by question, for each thinking language T in {tr, en} and each cue. This is the decomposition of
"the prompt is Turkish" from "the thinking is Turkish".

    uv run --locked python scripts/cross_stage.py --a runs/stageA_gpqa --b runs/stageB_tr [--out stageB/cross_stage]
        [--items-a data/items_gpqa.jsonl] [--items-b data/items_gpqa_tr.jsonl] [--hint-aware]

Two blocks.
  judge-free  answer rate, hint-following (uptake), accuracy, prose compliance, length, the regex settlement position, the
              English tail and the not-target switch (the tail-switch mechanism test). Built from extracted.jsonl of each
              stage through cotlang.phase.phase_frame, so it runs while Stage B is still blind. Its commitment detector is
              the hint-BLIND one on BOTH stages so its boundaries are like for like; --hint-aware applies the registered
              union exclusion on both stages and refuses a blind workdir, as scripts/phase_report.py does.
  judge       mention, first-mention position, mention before the settlement, reliance, the language of the first mention.
              Read from report/samples.csv of each stage; runs only when neither workdir is blind and both are analysed.
              Its positions sit on samples.csv's boundary, which `--stage analyze` writes HINT-AWARE in hint rows (the
              registered boundary, twenty-second addendum) — not the judge-free block's hint-blind one; so
              `mention_before_commit_regex_blind` is added beside the registered column, on samples.csv's blind boundary,
              for a like-for-like reading across the two blocks. A blind workdir's judged.jsonl and samples.csv are never
              opened; on an unblinded one judged.jsonl is only counted, to warn when samples.csv is stale.

Pairing: Stage A cell qen_t{T}_pzhao_long_c{cue}_clen (k = 3) against Stage B cell qtr_t{T}_pzhao_long_c{cue}_cltr (k = 1),
by item_id, with the registered estimators of cotlang.phase: paired_ratio for lengths (per-question median, ratio B / A,
median of ratios), paired_diff for bounded quantities (median of per-question differences B - A) and paired_diff(rate=True)
for 0/1 rates (per-question MEAN, so k = 3 against k = 1 is not degenerate; mean of differences). Bootstrap intervals over
questions. The uptake row is a difference in differences: per question, P(follow | cue) - P(follow | no hint) in each
stage, then B - A. Only the mention and first-mention rows (B against A per T, and the within-stage T = tr minus T = en
mention margin) and predictions (i)-(iii) are registered; the rest of the judge-free block, the 2 x 2 and the DiD
estimator are unregistered extras and are labelled as such in the report.

Predictions registered in the twenty-fifth addendum, read by hand from the tables: (i) uptake does not depend on the
prompt language; (ii) under the Turkish prompt T = tr mentions less than T = en by about the Stage A margin (a property of
the thinking language) or both Stage B cells sit near Stage A's T = tr cell (a property of the prompt language); (iii)
compliance for T = tr is at least Stage A's. The within-stage T = tr minus T = en contrast is printed for both stages with
the same estimator so the two margins are like for like.

Outputs (numbers, ids and cell keys only; no text): CROSS-STAGE.md, cross_stage_paired.csv (B against A; `block` says
judge-free or judge), cross_stage_2x2.csv and cross_stage_2x2_judge.csv (Q x T per cue and metric), cross_stage_within.csv
(T = tr minus T = en within each stage, both blocks), cross_stage_uptake.csv, cross_stage_items.csv (the per-question
values the paired rows are computed from: answered traces only, mean for rates, median for the rest; `answered` itself
over every generated trace).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cotlang import phase
from cotlang.phase import N_BOOT, RNG_SEED, paired_diff, paired_ratio
from cotlang.analysis import load_jsonl
from cotlang.items import load_items

RATE = ("hint_follow", "correct", "has_commit", "has_switch_nt", "has_tail_en", "has_tail_en_5")
DIFF = ("prose_compliance", "commit_frac_regex", "commit_last_frac", "switch_nt_frac", "pre_commit_en_share", "post_commit_en_share")
RATIO = ("tokens", "chars", "n_sent", "pre_commit_chars_regex", "post_commit_chars_regex")
JUDGE_RATE = ("mentions", "chen_verbal", "relies", "mention_before_commit_regex", "mention_before_commit_regex_blind",
              "mention_before_commit", "mention_post", "first_mention_in_en")
JUDGE_DIFF = ("first_mention_frac", "first_mention_deriv_frac", "first_mention_deriv_frac_chars")
T_LANGS = ("tr", "en")
LABEL_A, LABEL_B = "Q=en (Stage A)", "Q=tr (Stage B)"


def md(frame: pd.DataFrame, **kw) -> str:
    return frame.round(3).to_markdown(**kw) if frame is not None and not frame.empty else "(none)"


def boot_mean(d: np.ndarray) -> tuple[float, float, float]:
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return (float(d.mean()) if len(d) else np.nan), np.nan, np.nan
    rng = np.random.default_rng(RNG_SEED)
    boots = np.mean(d[rng.integers(0, len(d), size=(N_BOOT, len(d)))], axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def check_items(items_a: list[dict], items_b: list[dict]) -> list[str]:
    """The two item files must carry the same questions under the same ids, gold letters, hint letters and English text
    (question and options, so the option order is the same); otherwise the pairing is between different questions."""
    a = {it["item_id"]: it for it in items_a}
    b = {it["item_id"]: it for it in items_b}
    problems = []
    if set(a) != set(b):
        problems.append(f"item id sets differ: {len(set(a) - set(b))} only in A, {len(set(b) - set(a))} only in B")
    for k in ("answer", "hint_letter"):
        bad = [i for i in a if i in b and a[i].get(k) != b[i].get(k)]
        if bad:
            problems.append(f"{len(bad)} items differ in `{k}` between the two item files")
    both = [i for i in a if i in b and isinstance(a[i].get("text"), dict) and isinstance(b[i].get("text"), dict)]
    bad = [i for i in both if "en" in a[i]["text"] and "en" in b[i]["text"] and a[i]["text"]["en"] != b[i]["text"]["en"]]
    if bad:
        problems.append(f"{len(bad)} items differ in their English text (question or option order)")
    return problems


def load_stage(work: Path, items: list[dict], hint_aware: bool, q_lang: str) -> pd.DataFrame:
    ext = [e for e in load_jsonl(work / "extracted.jsonl") if "error" not in e]
    if hint_aware:
        from cotlang import mention
        rows = [phase.phase_frame(items, [e], None if e["cell"]["cue"] == "none" else mention.patterns_union(e["cell"]["cue"]))
                for e in ext]
        pf = pd.concat(rows, ignore_index=True)
    else:
        pf = phase.phase_frame(items, ext)
    if not pf.gen_id.is_unique:
        dup = int(pf.gen_id.duplicated().sum())
        print(f"[cross] WARNING {work}: {dup} duplicate gen_ids in extracted.jsonl; keeping the last record of each", file=sys.stderr)
        pf = pf.drop_duplicates("gen_id", keep="last")
    other = pf[pf.q_lang != q_lang]
    if len(other):
        print(f"[cross] WARNING {work}: {len(other)} traces with q_lang != {q_lang} are ignored", file=sys.stderr)
    return pf[(pf.q_lang == q_lang) & (pf.prefix == "zhao_long") & (pf.instruct == "none")].copy()


def prep(pf: pd.DataFrame) -> pd.DataFrame:
    """phase._prep (answered traces, the 0/1 event columns), plus: the not-target switch is undefined for English
    thinking (phase_switch returns None for target `en`, which _prep would print as a real 0)."""
    d = phase._prep(pf)
    d.loc[d.t_lang == "en", "has_switch_nt"] = np.nan
    return d


def judge_frame(work: Path):
    """(frame, reason_not_run, warning). Refuses a blind workdir before touching anything in it."""
    if (work / ".blind").exists():
        return None, "the workdir is blind (no judge labels may be read)", None
    p = work / "report" / "samples.csv"
    if not p.exists():
        return None, "report/samples.csv missing (run `--stage analyze` after judging)", None
    s = pd.read_csv(p)
    if "judged" not in s or not bool(s.judged.any()):
        return None, "no judged rows in report/samples.csv", None
    labelled = set()
    jp = work / "judged.jsonl"
    if jp.exists():
        for line in open(jp, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r and r.get("gen_id"):
                labelled.add(r["gen_id"])
    unseen = len(labelled) - int(s.judged.sum())
    warn = (f"judged.jsonl holds {unseen} labels that report/samples.csv has not absorbed; re-run `--stage analyze` before "
            f"reading the judge rows") if unseen > 0 else None
    f = s[s.judged & s.answered & (s.cue != "none") & s.hint_follow & (s.prefix == "zhao_long") & (s.instruct == "none")].copy()
    f["first_mention_in_en"] = np.where(f.first_mention_lang.isna(), np.nan, (f.first_mention_lang == "en").astype(float))
    if "commit_regex_sentence_blind" in f and "first_mention" in f:
        ok = f.first_mention.notna() & f.commit_regex_sentence_blind.notna()
        f["mention_before_commit_regex_blind"] = np.where(ok, (f.first_mention <= f.commit_regex_sentence_blind).astype(float), np.nan)
    return f, None, warn


def paired_rows(a: pd.DataFrame, b: pd.DataFrame, rates, diffs, ratios, base: dict) -> list[dict]:
    """B against A on every column present in both frames; rows with no paired question are dropped."""
    rows = []
    for col in rates:
        if col in a and col in b:
            r = paired_diff(b.dropna(subset=[col]), a.dropna(subset=[col]), col, rate=True)
            rows.append({**base, "metric": col, "kind": "rate", "n_items": r["n_items"], "a": r.get("mean_den", np.nan),
                         "b": r.get("mean_num", np.nan), "estimate": r["median_diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    for col in diffs:
        if col in a and col in b:
            r = paired_diff(b.dropna(subset=[col]), a.dropna(subset=[col]), col)
            rows.append({**base, "metric": col, "kind": "diff", "n_items": r["n_items"], "a": r["median_den"], "b": r["median_num"],
                         "estimate": r["median_diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    for col in ratios:
        if col in a and col in b:
            r = paired_ratio(b.dropna(subset=[col]), a.dropna(subset=[col]), col)
            rows.append({**base, "metric": col, "kind": "ratio", "n_items": r["n_items"], "a": r["median_den"], "b": r["median_num"],
                         "estimate": r["median_ratio"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    return [r for r in rows if r["n_items"] > 0]


def uptake_per_item(d: pd.DataFrame) -> pd.DataFrame:
    """Per question and T: P(follow | cue) - P(follow | no hint), over answered traces; NaN where either cell is missing."""
    hf = d[d.answered].groupby(["t_lang", "cue", "item_id"])["hint_follow"].mean().unstack("cue")
    if "none" not in hf:
        return pd.DataFrame()
    out = []
    for cue in [c for c in hf.columns if c != "none"]:
        u = (hf[cue] - hf["none"]).rename("uptake").reset_index()
        u["cue"] = cue
        out.append(u)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def two_by_two(a: pd.DataFrame, b: pd.DataFrame, rate_cols, med_cols) -> pd.DataFrame:
    rows = []
    for label, d in ((LABEL_A, a), (LABEL_B, b)):
        for (tl, cue), sub in d.groupby(["t_lang", "cue"]):
            row = {"stage": label, "t_lang": tl, "cue": cue, "n": len(sub)}
            for c in rate_cols:
                if c in sub:
                    row[c] = float(sub[c].mean()) if sub[c].notna().any() else np.nan
            for c in med_cols:
                if c in sub:
                    row[c] = float(sub[c].median()) if sub[c].notna().any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def per_item_values(d: pd.DataFrame, d_all: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Exactly the per-question values the paired rows aggregate: answered traces, mean for the rates, median for the
    bounded quantities and lengths; `answered` over every generated trace."""
    keys = ["item_id", "t_lang", "cue"]
    rates = [c for c in RATE if c in d]
    meds = [c for c in DIFF + RATIO if c in d]
    g = d.groupby(keys)
    out = pd.concat([g[rates].mean(), g[meds].median()], axis=1)
    out["answered"] = d_all.groupby(keys)["answered"].mean()
    return out.assign(stage=stage).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="runs/stageA_gpqa", help="Stage A workdir (English prompt)")
    ap.add_argument("--b", default="runs/stageB_tr", help="Stage B workdir (Turkish prompt)")
    ap.add_argument("--items-a", default="data/items_gpqa.jsonl")
    ap.add_argument("--items-b", default="data/items_gpqa_tr.jsonl")
    ap.add_argument("--out", default="stageB/cross_stage")
    ap.add_argument("--hint-aware", action="store_true", help="the registered union exclusion on both stages (neither workdir may be blind)")
    a_ = ap.parse_args()
    wa, wb, out = Path(a_.a), Path(a_.b), Path(a_.out)
    blind = {w: (w / ".blind").exists() for w in (wa, wb)}
    if a_.hint_aware and any(blind.values()):
        sys.exit("[blind] --hint-aware scores hint mention and may not run while either workdir is blind: "
                 + ", ".join(str(w) for w, v in blind.items() if v))
    for w in (wa, wb):
        if not (w / "extracted.jsonl").exists():
            sys.exit(f"[cross] {w/'extracted.jsonl'} missing: run `--stage extract` first")
    items_a, items_b = load_items(Path(a_.items_a)), load_items(Path(a_.items_b))
    problems = check_items(items_a, items_b)
    if problems:
        sys.exit("[cross] the item files do not pair: " + "; ".join(problems))
    n_items = len(items_a)
    out.mkdir(parents=True, exist_ok=True)
    A_all = load_stage(wa, items_a, False, "en")
    B_all = load_stage(wb, items_b, False, "tr")
    A = prep(load_stage(wa, items_a, True, "en") if a_.hint_aware else A_all)
    B = prep(load_stage(wb, items_b, True, "tr") if a_.hint_aware else B_all)
    cues = sorted(set(A.cue) & set(B.cue))
    missing = sorted((set(A.cue) | set(B.cue)) - set(cues))
    short = []
    for label, d in ((LABEL_A, A_all), (LABEL_B, B_all)):
        per = d.groupby(["t_lang", "cue"])["item_id"].nunique()
        short += [f"{label} T={tl} {cue}: {int(n)}/{n_items} questions" for (tl, cue), n in per.items() if n < n_items]
    partial = bool(short)
    lines = [f"# Cross-stage paired table{' — PARTIAL' if partial else ''} — Stage B `{wb}` (Turkish prompt) against Stage A `{wa}` (English prompt)", "",
             f"Stage B workdir: {'BLIND' if blind[wb] else 'unblinded'}; Stage A workdir: {'BLIND' if blind[wa] else 'unblinded'}; "
             f"judge-free commitment detector: {'hint-aware (union exclusion, both stages)' if a_.hint_aware else 'hint-blind (both stages)'}.", "",
             f"Traces in the zhao_long, no-instruction cells: Stage A {len(A_all)} ({int(A_all.answered.sum())} answered), "
             f"Stage B {len(B_all)} ({int(B_all.answered.sum())} answered); cues in both stages: {', '.join(cues)}"
             + (f"; only in one stage and left out of the pairing: {', '.join(missing)}" if missing else "") + ".", ""]
    if partial:
        lines += ["**PARTIAL: a cell is short of the item file, so its rows are over a subject-ordered subset of questions, not a "
                  "random one — read nothing from them but plumbing.** " + "; ".join(short), ""]
    lines += ["Pairing by question id (the two item files agree on ids, gold letters, hint letters and English text). Estimates are "
              "B minus A (rates by paired mean, bounded quantities by paired median) or B / A (lengths, median of per-question "
              "ratios); 95% bootstrap intervals over questions. Stage A has three samples per question and Stage B one; the "
              "per-question value is the mean (rates) or median (others) of a question's samples. Registered (twenty-fifth "
              "addendum): the mention and first-mention rows of the judge block and predictions (i)-(iii); the judge-free block, "
              "the 2 x 2 and the DiD estimator are unregistered extras.", ""]

    rate_cols = ["hint_follow", "correct", "prose_compliance", "has_tail_en", "has_switch_nt", "has_commit"]
    med_cols = ["tokens", "n_sent", "commit_frac_regex", "commit_last_frac", "switch_nt_frac"]
    ans = pd.concat([A_all.groupby(["t_lang", "cue"])["answered"].mean().reset_index().assign(stage=LABEL_A),
                     B_all.groupby(["t_lang", "cue"])["answered"].mean().reset_index().assign(stage=LABEL_B)], ignore_index=True)
    tt = two_by_two(A, B, rate_cols, med_cols).merge(ans, on=["stage", "t_lang", "cue"], how="left")
    up_a, up_b = uptake_per_item(A), uptake_per_item(B)
    ups = [up.groupby(["t_lang", "cue"])["uptake"].mean().rename("uptake").reset_index().assign(stage=label)
           for label, up in ((LABEL_A, up_a), (LABEL_B, up_b)) if not up.empty]
    if ups:
        tt = tt.merge(pd.concat(ups, ignore_index=True), on=["stage", "t_lang", "cue"], how="left")
    tt = tt.sort_values(["cue", "t_lang", "stage"]).reset_index(drop=True)
    lines += ["## The 2 x 2 per cue: prompt language (stage) x thinking language (judge-free)", "",
              "Over answered traces except `answered` (every generated trace). `n` counts answered traces. `uptake` = P(follow | cue) - "
              "P(follow | no hint) as the mean of per-question differences. `prose_compliance` and the `has_*` columns are means; "
              "`tokens` and `n_sent` medians over answered traces; `commit_frac_regex`, `commit_last_frac` and `switch_nt_frac` are "
              "medians over the traces that HAVE that event. `has_tail_en`, `has_switch_nt` and `switch_nt_frac` are undefined for "
              "English thinking (NaN).", "", md(tt, index=False), ""]

    rows = []
    for tl in T_LANGS:
        for cue in cues:
            a, b = A[(A.t_lang == tl) & (A.cue == cue)], B[(B.t_lang == tl) & (B.cue == cue)]
            aa, bb = A_all[(A_all.t_lang == tl) & (A_all.cue == cue)], B_all[(B_all.t_lang == tl) & (B_all.cue == cue)]
            if a.empty or b.empty:
                continue
            base = {"block": "judge-free", "t_lang": tl, "cue": cue}
            r = paired_diff(bb, aa, "answered", rate=True)
            rows.append({**base, "metric": "answered", "kind": "rate", "n_items": r["n_items"], "a": r.get("mean_den", np.nan),
                         "b": r.get("mean_num", np.nan), "estimate": r["median_diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
            rows += paired_rows(a, b, RATE, DIFF, RATIO, base)
    paired = pd.DataFrame(rows)
    lines += ["## Stage B against Stage A, paired by question, per thinking language and cue (judge-free)", "",
              "Rates: `estimate` = mean of per-question differences B - A. Diffs: median of per-question differences. Ratios: B / A. "
              "`a` and `b` are the two stages' values over the paired questions. Denominators as in the 2 x 2 (position medians over "
              "traces with the event; `answered` over every generated trace, everything else over answered traces).", "",
              md(paired, index=False), ""]

    urows = []
    if not up_a.empty and not up_b.empty:
        for (tl, cue), sb in up_b.groupby(["t_lang", "cue"]):
            sa = up_a[(up_a.t_lang == tl) & (up_a.cue == cue)]
            m = sb.merge(sa, on=["t_lang", "cue", "item_id"], suffixes=("_b", "_a")).dropna(subset=["uptake_a", "uptake_b"])
            if m.empty:
                continue
            est, lo, hi = boot_mean((m.uptake_b - m.uptake_a).values.astype(float))
            urows.append({"t_lang": tl, "cue": cue, "n_items": len(m), "uptake_a": float(m.uptake_a.mean()),
                          "uptake_b": float(m.uptake_b.mean()), "b_minus_a": est, "ci_lo": lo, "ci_hi": hi})
    uptake = pd.DataFrame(urows)
    lines += ["## Uptake, difference in differences (prediction i: no dependence on the prompt language)", "",
              "Per question and stage, P(follow | cue) - P(follow | no hint) over answered traces; then B - A, mean over the "
              "questions with all four cells, bootstrap interval. (An unregistered estimator; the registered reading of (i) is the "
              "`hint_follow` rows above, cue cells against control cells.)", "", md(uptake, index=False), ""]

    wrows = []
    for label, d in ((LABEL_A, A), (LABEL_B, B)):
        for cue in sorted(set(d.cue)):
            tr, en = d[(d.t_lang == "tr") & (d.cue == cue)], d[(d.t_lang == "en") & (d.cue == cue)]
            if tr.empty or en.empty:
                continue
            wrows += paired_rows(en, tr, ("hint_follow", "correct", "has_commit"), ("prose_compliance", "commit_frac_regex"),
                                 ("tokens", "n_sent"), {"block": "judge-free", "stage": label, "cue": cue})
    within = pd.DataFrame(wrows).rename(columns={"a": "t_en", "b": "t_tr"}) if wrows else pd.DataFrame()
    lines += ["## Within each stage: T = tr against T = en, paired by question (judge-free)", "",
              "The same estimator in both stages, so the Turkish-minus-English margin under the Turkish prompt can be read against "
              "the one under the English prompt. `estimate` is tr - en (rates and diffs) or tr / en (ratios).", "",
              md(within, index=False), ""]

    JA, why_a, warn_a = judge_frame(wa)
    JB, why_b, warn_b = judge_frame(wb)
    jrows, jwithin, j2 = [], [], pd.DataFrame()
    if JA is None or JB is None:
        lines += ["## Judge block: not run", "",
                  f"Stage A: {why_a or 'labels available'}; Stage B: {why_b or 'labels available'}. The mention, position and "
                  "reliance rows appear once both workdirs are unblinded, judged and analysed.", ""]
    else:
        for tl in T_LANGS:
            for cue in sorted(set(JA.cue) & set(JB.cue)):
                a, b = JA[(JA.t_lang == tl) & (JA.cue == cue)], JB[(JB.t_lang == tl) & (JB.cue == cue)]
                if a.empty or b.empty:
                    continue
                jrows += paired_rows(a, b, JUDGE_RATE, JUDGE_DIFF, (), {"block": "judge", "t_lang": tl, "cue": cue})
        for label, d in ((LABEL_A, JA), (LABEL_B, JB)):
            for cue in sorted(set(d.cue)):
                tr, en = d[(d.t_lang == "tr") & (d.cue == cue)], d[(d.t_lang == "en") & (d.cue == cue)]
                if tr.empty or en.empty:
                    continue
                jwithin += paired_rows(en, tr, ("mentions", "chen_verbal", "relies", "mention_before_commit_regex", "mention_before_commit_regex_blind"),
                                       ("first_mention_frac", "first_mention_deriv_frac"), (), {"block": "judge", "stage": label, "cue": cue})
        j2 = two_by_two(JA, JB, ["mentions", "chen_verbal", "relies", "mention_before_commit_regex", "mention_before_commit_regex_blind",
                                 "mention_post", "first_mention_in_en"], ["first_mention_frac", "first_mention_deriv_frac"])
        jw = pd.DataFrame(jwithin).rename(columns={"a": "t_en", "b": "t_tr"}) if jwithin else pd.DataFrame()
        warns = [w for w in (warn_a, warn_b) if w]
        lines += ["## Judge block (judged, answered, hint-following traces; both stages unblinded)", ""]
        if warns:
            lines += ["WARNING: " + " / ".join(warns), ""]
        lines += ["Positions and `mention_before_commit_regex` are relative to report/samples.csv's boundary, which is the registered "
                  "HINT-AWARE settlement in hint rows — not the judge-free block's hint-blind one; `mention_before_commit_regex_blind` "
                  "is the same share on the hint-blind boundary (a first mention in the settlement sentence counts as before it). "
                  "The 2 x 2 (rates are means, positions medians; `first_mention_in_en` is the GlotLID label of the first-mention "
                  "sentence, so it is ≈ 1 for English thinking):", "", md(j2, index=False), "",
                  "Stage B against Stage A, paired by question:", "", md(pd.DataFrame(jrows), index=False), "",
                  "Within each stage, T = tr against T = en (prediction ii reads Stage B's `mentions` margin against Stage A's):", "",
                  md(jw, index=False), ""]
        if not jw.empty:
            within = pd.concat([within, jw], ignore_index=True)
        if jrows:
            paired = pd.concat([paired, pd.DataFrame(jrows)], ignore_index=True)

    per_item = pd.concat([per_item_values(A, A_all, "A"), per_item_values(B, B_all, "B")], ignore_index=True)
    text = "\n".join(lines)
    (out / "CROSS-STAGE.md").write_text(text, encoding="utf-8")
    paired.to_csv(out / "cross_stage_paired.csv", index=False)
    tt.to_csv(out / "cross_stage_2x2.csv", index=False)
    if not j2.empty:
        j2.to_csv(out / "cross_stage_2x2_judge.csv", index=False)
    within.to_csv(out / "cross_stage_within.csv", index=False)
    uptake.to_csv(out / "cross_stage_uptake.csv", index=False)
    per_item.to_csv(out / "cross_stage_items.csv", index=False)
    print(text)
    print(f"[cross] wrote {out/'CROSS-STAGE.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
