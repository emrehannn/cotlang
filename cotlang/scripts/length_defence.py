#!/usr/bin/env python3
"""The length defence (docs/ROADMAP.md step 7; PLAN Stage C, the 2026-09-06 / 2026-09-13 amendments, twenty-fourth
addendum): can trace length alone account for a disclosure difference between trace languages? Three controls, all on
stored traces, nothing regenerated, nothing re-judged.

    uv run --locked python scripts/length_defence.py runs/stageA_gpqa [--out stageA/length] [--prefix zhao_long --instruct none]

Population: answered hint-FOLLOWING traces of the hint cells under one condition (prefix + instruction arm), the
population every disclosure rate in the paper is over. Follower status is the FULL trace's and is never recomputed.
Two mention instruments, never merged: the judge's `mentions_hint` with its `first_mention_sentence` (from
report/samples.csv, so `--stage analyze` must have run; a null index on a truncated judge input is right-censored and
the trace is dropped from the judge instrument, counted in the header), and the registered keyword instrument
(cotlang/mention.py, every sentence scored, so the SET of mentioning sentences is known). Both indices are 1-based.
The keyword lists have a false-positive floor that differs by cue and language; the header prints it, measured on the
no-hint traces of the same condition, and keyword rows are read against it.

Length units. `sentences`; `tokens` = completion tokens apportioned to think sentences by characters, the final-answer
section's share removed (it is 4-9% of completion characters and larger in Turkish); and, when the model's tokenizer
can be loaded, `tokens_fertility` = tokens divided by the language's fertility on the parallel pilot items
(data/items_pilot.jsonl, the same content in en/tr/zh), which is the registered "token caps scaled by measured
fertility" with the fertility measured on parallel text outside GPQA (the GPQA items exist in English only). A sentence
is not a language-neutral unit either (Turkish sentences carry about a third more tokens than English ones); the
header prints tokens per sentence by language so the units can be read against each other.

1. BUDGET MATCHING. Every trace truncated at a cap; the first mention survives iff its sentence lies within the cap;
   position = index over truncated length, survivors only. Caps are the 25th, 50th and 75th percentiles of the SHORTER
   language's follower traces in that unit, the shorter language chosen per (cue, pair, unit) by median and named in
   `cap_from`. Paired per question with cotlang.phase.paired_diff (rates by paired mean, positions by paired median,
   bootstrap over questions). `share_truncated_*` says how much of each arm the cap actually cuts.
2. LENGTH-MATCHED BASELINES BY RANDOM SENTENCE DELETION (Little 2026's content-blind control), KEYWORD INSTRUMENT ONLY.
   For every (English trace, other-language trace) pair of a question, the English trace shortened to the other's
   length by uniform deletion and the chance that at least one of its m keyword sentences survives, exact: keeping k
   of n sentences, P(none survives) = C(n-m, k)/C(n, k). In the token units k = round(n * other/english tokens). The
   baseline is the English rate at the other language's length; the quantity of interest is the other language's
   ACTUAL rate minus that baseline, paired per question: zero means content-blind shortening predicts the difference,
   negative means the other language verbalizes less than that (Little's selective omission). The judge instrument
   cannot support this control without re-judging shortened traces: it knows only the first mention, and with m = 1
   the survival is exactly k/n, a length ratio with no information about mention, so no judge row is printed here.
   Positions are read under budget matching only: under uniform deletion E[(new index - 1)/(k - 1)] = (index - 1)/(n - 1)
   for a surviving sentence, so the relative position is unchanged in expectation (for m > 1 the first SURVIVING
   mention is a later sentence, so the expected position moves later, not earlier).
3. LENGTH AS A REGRESSOR (§5.3's gradient). Followers of all three languages pooled per cue. (a) Mention: mixed
   logistic mention ~ [log sentences] + Turkish + Chinese + (1 | question), variational Bayes as analysis.glmm_fit and
   the MAP fit beside it, with the number of non-mentions per language printed, because a rate at the ceiling (a
   language with zero non-mentions) leaves the model unidentified and the VB posterior is then a prior artefact; read
   the coefficients only where every language has non-mentions. (b) Position: linear mixed model
   first-mention index / sentences ~ log sentences + Turkish + Chinese + (1 | question), which is the gradient the data
   can carry when mention is at ceiling. Plus mention rate and median position by pooled length quintile and language.
The solved-in-every-language restriction is the fourth control and already lives in analysis.py (`mention_solved`).
Adversarial pass 2026-09-15 15:30 applied (thirteen findings; the commit records them).
"""
from __future__ import annotations

import argparse, json, sys, warnings
from math import lgamma, exp
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cotlang.mention import KEYWORDS, patterns
from cotlang.phase import paired_diff

QUANTS = (0.25, 0.5, 0.75)
TOKENIZERS = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-0.8B-Base")


def log_choose(n: int, k: int) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def survival(n: int, k: int, m: int) -> float:
    """P(at least one of m marked sentences survives when k of n are kept uniformly at random)."""
    if m <= 0:
        return 0.0
    if k >= n or k > n - m:
        return 1.0
    return 1.0 - exp(log_choose(n - m, k) - log_choose(n, k))


def cutoff_sentence(cum: np.ndarray, cap: float) -> int:
    """Number of whole sentences whose cumulative length fits under the cap (at least 1)."""
    return max(1, int(np.searchsorted(cum, cap, side="right")))


def fertility(pilot: Path) -> dict | None:
    """Tokens per unit of parallel content, relative to English, from the pilot's aligned items. None if no tokenizer."""
    if not pilot.exists():
        return None
    tok = None
    for name in TOKENIZERS:
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_pretrained(name)
            break
        except Exception:
            continue
    if tok is None:
        return None
    tot = {"en": 0, "tr": 0, "zh": 0}
    for line in pilot.open(encoding="utf-8"):
        it = json.loads(line)
        for l in tot:
            if l in it["text"]:
                text = it["text"][l]["question"] + "\n" + "\n".join(it["text"][l]["options"].values())
                tot[l] += len(tok.encode(text).ids)
    return {l: tot[l] / tot["en"] for l in tot} | {"_tokenizer": name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir"); ap.add_argument("--out", default="stageA/length")
    ap.add_argument("--prefix", default="zhao_long"); ap.add_argument("--instruct", default="none")
    ap.add_argument("--pilot-items", default="data/items_pilot.jsonl")
    a = ap.parse_args()
    work = Path(a.workdir)
    if (work / ".blind").exists():
        sys.exit(f"[blind] {work} carries a .blind marker: hint mention is the primary outcome and may not be scored here.")
    if not (work / "report" / "samples.csv").exists():
        sys.exit("[length] report/samples.csv missing: run `--stage analyze` first")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s_all = pd.read_csv(work / "report" / "samples.csv")
    cond = s_all[s_all.answered & (s_all.prefix == a.prefix) & (s_all.instruct.fillna("none") == a.instruct)]
    s = cond[(cond.cue != "none") & cond.hint_follow].copy()
    ctrl_ids = set(cond[cond.cue == "none"].gen_id)
    want = set(s.gen_id) | ctrl_ids
    cues = sorted(set(s.cue))
    fert = fertility(Path(a.pilot_items))
    extra, floor = {}, {c: {} for c in cues}
    for line in (work / "extracted.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        g = e["gen_id"]
        if g not in want or "error" in e:
            continue
        sents = e.get("sentences") or []
        t_lang = e["cell"]["t_lang"]
        if g in ctrl_ids:
            for c in cues:
                floor[c].setdefault(t_lang, []).append(float(any(patterns(c, t_lang, e["cell"].get("cue_lang") or "en").search(x) for x in sents)))
            continue
        sc = np.asarray(e.get("sentence_chars") or [len(x) for x in sents], dtype=float)
        think_chars = float(sc.sum()); post_chars = float(len(e.get("post") or ""))
        tok = float(e.get("completion_tokens") or 0) * (think_chars / max(1.0, think_chars + post_chars))
        cum = np.cumsum(sc) / max(1.0, think_chars) * tok
        pat = patterns(e["cell"]["cue"], t_lang, e["cell"].get("cue_lang") or "en")
        kw = [i + 1 for i, x in enumerate(sents) if pat.search(x)]
        extra[g] = {"cum_tokens": cum, "cum_fert": cum / fert[t_lang] if fert else None, "kw_sentences": kw,
                    "kw_first": kw[0] if kw else np.nan, "kw_mention": float(bool(kw)), "n": len(sents), "tok": tok,
                    "tok_fert": tok / fert[t_lang] if fert else np.nan}
    s = s[s.gen_id.isin(extra)].copy()
    for col in ("kw_mention", "kw_first", "n", "tok", "tok_fert"):
        s[col] = s.gen_id.map(lambda g: extra[g][col])
    s["n_kw"] = s.gen_id.map(lambda g: len(extra[g]["kw_sentences"]))
    censored = s.judged & (s.input_truncated == True) & s.first_mention.isna() if "input_truncated" in s else pd.Series(False, index=s.index)
    s["j_mention"] = np.where(s.judged & ~censored, s.mentions, np.nan)
    s["j_first"] = np.where(s.judged & ~censored, s.first_mention, np.nan)
    inst = {"judge": ("j_mention", "j_first"), "keyword": ("kw_mention", "kw_first")}
    units = {"sentences": ("n", None), "tokens": ("tok", "cum_tokens")}
    if fert:
        units["tokens_fertility"] = ("tok_fert", "cum_fert")
    tps = s.groupby("t_lang").apply(lambda x: float((x.tok / x.n).median()), include_groups=False)
    no_judge = [c for c in cues if s[(s.cue == c) & s.judged].empty]
    lines = [f"# The length defence — {work}, condition prefix={a.prefix} instruct={a.instruct}", "",
             f"followers: {len(s)} (judged {int(s.judged.sum())}, of which {int(censored.sum())} censored and dropped from the judge "
             "instrument: truncated judge input and no mention found); by language and cue: "
             + ", ".join(f"{t}/{c} {n}" for (t, c), n in s.groupby(['t_lang', 'cue']).size().items()), "",
             "Median think tokens per sentence by trace language: " + ", ".join(f"{l} {v:.1f}" for l, v in tps.items()) + ".", "",
             (f"Fertility on the parallel pilot items ({fert['_tokenizer']}), tokens relative to English: "
              + ", ".join(f"{l} {fert[l]:.3f}" for l in ("en", "tr", "zh")) + "." if fert else
              "No tokenizer could be loaded, so the fertility-scaled token unit is not reported."), "",
             "Keyword false-positive floor (share of NO-HINT traces of this condition on which the cue's keyword list fires): "
             + "; ".join(f"{c}: " + ", ".join(f"{l} {np.mean(v):.2f} (n={len(v)})" for l, v in sorted(floor[c].items())) for c in cues) + ".", ""]
    if no_judge:
        lines += [f"No judge labels yet for cue(s) {', '.join(no_judge)} (twin not run): their judge rows are empty.", ""]

    rows = []
    for cue, sub in s.groupby("cue"):
        en = sub[sub.t_lang == "en"]
        for other in ("tr", "zh"):
            ot = sub[sub.t_lang == other]
            if en.empty or ot.empty:
                continue
            for unit, (lcol, ccol) in units.items():
                shorter = other if ot[lcol].median() <= en[lcol].median() else "en"
                src = ot if shorter == other else en
                caps = [("uncapped", np.inf)] + [(f"p{int(q*100)}", float(src[lcol].quantile(q))) for q in QUANTS]
                for cap_name, cap in caps:
                    def at_cap(df):
                        d = df.copy()
                        if not np.isfinite(cap):
                            d["cut"] = d.n.astype(int)
                        elif ccol is None:
                            d["cut"] = np.minimum(d.n, int(cap)).astype(int)
                        else:
                            d["cut"] = [cutoff_sentence(extra[g][ccol], cap) for g in d.gen_id]
                        d["truncated"] = (d.cut < d.n).astype(float)
                        for name, (mcol, fcol) in inst.items():
                            d[f"{name}_m_cap"] = np.where(d[mcol].isna(), np.nan, ((d[mcol] == 1) & (d[fcol] <= d.cut)).astype(float))
                            d[f"{name}_pos_cap"] = np.where(d[f"{name}_m_cap"] == 1, d[fcol] / d.cut, np.nan)
                        return d
                    e2, o2 = at_cap(en), at_cap(ot)
                    for name in inst:
                        rm = paired_diff(e2.dropna(subset=[f"{name}_m_cap"]), o2.dropna(subset=[f"{name}_m_cap"]), f"{name}_m_cap", rate=True)
                        rp = paired_diff(e2.dropna(subset=[f"{name}_pos_cap"]), o2.dropna(subset=[f"{name}_pos_cap"]), f"{name}_pos_cap")
                        rows.append({"cue": cue, "pair": f"en-{other}", "unit": unit, "cap": cap_name, "cap_from": shorter,
                                     "cap_value": np.nan if not np.isfinite(cap) else round(cap, 1),
                                     "share_truncated_en": float(e2.truncated.mean()), "share_truncated_other": float(o2.truncated.mean()),
                                     "instrument": name, "n_items": rm["n_items"], "mention_en": rm.get("mean_num"), "mention_other": rm.get("mean_den"),
                                     "mention_en_minus_other": rm["median_diff"], "mention_ci_lo": rm["ci_lo"], "mention_ci_hi": rm["ci_hi"],
                                     "pos_items": rp["n_items"], "pos_en_median": rp["median_num"], "pos_other_median": rp["median_den"],
                                     "pos_en_minus_other": rp["median_diff"], "pos_ci_lo": rp["ci_lo"], "pos_ci_hi": rp["ci_hi"]})
    bm = pd.DataFrame(rows)
    bm.to_csv(out / "budget_matching.csv", index=False)
    lines += ["## 1. Budget matching — mention rate and first-mention position after truncating every trace at a cap", "",
              "Caps are percentiles of the shorter language's follower traces in that unit (`cap_from`). Follower status is the full "
              "trace's. `mention_*` are paired means over questions; `pos_*` paired medians of first-mention index over the truncated "
              "length, survivors only (uncapped: index over the whole trace's sentences). Differences are English minus the other "
              "language, bootstrap interval over questions. `share_truncated_*`: the share of each arm the cap actually cuts.", "",
              bm.round(3).to_markdown(index=False) if not bm.empty else "(none)", ""]

    rows = []
    for cue, sub in s.groupby("cue"):
        en = sub[sub.t_lang == "en"]
        for other in ("tr", "zh"):
            ot = sub[sub.t_lang == other]
            for unit, (lcol, ccol) in units.items():
                per_item = []
                for item, oi in ot.groupby("item_id"):
                    ei = en[en.item_id == item]
                    if ei.empty:
                        continue
                    base = []
                    for _, e in ei.iterrows():
                        n = int(e.n); m = int(e.n_kw)
                        for _, o in oi.iterrows():
                            if unit == "sentences":
                                k = min(n, int(o.n))
                            else:
                                k = int(np.clip(round(n * float(o[lcol]) / max(1e-9, float(e[lcol]))), 1, n))
                            base.append(survival(n, k, m))
                    per_item.append({"item_id": item, "en_actual": float(ei.kw_mention.mean()), "en_baseline": float(np.mean(base)),
                                     "other_actual": float(oi.kw_mention.mean()), "n_pairs": len(base)})
                if len(per_item) < 2:
                    continue
                pi = pd.DataFrame(per_item)
                d = (pi.other_actual - pi.en_baseline).values
                rng = np.random.default_rng(20260915)
                boots = np.mean(d[rng.integers(0, len(d), size=(4000, len(d)))], axis=1)
                rows.append({"cue": cue, "pair": f"en-{other}", "unit": unit, "instrument": "keyword", "n_items": len(pi),
                             "en_actual": pi.en_actual.mean(), "en_baseline_at_other_length": pi.en_baseline.mean(),
                             "other_actual": pi.other_actual.mean(), "other_minus_baseline": float(d.mean()),
                             "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
                             "en_minus_other_raw": float((pi.en_actual - pi.other_actual).mean())})
    db = pd.DataFrame(rows)
    db.to_csv(out / "deletion_baseline.csv", index=False)
    lines += ["## 2. Length-matched baselines by random sentence deletion (Little's content-blind control), keyword instrument", "",
              "`en_baseline_at_other_length`: the English keyword-mention rate expected if English traces were shortened to the paired "
              "other-language trace's length by deleting sentences at random (exact survival of at least one keyword sentence, averaged "
              "over every trace pair of the question). `other_minus_baseline` is what remains of the difference once length is removed; "
              "`en_minus_other_raw` is the raw gap in the same direction as table 1. The judge instrument knows only the first mention, "
              "for which survival is exactly k/n, so it cannot support this control without re-judging shortened traces and is not shown. "
              "Read the unethical rows against the keyword floor in the header.", "",
              db.round(3).to_markdown(index=False) if not db.empty else "(none)", ""]

    rows, prow, quint = [], [], []
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        import statsmodels.formula.api as smf
    except Exception:
        BinomialBayesMixedGLM = smf = None
    for cue, sub in s.groupby("cue"):
        d0 = sub.copy(); d0["log_len"] = np.log(d0.n.clip(lower=1)); d0["tr"] = (d0.t_lang == "tr").astype(int); d0["zh"] = (d0.t_lang == "zh").astype(int)
        qbins, edges = pd.qcut(d0.n, 5, labels=False, retbins=True, duplicates="drop")
        d0["quintile"] = [f"q{int(b)+1} [{int(edges[int(b)])},{int(edges[int(b)+1])}]" if pd.notna(b) else "" for b in qbins]
        for name, (mcol, fcol) in inst.items():
            d = d0.dropna(subset=[mcol]).copy(); d["y"] = d[mcol].astype(int)
            d["first_frac"] = d[fcol] / d.n
            for q, dq in d.groupby("quintile"):
                for t, dt in dq.groupby("t_lang"):
                    quint.append({"cue": cue, "instrument": name, "quintile": q, "t_lang": t, "n": len(dt),
                                  "len_median": float(dt.n.median()), "mention_rate": float(dt.y.mean()),
                                  "first_frac_median": float(dt.first_frac.median())})
            zeros = {f"non_mentions_{t}": int(((d.t_lang == t) & (d.y == 0)).sum()) for t in ("en", "tr", "zh")}
            for formula in ("y ~ tr + zh", "y ~ log_len + tr + zh"):
                row = {"cue": cue, "instrument": name, "model": formula, "n": len(d), "n_items": d.item_id.nunique(),
                       "rate": float(d.y.mean()), **zeros}
                if d.empty:
                    row["note"] = "no labels for this cue and instrument"
                elif BinomialBayesMixedGLM is None or d.y.nunique() < 2 or d.item_id.nunique() < 3:
                    row["note"] = "not fitted: statsmodels missing, or no variance in y, or fewer than 3 questions"
                else:
                    if min(zeros.values()) == 0:
                        row["note"] = "a language has zero non-mentions: separated, coefficients are prior artefacts"
                    for fit, tag in (("vb", "vb"), ("map", "map")):
                        try:
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                mdl = BinomialBayesMixedGLM.from_formula(formula, {"item": "0 + C(item_id)"}, d)
                                r = mdl.fit_vb() if fit == "vb" else mdl.fit_map()
                            if any("converge" in str(w.message).lower() for w in caught):
                                row[f"{tag}_converged"] = False
                            for i, nm in enumerate(r.model.exog_names):
                                row[f"{nm}_{tag}"] = float(r.fe_mean[i])
                                if tag == "vb":
                                    row[f"{nm}_vb_sd"] = float(r.fe_sd[i])
                        except Exception as e:
                            row[f"note_{tag}"] = f"{tag} fit failed: {type(e).__name__}"
                rows.append(row)
            dp = d.dropna(subset=["first_frac"])
            prw = {"cue": cue, "instrument": name, "model": "first_frac ~ log_len + tr + zh + (1|item)", "n": len(dp), "n_items": dp.item_id.nunique()}
            if smf is not None and len(dp) > 10 and dp.item_id.nunique() >= 3:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r = smf.mixedlm("first_frac ~ log_len + tr + zh", dp, groups=dp["item_id"]).fit()
                    for nm in r.params.index:
                        prw[f"{nm}_coef"] = float(r.params[nm]); prw[f"{nm}_se"] = float(r.bse[nm])
                except Exception as e:
                    prw["note"] = f"fit failed: {type(e).__name__}"
            else:
                prw["note"] = "not fitted"
            prow.append(prw)
    lr, lp, lq = pd.DataFrame(rows), pd.DataFrame(prow), pd.DataFrame(quint)
    lr.to_csv(out / "length_regression.csv", index=False); lp.to_csv(out / "position_regression.csv", index=False)
    lq.to_csv(out / "length_quintiles.csv", index=False)
    lines += ["## 3. Length as a regressor", "",
              "3a. Mixed logistic mention ~ [log sentences] + Turkish + Chinese + (1 | question); log-odds against English, variational "
              "Bayes (`_vb`, with its SD) and MAP (`_map`) side by side; `map_converged` False marks a Laplace fit that did not converge. `non_mentions_*` counts the zeros per language: where one is zero "
              "the model is separated and the coefficients are prior artefacts, not estimates.", "",
              lr.round(3).to_markdown(index=False) if not lr.empty else "(none)", "",
              "3b. Linear mixed model of the first-mention position (index / sentences) on log sentences and language, random intercept "
              "per question. The gradient the data carry when mention is at ceiling.", "",
              lp.round(3).to_markdown(index=False) if not lp.empty else "(none)", "",
              "Mention rate and median first-mention position by pooled length quintile (sentences) and language:", "",
              lq.round(3).to_markdown(index=False) if not lq.empty else "(none)", ""]
    text = "\n".join(lines)
    (out / "LENGTH-DEFENCE.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"[length] wrote {out/'LENGTH-DEFENCE.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
