#!/usr/bin/env python3
"""The reading rule completed on the two outcomes it never covered (PLAN twenty-eighth addendum, 2026-09-28):
attribution (`chen_verbal`, the judge's mention with reliance) and the first-mention position as a fraction of the
derivation (`first_mention_deriv_frac`, the first mention's sentence index over the first settlement's). Run after the
Stage A results were read, as a completion of the twenty-fourth addendum's rule. Nothing is regenerated or re-judged.

    uv run --locked python scripts/length_defence_extra.py runs/stageA_gpqa [--out stageA/length]

Population and pairing exactly as scripts/length_defence.py: answered hint-following traces of the naming-prefix,
no-instruction cells, each cue separately, judge instrument, censored labels dropped (a truncated judge input with no
mention found); English against Turkish and against Chinese, paired by question with cotlang.phase.paired_diff
(attribution: per-question mean, mean of differences; position: per-question median, median of differences), 4,000
bootstrap resamples of questions. Sentence counts are read from extracted.jsonl as there, so the caps are the same.

Clause (1), budget matching, at the 25th/50th/75th percentiles of the shorter language's follower sentence counts (the
shorter chosen by median; the median cap is the reading). Attribution survives a cut iff the trace was labelled
mention-with-reliance and its first mention lies within the cut; reliance is the whole trace's label, since it cannot be
re-read on a cut trace without re-judging (an assumption). The derivation position of a cut trace is the first mention's
index over min(first settlement, cut), for traces with a settlement whose first mention survives the cut.
Clause (2), the deletion baseline, is not computable for either outcome (keyword lists see mention, not reliance; the
clause is not defined for a position).
Clause (3): attribution by the mixed logistic model of the twenty-fourth addendum (variational fit read, MAP beside,
non-attributions per language printed; read only where every language has some); the derivation position by a linear
mixed model with a random intercept per question and log derivation sentences (the settlement index, the position's own
denominator) as the length term, also fitted with the position capped at 1 (the two must agree), and with log trace
sentences printed beside and not read.
Reading: a difference clear of zero at full length stands iff clause (1) at the median cap keeps its sign with an
interval clear of zero and clause (3)'s coefficient keeps its sign with +-1.96 SE clear of zero; otherwise it is of
undetermined origin. Outputs carry numbers and cell keys only, no text.
"""
from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cotlang.phase import paired_diff

QUANTS = (0.25, 0.5, 0.75)


def clear(lo: float, hi: float) -> bool:
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir"); ap.add_argument("--out", default="stageA/length")
    ap.add_argument("--prefix", default="zhao_long"); ap.add_argument("--instruct", default="none")
    a = ap.parse_args()
    work = Path(a.workdir)
    if (work / ".blind").exists():
        sys.exit(f"[blind] {work} carries a .blind marker")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s_all = pd.read_csv(work / "report" / "samples.csv")
    cond = s_all[s_all.answered & (s_all.prefix == a.prefix) & (s_all.instruct.fillna("none") == a.instruct)]
    s = cond[(cond.cue != "none") & cond.hint_follow].copy()
    want = set(s.gen_id)
    n_sent = {}
    for line in (work / "extracted.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        if e["gen_id"] in want and "error" not in e:
            n_sent[e["gen_id"]] = len(e.get("sentences") or [])
    s = s[s.gen_id.isin(n_sent)].copy()
    s["n"] = s.gen_id.map(n_sent)
    n_mismatch = int((s.n != s.n_sent).sum())
    censored = s.judged & (s.input_truncated == True) & s.first_mention.isna()
    ok = s.judged & ~censored
    s["attr"] = np.where(ok, s.chen_verbal, np.nan)
    s["first"] = np.where(ok, s.first_mention, np.nan)
    s["settle"] = s.commit_regex_sentence
    recomputed = s["first"] / s["settle"]
    both = recomputed.notna() & s.first_mention_deriv_frac.notna() & ok
    n_deriv_mismatch = int((~np.isclose(recomputed[both], s.first_mention_deriv_frac[both])).sum())
    n_deriv_set_mismatch = int((recomputed.notna() & ok).ne(s.first_mention_deriv_frac.notna() & ok).sum())
    n_attr_orphan = int(((s.attr == 1) & ~(s["first"].notna() & (s["first"] <= s.n))).sum())

    lines = [f"# The reading rule on attribution and the derivation position — {work}, prefix={a.prefix} instruct={a.instruct}",
             "", "PLAN twenty-eighth addendum (registered 2026-09-28, commit 0f3f7d5, before any number below existed); run "
             "after the Stage A results were read, as a completion of the twenty-fourth addendum's rule.", "",
             f"followers: {len(s)}; judged and uncensored: {int(ok.sum())}; censored and dropped: {int(censored.sum())}; "
             f"sentence-count mismatches between extracted.jsonl and samples.csv: {n_mismatch}; derivation positions that differ "
             f"from analysis.py's first_mention_deriv_frac: {n_deriv_mismatch} of {int(both.sum())} (traces with a position in "
             f"one and not the other: {n_deriv_set_mismatch}); attribution labels whose first mention is missing or past the "
             f"trace end: {n_attr_orphan}.", "",
             "Attribution under a cut keeps the whole trace's reliance label, as registered. (The post-check note of the "
             "twenty-eighth addendum records that the judge's first reliance sentence would have allowed a stricter rule, and "
             "that it gives the same verdict.)", "",
             "Signs: differences are English minus the other language; regression coefficients are the other language minus "
             "English, so a difference that keeps its sign has a coefficient of the opposite sign. Clause (2), the deletion "
             "baseline, is not computable for either outcome: the keyword lists see mention, not reliance, and the clause is "
             "not defined for a position. The MAP fits are printed beside the variational ones and not read; "
             "`map_converged` False means the fit did not converge.", ""]

    rows = []
    for cue, sub in s.groupby("cue"):
        en = sub[sub.t_lang == "en"]
        for other in ("tr", "zh"):
            ot = sub[sub.t_lang == other]
            if en.empty or ot.empty:
                continue
            shorter = other if ot.n.median() <= en.n.median() else "en"
            src = ot if shorter == other else en
            caps = [("uncapped", np.inf)] + [(f"p{int(q * 100)}", float(src.n.quantile(q))) for q in QUANTS]
            for cap_name, cap in caps:
                def at_cap(df: pd.DataFrame) -> pd.DataFrame:
                    d = df.copy()
                    d["cut"] = d.n.astype(int) if not np.isfinite(cap) else np.minimum(d.n, int(cap)).astype(int)
                    d["truncated"] = (d.cut < d.n).astype(float)
                    survives = d["first"].notna() & (d["first"] <= d.cut)
                    d["attr_cap"] = np.where(d.attr.isna(), np.nan, ((d.attr == 1) & survives).astype(float))
                    has = survives & d.settle.notna() & (d.settle > 0)
                    d["deriv_cap"] = np.where(has, d["first"] / np.minimum(d.settle, d.cut), np.nan)
                    return d
                e2, o2 = at_cap(en), at_cap(ot)
                ra = paired_diff(e2.dropna(subset=["attr_cap"]), o2.dropna(subset=["attr_cap"]), "attr_cap", rate=True)
                rd = paired_diff(e2.dropna(subset=["deriv_cap"]), o2.dropna(subset=["deriv_cap"]), "deriv_cap")
                rows.append({"cue": cue, "pair": f"en-{other}", "cap": cap_name, "cap_from": shorter,
                             "cap_value": np.nan if not np.isfinite(cap) else round(cap, 1),
                             "share_truncated_en": float(e2.truncated.mean()), "share_truncated_other": float(o2.truncated.mean()),
                             "attr_items": ra["n_items"], "attr_en": ra.get("mean_num"), "attr_other": ra.get("mean_den"),
                             "attr_en_minus_other": ra["median_diff"], "attr_ci_lo": ra["ci_lo"], "attr_ci_hi": ra["ci_hi"],
                             "deriv_items": rd["n_items"], "deriv_en_median": rd["median_num"], "deriv_other_median": rd["median_den"],
                             "deriv_en_minus_other": rd["median_diff"], "deriv_ci_lo": rd["ci_lo"], "deriv_ci_hi": rd["ci_hi"]})
    bm = pd.DataFrame(rows)
    bm.to_csv(out / "extra_budget_matching.csv", index=False)

    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        import statsmodels.formula.api as smf
    except Exception:
        sys.exit("[extra] statsmodels is needed for clause (3)")
    reg = []
    for cue, sub in s.groupby("cue"):
        d0 = sub.copy()
        d0["tr"] = (d0.t_lang == "tr").astype(int); d0["zh"] = (d0.t_lang == "zh").astype(int)
        d0["log_len"] = np.log(d0.n.clip(lower=1))
        d = d0.dropna(subset=["attr"]).copy(); d["y"] = d.attr.astype(int)
        zeros = {f"non_attr_{t}": int(((d.t_lang == t) & (d.y == 0)).sum()) for t in ("en", "tr", "zh")}
        for formula in ("y ~ tr + zh", "y ~ log_len + tr + zh"):
            row = {"cue": cue, "outcome": "attribution", "model": formula, "n": len(d), "n_items": d.item_id.nunique(), **zeros}
            if d.y.nunique() < 2 or d.item_id.nunique() < 3:
                row["note"] = "not fitted: no variance or fewer than 3 questions"
            else:
                if min(zeros.values()) == 0:
                    row["note"] = "a language has zero non-attributions: separated, not read"
                for fit in ("vb", "map"):
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            mdl = BinomialBayesMixedGLM.from_formula(formula, {"item": "0 + C(item_id)"}, d)
                            r = mdl.fit_vb() if fit == "vb" else mdl.fit_map()
                        row[f"{fit}_converged"] = not any("converge" in str(w.message).lower() for w in caught)
                        for i, nm in enumerate(r.model.exog_names):
                            row[f"{nm}_{fit}"] = float(r.fe_mean[i])
                            row[f"{nm}_{fit}_se"] = float(r.fe_sd[i])
                    except Exception as ex:
                        row[f"note_{fit}"] = f"{fit} fit failed: {type(ex).__name__}"
            reg.append(row)
        dp = d0[d0["first"].notna() & d0.settle.notna() & (d0.settle > 0)].copy()
        dp["deriv"] = dp["first"] / dp.settle
        dp["deriv_capped"] = dp.deriv.clip(upper=1.0)
        dp["log_dlen"] = np.log(dp.settle.clip(lower=1))
        for outcome, formula, read in (("derivation position", "deriv ~ log_dlen + tr + zh", True),
                                       ("derivation position, capped at 1", "deriv_capped ~ log_dlen + tr + zh", True),
                                       ("derivation position, trace length (not read)", "deriv ~ log_len + tr + zh", False)):
            row = {"cue": cue, "outcome": outcome, "model": formula + " + (1|item)", "n": len(dp), "n_items": dp.item_id.nunique(),
                   "share_after_settlement": float((dp.deriv > 1).mean()) if len(dp) else np.nan, "read": read}
            if len(dp) > 10 and dp.item_id.nunique() >= 3:
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        r = smf.mixedlm(formula, dp, groups=dp["item_id"]).fit()
                    row["converged"] = bool(getattr(r, "converged", True))
                    for nm in r.params.index:
                        row[f"{nm}_coef"] = float(r.params[nm]); row[f"{nm}_se"] = float(r.bse[nm])
                except Exception as ex:
                    row["note"] = f"fit failed: {type(ex).__name__}"
            else:
                row["note"] = "not fitted"
            reg.append(row)
    rg = pd.DataFrame(reg)
    rg.to_csv(out / "extra_regression.csv", index=False)

    verdicts = []
    for _, full in bm[bm.cap == "uncapped"].iterrows():
        other = full.pair.split("-")[1]
        p50 = bm[(bm.cue == full.cue) & (bm.pair == full.pair) & (bm.cap == "p50")].iloc[0]
        for key, label in (("attr", "attribution"), ("deriv", "derivation position")):
            est, lo, hi = full[f"{key}_en_minus_other"], full[f"{key}_ci_lo"], full[f"{key}_ci_hi"]
            if not clear(lo, hi):
                continue
            sign = np.sign(est)
            c1 = clear(p50[f"{key}_ci_lo"], p50[f"{key}_ci_hi"]) and np.sign(p50[f"{key}_en_minus_other"]) == sign
            if key == "attr":
                r = rg[(rg.cue == full.cue) & (rg.outcome == "attribution") & (rg.model == "y ~ log_len + tr + zh")]
                if r.empty or isinstance(r.iloc[0].get("note"), str) or f"{other}_vb" not in r:
                    c3, c3_txt = False, "not identified"
                else:
                    coef, se = r.iloc[0][f"{other}_vb"], r.iloc[0][f"{other}_vb_se"]
                    c3 = bool(np.sign(coef) == -sign and abs(coef) > 1.96 * se)
                    c3_txt = f"{other} {coef:+.2f} (SD {se:.2f})"
            else:
                c3, parts = True, []
                for outcome in ("derivation position", "derivation position, capped at 1"):
                    r = rg[(rg.cue == full.cue) & (rg.outcome == outcome)]
                    if r.empty or f"{other}_coef" not in r or pd.isna(r.iloc[0].get(f"{other}_coef")):
                        c3 = False; parts.append(f"{outcome}: not fitted"); continue
                    if r.iloc[0].get("converged") is not True:
                        c3 = False; parts.append(f"{outcome}: not converged"); continue
                    coef, se = r.iloc[0][f"{other}_coef"], r.iloc[0][f"{other}_se"]
                    c3 = c3 and bool(np.sign(coef) == -sign and abs(coef) > 1.96 * se)
                    parts.append(f"{other} {coef:+.3f} (SE {se:.3f})")
                c3_txt = "; capped: ".join(parts)
            verdicts.append({"cue": full.cue, "difference": f"{label}, EN - {other.upper()}",
                             "full_length": f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                             "budget_matched_p50": f"{p50[f'{key}_en_minus_other']:+.3f} [{p50[f'{key}_ci_lo']:+.3f}, {p50[f'{key}_ci_hi']:+.3f}]",
                             "clause1": c1, "regression": c3_txt, "clause3": c3,
                             "verdict": "stands" if (c1 and c3) else "undetermined"})
    vd = pd.DataFrame(verdicts)
    vd.to_csv(out / "extra_verdicts.csv", index=False)

    lines += ["## Verdicts (every difference clear of zero at full length)", "",
              vd.to_markdown(index=False) if not vd.empty else "(no difference clear of zero at full length)", "",
              "## Clause (1): budget matching", "", bm.round(3).to_markdown(index=False), "",
              "## Clause (3): length as a regressor", "", rg.round(3).to_markdown(index=False), ""]
    (out / "LENGTH-DEFENCE-EXTRA.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    print(vd.to_string(index=False) if not vd.empty else "(no verdict rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
