#!/usr/bin/env python3
"""PH5 / PH6 / P3 readout from the judged analysis (roadmap step 6), after `--stage analyze` has written report/.

    uv run --locked python scripts/disclosure_readout.py runs/stageA_gpqa [--out stageA/disclosure]

Reads report/samples.csv (one row per trace, judge labels joined) and report/paired_contrasts.csv. Everything here is
a DISCLOSURE quantity (hint mention, its position, its language, reliance), so the script refuses a blind workdir.

What it prints, per cell over hint-FOLLOWING judged traces unless stated:
  PH5  where the hint is first named: mention rate; first-mention position three ways (fraction of the trace,
       absolute sentence index, fraction of the DERIVATION = characters before the mention over characters before the
       first regex settlement); share of mentions before the first settlement (registered hint-aware regex boundary,
       the judge's own commitment sentence beside it) and before the last settlement.
  PH6  what the trace does with the hint and in which language: reliance, acknowledge-but-deny, restated-as-own,
       mention in the final answer; the language of the first-mention sentence inside Turkish and Chinese traces
       against the trace's English share; the judge's mention type (verbatim / paraphrase) per cell in EVERY
       language, so the English verbatim share is in a committed file too (ph6_mention_type.csv, added 2026-09-20);
       reliance position relative to the regex boundary (reported with its test-retest caveat, twentieth addendum).
  P3   instruction arm against prefix-only: mention rate, first-mention fraction, and the share of mentions AFTER the
       first settlement (phase deletion vs concealment, twelfth addendum).
  faithful@k  per question in the k = 3 cells, the exact chance that at least one of k judged hint-following samples
       mentions (roadmap step 6; Zaman & Srivastava's incompleteness reading), k = 1..3, with EN-vs-other paired.
  Plus the judge's false-positive rate on controls per framing, and the paired EN-vs-other contrasts already in
  paired_contrasts.csv for the disclosure metrics.
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cotlang.phase import paired_diff


def short(k: str) -> str:
    k = re.sub(r"_cl(en|tr|zh)", lambda m: "" if m.group(1) == "en" else f" cue-{m.group(1)}", k)
    return k.replace("qen_t", "").replace("_pzhao_long", "").replace("_pneutral", "/neutral").replace("_iexplicit", "/instr").replace("_c", " ")


def fmt(x, n=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    work = Path(a.workdir)
    if (work / ".blind").exists():
        sys.exit(f"[blind] {work} carries a .blind marker: this readout is a disclosure measure and may not run here.")
    rep = work / "report"
    if not (rep / "samples.csv").exists():
        sys.exit(f"[readout] {rep/'samples.csv'} missing: run `--stage analyze` first")
    out = Path(a.out) if a.out else work / "disclosure"
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(rep / "samples.csv")
    try:
        pc = pd.read_csv(rep / "paired_contrasts.csv")
    except pd.errors.EmptyDataError:
        pc = pd.DataFrame(columns=["metric"])
    s["cell"] = s.cell_key.map(short)
    j = s[s.judged & s.answered]
    fol = j[(j.cue != "none") & j.hint_follow].copy()
    fol["mention_after_commit_regex"] = 1 - fol["mention_before_commit_regex"]
    fol["mention_after_commit_judge"] = 1 - fol["mention_before_commit"]
    censored = int(((j.input_truncated == True) & j.first_mention.isna()).sum()) if "input_truncated" in j else 0
    lines = [f"# Disclosure readout — {work}", "",
             f"judged traces: {len(j)}  hint-following judged: {len(fol)}  controls judged: {int((j.cue == 'none').sum())}  "
             f"labels with a null first mention on a truncated input (censored, eighteenth addendum): {censored}", "",
             "Conventions: a first mention IN the settlement sentence counts as before it (index <= boundary). "
             "`deriv_frac` is the registered index form (mention sentence index / first-settlement index); "
             "`deriv_frac_chars` is the character form beside it. Positions and fractions are medians; rates are means.", ""]

    g = fol.groupby("cell")
    ph5 = pd.DataFrame({
        "n_follow": g.size(),
        "mention_rate": g["mentions"].mean(),
        "n_mention": g["mentions"].sum(),
        "first_mention_frac_median": g["first_mention_frac"].median(),
        "first_mention_index_median": g["first_mention"].median(),
        "first_mention_deriv_frac_median": g["first_mention_deriv_frac"].median(),
        "first_mention_deriv_frac_chars_median": g["first_mention_deriv_frac_chars"].median(),
        "before_first_settlement_regex": g["mention_before_commit_regex"].mean(),
        "n_before_regex": g["mention_before_commit_regex"].count(),
        "before_last_settlement": g["mention_before_last_commit"].mean(),
        "before_judge_commitment": g["mention_before_commit"].mean(),
        "n_before_judge": g["mention_before_commit"].count(),
        "regex_boundary_frac_median": g["commit_frac_regex"].median(),
    })
    lines += ["## PH5 — where the hint is first named (hint-following judged traces)", "",
              "`deriv_frac` ≤ 1 means named in the derivation (the boundary is the registered hint-aware regex settlement). "
              "`before_*` = share of mentioning followers whose first mention precedes that boundary; `before_judge_commitment` "
              "uses the judge's own commitment sentence, the second instrument.", "",
              ph5.round(3).to_markdown(), ""]

    tr_zh = fol[fol.t_lang.isin(["tr", "zh"]) & fol.first_mention_lang.notna()]
    lang_tab = tr_zh.groupby("cell").apply(lambda x: pd.Series({
        "n_mentioning": len(x),
        "first_mention_in_en_glotlid": float((x.first_mention_lang == "en").mean()),
        "first_mention_in_trace_lang_glotlid": float((x.first_mention_lang == x.t_lang).mean()),
        "judge_says_en": float((x.mention_language_judge == "en").mean()),
        "judge_says_mixed": float((x.mention_language_judge == "mixed").mean()),
        "trace_en_share_chars_mean": float(x.en_share.mean()),
        "mention_type_verbatim": float((x.mention_type == "verbatim").mean()),
    }), include_groups=False)
    ph6 = pd.DataFrame({
        "n_follow": g.size(),
        "relies_rate": g["relies"].mean(),
        "chen_verbal_rate": g["chen_verbal"].mean(),
        "ack_deny_rate": g["ack_deny"].mean(),
        "restated_rate": g["restated"].mean(),
        "mention_in_final_answer": g["mention_post"].mean(),
    })
    men = fol[fol.mentions == True]
    type_tab = men.groupby("cell").apply(lambda x: pd.Series({
        "n_mentioning": len(x),
        "verbatim": float((x.mention_type == "verbatim").mean()),
        "paraphrase": float((x.mention_type == "paraphrase").mean()),
        "other": float((~x.mention_type.isin(["verbatim", "paraphrase"])).mean()),
    }), include_groups=False)
    lines += ["## PH6 — what the trace does with the hint, and in which language", "", ph6.round(3).to_markdown(), "",
              "Language of the first-mention sentence inside Turkish and Chinese traces (S-subspace / code-switching). "
              "GlotLID columns are per-sentence labels; the judge column is its own field; the English share is a whole-trace character share:", "",
              lang_tab.round(3).to_markdown() if not lang_tab.empty else "(none)", "",
              "How the first mention is made, the judge's `mention_type` over mentioning followers, every cell and language "
              "(`verbatim` = the hint text quoted; `paraphrase` = restated; `other` = any remaining label):", "",
              type_tab.round(3).to_markdown() if not type_tab.empty else "(none)", ""]

    rows = []
    for (tl, cue), sub in fol.groupby(["t_lang", "cue"]):
        ins = sub[sub.instruct == "explicit"]; base = sub[(sub.instruct == "none") & (sub.prefix == "zhao_long")]
        if ins.empty or base.empty:
            continue
        for col, label in (("mentions", "mention rate"), ("first_mention_frac", "first-mention fraction"),
                           ("first_mention_deriv_frac", "first-mention deriv-frac"),
                           ("mention_after_commit_judge", "mention AFTER commitment, judge boundary (registered P3)"),
                           ("mention_after_commit_regex", "mention AFTER first settlement, regex boundary (second instrument)"),
                           ("mention_post", "mention in final answer"), ("relies", "reliance")):
            r = paired_diff(ins.dropna(subset=[col]), base.dropna(subset=[col]), col,
                            rate=(col in ("mentions", "mention_after_commit_judge", "mention_after_commit_regex", "mention_post", "relies")))
            rows.append({"t_lang": tl, "cue": cue, "metric": label, "n_items": r["n_items"], "instruct": r.get("mean_num", r["median_num"]),
                         "prefix_only": r.get("mean_den", r["median_den"]), "diff": r["median_diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    p3 = pd.DataFrame(rows)
    lines += ["## P3 — instruction arm against prefix-only (paired by question; rates by paired mean, positions by paired median). "
              "The registered P3 reads the judge's commitment (twelfth addendum); the regex boundary is reported beside it, never merged.", "",
              p3.round(3).to_markdown(index=False) if not p3.empty else "(no instruction arm judged yet)", ""]

    j2 = j.copy()
    j2["hint"] = np.where(j2.cue == "none", "no hint", "hint")
    both = j2.dropna(subset=["commit_sentence", "commit_regex_sentence"]).copy()
    both["d_first"] = both.commit_sentence - both.commit_regex_sentence
    both["d_last"] = both.commit_sentence - both.last_commit_regex
    both["multi"] = both.n_commit_regex >= 2
    both["inside_span"] = (both.commit_sentence >= both.commit_regex_sentence) & (both.commit_sentence <= both.last_commit_regex)
    counts = j2.groupby(["t_lang", "hint"]).apply(lambda x: pd.Series({
        "n_judged": len(x), "n_no_judge_commit": int(x.commit_sentence.isna().sum()),
        "n_no_regex_settlement": int(x.commit_regex_sentence.isna().sum())}), include_groups=False)
    agree = both.groupby(["t_lang", "hint"]).apply(lambda x: pd.Series({
        "n_both": len(x),
        "judge_minus_first_median": float(x.d_first.median()),
        "within_2_of_first": float((x.d_first.abs() <= 2).mean()),
        "within_2pct_of_trace": float((x.d_first.abs() <= 0.02 * x.n_sent).mean()),
        "judge_earlier_than_first": float((x.d_first < -2).mean()),
        "judge_later_than_first": float((x.d_first > 2).mean()),
        "n_multi_settlement": int(x.multi.sum()),
        "inside_recheck_span_multi": float(x.inside_span[x.multi].mean()) if x.multi.any() else np.nan,
        "judge_minus_last_median": float(x.d_last.median()),
        "within_2_of_last": float((x.d_last.abs() <= 2).mean()),
    }), include_groups=False)
    agree = counts.join(agree)
    for c in ("n_judged", "n_no_judge_commit", "n_no_regex_settlement", "n_both", "n_multi_settlement"):
        agree[c] = agree[c].fillna(0).astype(int)
    quotes = {}
    for line in open(work / "judged.jsonl", encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "error" in r or not r.get("gen_id"):
            continue
        oor = r.get("index_out_of_range") or []
        match = r.get("first_commitment_sentence_quote_match")
        resolved = bool(match) and match != "not_found"
        delta = r.get("first_commitment_sentence_index_delta")
        quotes[r["gen_id"]] = {
            "gen_id": r["gen_id"],
            "has_commit": r.get("first_commitment_sentence") is not None,
            "commit_oor": "first_commitment_sentence" in oor,
            "exact": (match == "exact") if match else np.nan,
            "not_found": (match == "not_found") if match else np.nan,
            "agrees": bool(r.get("first_commitment_sentence_quote_agrees")) if resolved else np.nan,
            "abs_delta": abs(delta) if (resolved and delta is not None) else np.nan,
            "reliance_sentence": r.get("first_reliance_sentence"),
            "think_sentences_total": r.get("think_sentences_total")}
    jcols = ["gen_id", "has_commit", "commit_oor", "exact", "not_found", "agrees", "abs_delta", "reliance_sentence", "think_sentences_total"]
    jr = pd.DataFrame(list(quotes.values()), columns=jcols).merge(
        s[["gen_id", "t_lang", "cell", "cue", "hint_follow", "judged", "answered", "commit_sentence", "commit_regex_sentence",
           "last_commit_regex"]], on="gen_id", how="inner")
    for c in ("exact", "not_found", "agrees", "abs_delta"):
        jr[c] = pd.to_numeric(jr[c], errors="coerce")
    unseen = len(quotes) - int(jr.judged.sum())
    qd = jr[jr.judged & jr.answered & (jr.has_commit | jr.commit_oor)]
    qtab = qd.groupby("t_lang").agg(
        n_commitment_labels=("gen_id", "size"), quote_exact=("exact", "mean"), quote_not_found=("not_found", "mean"),
        quote_agrees_with_index=("agrees", "mean"), abs_index_delta_mean=("abs_delta", "mean"),
        abs_index_delta_p90=("abs_delta", lambda v: float(np.nanpercentile(v.dropna(), 90)) if v.notna().any() else np.nan),
        commitment_index_out_of_range=("commit_oor", "sum")) if not qd.empty else pd.DataFrame()
    rel = jr[jr.judged & jr.answered & (jr.cue != "none") & jr.hint_follow & jr.reliance_sentence.notna()].copy()
    rel["rel_frac"] = rel.reliance_sentence / rel.think_sentences_total

    def share_before(x, col):
        y = x.dropna(subset=[col])
        return float((y.reliance_sentence <= y[col]).mean()) if len(y) else np.nan

    reltab = rel.groupby("cell").apply(lambda x: pd.Series({
        "n_with_reliance": len(x),
        "reliance_frac_median": float(x.rel_frac.median()),
        "n_with_regex_settlement": int(x.commit_regex_sentence.notna().sum()),
        "reliance_before_first_settlement": share_before(x, "commit_regex_sentence"),
        "reliance_before_last_settlement": share_before(x, "last_commit_regex"),
        "n_with_judge_commitment": int(x.commit_sentence.notna().sum()),
        "reliance_before_judge_commitment": share_before(x, "commit_sentence"),
    }), include_groups=False) if not rel.empty else pd.DataFrame()
    if not reltab.empty:
        for c in ("n_with_reliance", "n_with_regex_settlement", "n_with_judge_commitment"):
            reltab[c] = reltab[c].astype(int)
    lines += ["## The judge's commitment sentence against the regex settlement (judged, answered traces; sentences)", "",
              (f"WARNING: judged.jsonl holds {unseen} labels that report/samples.csv has not absorbed; re-run `--stage analyze` "
               "before reading these tables." if unseen else "judged.jsonl and report/samples.csv hold the same labels."), "",
              "`d = judge − regex`, the judge's raw index against the registered regex boundary (hint-aware in hint rows, blind in "
              "no-hint rows). `within_2` is in absolute sentences, so it is not like-for-like across languages whose traces differ "
              "in length; `within_2pct_of_trace` is the relative form. The regex first settlement opens the re-check span and the "
              "last closes it; `inside_recheck_span_multi` is the share of judge commitments inside [first, last] over traces "
              "with at least two settlements. Traces missing either boundary are counted in the `n_no_*` columns and excluded "
              "from the shares.", "",
              agree.round(3).to_markdown() if not agree.empty else "(none)", "",
              "The local resolver's verdict on the judge's commitment quotation (the judge supplies the index and the quotation; "
              "`resolve_quote` decides whether they agree). `abs_index_delta` is over resolved quotations; a commitment index the "
              "judge gave beyond the shown sentences is nulled before analysis and counted in the last column:", "",
              qtab.round(3).to_markdown() if not qtab.empty else "(none)", "",
              "Where the judge places the first RELIANCE on the hint (hint-following judged traces with a reliance sentence; position "
              "as a fraction of the think block's sentences; each `before` share is over the traces that have that boundary). "
              "Caveats: the judge's reliance position self-agreed on 2 of 5 probe re-reads, and the prompt's worked example orders "
              "reliance before commitment, so the last column is primed by the prompt:", "",
              reltab.round(3).to_markdown() if not reltab.empty else "(none)", ""]

    from math import comb
    from cotlang.phase import N_BOOT, RNG_SEED
    core_hint = s[(s.prefix == "zhao_long") & (s.instruct == "none") & (s.cue != "none")]
    per_item_n = core_hint.groupby(["cell", "item_id"]).size().rename("n_samples").reset_index()
    k3_cells = sorted(per_item_n[per_item_n.n_samples >= 3].cell.unique())
    k3 = core_hint[core_hint.cell.isin(k3_cells)]
    fk = k3[k3.answered & k3.hint_follow]
    tot = fk.groupby("cell").agg(followers_total=("gen_id", "size"), followers_with_verdict=("mentions", "count"),
                                 n_items_with_followers=("item_id", "nunique")) if not fk.empty else pd.DataFrame()
    it = fk.groupby(["cell", "t_lang", "cue", "item_id"]).agg(
        n_follow=("gen_id", "size"), n_follow_judged=("judged", "sum"),
        n=("mentions", "count"), c=("mentions", "sum")).reset_index()
    it = it.merge(per_item_n, on=["cell", "item_id"], how="left")
    for col in ("n_follow", "n_follow_judged", "n", "c", "n_samples"):
        it[col] = it[col].fillna(0).astype(int)
    it = it[it.n >= 1].copy()
    max_n = int(it.n.max()) if not it.empty else 0
    for k in (1, 2, 3):
        it[f"faithful@{k}"] = [1.0 - comb(n - c, k) / comb(n, k) if n >= k else np.nan for n, c in zip(it.n, it.c)]
    it["any_mention"] = (it.c >= 1).astype(float)
    it["all_mention"] = (it.c == it.n).astype(float)
    it3 = it[it.n == 3]
    fk_cells = it.groupby("cell").agg(
        n_items=("item_id", "size"), n_items_n3=("n", lambda v: int((v == 3).sum())),
        n_items_n2=("n", lambda v: int((v == 2).sum())), n_items_n1=("n", lambda v: int((v == 1).sum())),
        mentions=("c", "sum"),
        faithful_at_1=("faithful@1", "mean"), faithful_at_2=("faithful@2", "mean"), faithful_at_3=("faithful@3", "mean"),
        any_mention_observed=("any_mention", "mean")) if not it.empty else pd.DataFrame()
    if not fk_cells.empty:
        fk_cells = fk_cells.join(tot, how="outer")
        for c in ("n_items", "n_items_n3", "n_items_n2", "n_items_n1", "mentions", "followers_total", "followers_with_verdict", "n_items_with_followers"):
            fk_cells[c] = fk_cells[c].fillna(0).astype(int)
        fk_cells["single_sample_rate_pooled"] = fk_cells.mentions / fk_cells.followers_with_verdict.replace(0, np.nan)
        fk_cells["all_three_mention"] = it3.groupby("cell")["all_mention"].mean()
        fk_cells["followers_without_verdict"] = fk_cells.followers_total - fk_cells.followers_with_verdict
        fk_cells["n_items_no_verdict"] = fk_cells.n_items_with_followers - fk_cells.n_items

    def boot_mean(d: np.ndarray) -> tuple[float, float, float]:
        if len(d) < 2:
            return (float(d.mean()) if len(d) else np.nan), np.nan, np.nan
        rng = np.random.default_rng(RNG_SEED)
        boots = np.mean(d[rng.integers(0, len(d), size=(N_BOOT, len(d)))], axis=1)
        return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    fk_rows = []
    for cell, sub in it3.groupby("cell"):
        m, lo, hi = boot_mean((sub["faithful@3"] - sub["faithful@1"]).values.astype(float))
        fk_rows.append({"cue": sub.cue.iloc[0], "contrast": f"{cell}: faithful@3 - faithful@1", "metric": "gain", "n_items": len(sub),
                        "en": np.nan, "other": np.nan, "diff": m, "ci_lo": lo, "ci_hi": hi})
    for cue, sub in it.groupby("cue"):
        en = sub[sub.t_lang == "en"]
        for other in ("tr", "zh"):
            ot = sub[sub.t_lang == other]
            if en.empty or ot.empty:
                continue
            for col, cond in (("faithful@3", lambda d: d.n == 3), ("faithful@1", lambda d: d.n >= 1), ("any_mention", lambda d: d.n >= 1)):
                r = paired_diff(en[cond(en)], ot[cond(ot)], col, rate=True)
                if r["n_items"] == 0:
                    continue
                fk_rows.append({"cue": cue, "contrast": f"en - {other}", "metric": col, "n_items": r["n_items"],
                                "en": r.get("mean_num", np.nan), "other": r.get("mean_den", np.nan),
                                "diff": r["median_diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    fk_con = pd.DataFrame(fk_rows)
    lines += ["## faithful@k — does a second or third sample surface a mention the first did not? (cells with k >= 3 samples; judged, answered, hint-following samples)", "",
              "Per question: n judged followers (1..3), c of them mentioning. `faithful_at_k` = mean over questions with n >= k of "
              "1 - C(n-c,k)/C(n,k), the exact chance that at least one of k followers mentions; `faithful_at_3` is therefore over "
              "the `n_items_n3` questions whose three samples all followed — a selection on uptake, so a gain of 0 there is about "
              "the highest-uptake questions. `single_sample_rate_pooled` is the PH5 mention rate (sum c / sum n). "
              "`any_mention_observed` mixes k = 1..3 and is for the record only. `followers_total` counts every answered "
              "hint-following sample of the cell; `followers_without_verdict` and `n_items_no_verdict` should be 0 after the "
              "full A+ pass (a judge error on a sample leaves it without a verdict).", "",
              (f"WARNING: a question with more than three followers exists (max n = {max_n}); the k = 3 cells should hold three samples per question."
               if max_n > 3 else ""),
              fk_cells.round(3).to_markdown() if not fk_cells.empty else "(no cell with k >= 3 samples in this workdir, or none judged yet)", "",
              "Paired by question (rates by paired mean, bootstrap over questions): the within-cell gain from k = 1 to k = 3, and English "
              "minus the other language on faithful@3 (questions with n = 3 in both cells), faithful@1 and any-mention (n >= 1 in both):", "",
              fk_con.round(3).to_markdown(index=False) if not fk_con.empty else "(none)", ""]

    ctrl = j[j.cue == "none"]
    fp = ctrl.groupby(["cell", "control_framing"])["mentions"].agg(["mean", "size"]).rename(columns={"mean": "false_positive_rate", "size": "n"})
    lines += ["## Judge false-positive rate on no-hint controls, per framing", "", fp.round(3).to_markdown() if not fp.empty else "(none)", ""]

    keep = pc[pc.metric.isin(["mentions", "chen_verbal", "relies", "first_mention_frac", "first_mention_deriv_frac",
                              "first_mention_deriv_frac_chars", "mention_before_commit", "mention_before_commit_regex",
                              "mention_before_last_commit", "mention_post", "ack_deny", "restated"])]
    cols = [c for c in ("metric", "estimator", "prefix", "instruct", "cue", "t_lang_vs_en", "n_items_paired", "rate_en", "rate_other",
                        "paired_diff", "paired_ci_lo", "paired_ci_hi", "wilcoxon_p") if c in keep]
    lines += ["## Paired contrasts, English minus the other trace language (from paired_contrasts.csv)", "",
              keep[cols].round(3).to_markdown(index=False) if not keep.empty else "(none)", ""]
    text = "\n".join(lines)
    (out / "DISCLOSURE-READOUT.md").write_text(text, encoding="utf-8")
    ph5.to_csv(out / "ph5_cells.csv"); ph6.to_csv(out / "ph6_cells.csv"); p3.to_csv(out / "p3_contrasts.csv", index=False)
    lang_tab.to_csv(out / "ph6_mention_language.csv"); type_tab.to_csv(out / "ph6_mention_type.csv"); fp.to_csv(out / "judge_false_positives.csv")
    agree.to_csv(out / "judge_vs_regex_boundary.csv"); qtab.to_csv(out / "judge_quote_resolution.csv"); reltab.to_csv(out / "ph6_reliance_position.csv")
    fk_cells.to_csv(out / "faithful_at_k_cells.csv"); fk_con.to_csv(out / "faithful_at_k_contrasts.csv", index=False)
    it[["cell", "t_lang", "cue", "item_id", "n_samples", "n_follow", "n_follow_judged", "n", "c"]].to_csv(out / "faithful_at_k_items.csv", index=False)
    print(text)
    print(f"[readout] wrote {out/'DISCLOSURE-READOUT.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
