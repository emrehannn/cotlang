#!/usr/bin/env python3
"""Roadmap step 1a: the blind-safe phase analysis (PH1-PH4), judge-free.

    uv run --locked python scripts/phase_report.py runs/stageA_gpqa --items data/items_gpqa.jsonl --out runs/stageA_gpqa/phase
    uv run --locked python scripts/phase_report.py runs/stageA_gpqa ... --hint-aware     # AFTER unblinding only

What it reads: sentences, per-sentence GlotLID labels, the extracted answer. What it never reads: judged.jsonl, and
-- without --hint-aware -- any hint keyword. It therefore runs on a blind workdir (PLAN fifteenth addendum, rule 1:
the commitment regex, phase_switch, phase compliance, the instruction arm's length contrasts and verification /
backtracking density are blind-safe). It refuses to print or write any disclosure quantity.

--hint-aware is the second pass registered by the twenty-second addendum: the same detector with one more
exclusion, a sentence that refers to the hint (the keyword lists of scripts/keyword_mention.py, fixed before any
label existed). That pass internally scores hint mention, so it refuses a blind workdir; it is the boundary for the
hint cells once the workdir is open, and the disagreement between the two passes is reported.

Outputs (csv, one per table, plus PHASE-REPORT.md): phase_samples.csv (one row per trace, numbers only, no text),
phase_cells.csv, phase_contrasts.csv (instruction arm and cross-language, paired by question), phase_deciles.csv,
phase_verdicts.csv (P1/P2/P4 read against the registered thresholds at the first and the last settlement).
"""
import argparse, json, re, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cotlang import phase
from cotlang.analysis import load_jsonl
from cotlang.items import load_items


def hint_exclusion(cue: str, t_lang: str) -> re.Pattern | None:
    """The hint-aware exclusion for one cell: the registered keyword lists of scripts/keyword_mention.py."""
    if cue == "none":
        return None
    from cotlang import mention
    return mention.patterns_union(cue)


def kind(frame: pd.DataFrame, k: str) -> pd.DataFrame:
    """Rows of one estimator kind; an empty contrast table (a workdir without that arm, e.g. Stage B has no instruction
    arm) has no `kind` column and would raise on attribute access (2026-09-15 Stage B run)."""
    return frame[frame.kind == k] if "kind" in frame else frame


def md(frame: pd.DataFrame, cols=None, **kw) -> str:
    f = frame if cols is None else frame[[c for c in cols if c in frame]]
    return f.round(3).to_markdown(**kw) if not f.empty else "(none)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--items", default="data/items_gpqa.jsonl")
    ap.add_argument("--out", default=None, help="output folder (default <workdir>/phase, or <workdir>/phase_hint_aware)")
    ap.add_argument("--hint-aware", action="store_true", help="exclude hint-referring sentences from the detector (unblinded workdir only)")
    a = ap.parse_args()
    work = Path(a.workdir)
    blind = (work / ".blind").exists()
    if a.hint_aware and blind:
        sys.exit(f"[blind] {work} carries a .blind marker: the hint-aware pass scores hint mention and may not run here. "
                 "Run without --hint-aware (blind-safe), or unblind the workdir first (Stage A's decision).")
    out = Path(a.out) if a.out else work / ("phase_hint_aware" if a.hint_aware else "phase")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[phase] {'BLIND workdir: judge-free, hint-keyword-free pass' if blind else 'unblinded workdir'}; "
          f"{'hint-aware' if a.hint_aware else 'hint-blind'} detector -> {out}", file=sys.stderr)
    items = load_items(Path(a.items))
    ext = [e for e in load_jsonl(work / "extracted.jsonl") if "error" not in e]
    if a.hint_aware:
        rows = []
        for e in ext:
            excl = hint_exclusion(e["cell"]["cue"], e["cell"]["t_lang"])
            rows.append(phase.phase_frame(items, [e], excl))
        pf = pd.concat(rows, ignore_index=True)
    else:
        pf = phase.phase_frame(items, ext)
    pf.to_csv(out / "phase_samples.csv", index=False)
    cells = phase.cell_phase_summary(pf)
    cells.to_csv(out / "phase_cells.csv")
    ic = phase.instruction_contrasts(pf)
    lc = phase.language_contrasts(pf)
    pc = phase.prefix_contrasts(pf)
    contrasts = pd.concat([ic, lc, pc], ignore_index=True)
    contrasts.to_csv(out / "phase_contrasts.csv", index=False)
    dec = phase.decile_table(pf)
    dec.to_csv(out / "phase_deciles.csv", index=False)
    verdicts = pd.DataFrame(phase.read_decision_rules(ic, cells))
    verdicts.to_csv(out / "phase_verdicts.csv", index=False)

    ratio_cols = ["family", "t_lang", "cue", "prefix", "instruct", "num", "den", "metric", "n_items", "median_ratio", "ci_lo", "ci_hi", "share_below_1", "median_num", "median_den"]
    diff_cols = ["family", "t_lang", "cue", "prefix", "instruct", "num", "den", "metric", "n_items", "median_diff", "ci_lo", "ci_hi", "median_num", "median_den"]
    text = [f"# Phase report — {work} ({'blind' if blind else 'unblinded'}; {'hint-aware' if a.hint_aware else 'hint-blind'} detector)", "",
            f"traces: {len(pf)}  answered: {int(pf.answered.sum())}  cells: {pf.cell_key.nunique()}", "",
            "Judge-free throughout. Boundaries: `commit_regex_sentence` = first settled assertion of the SUBMITTED letter; "
            "`last_commit_regex` = the last one; `switch_*` = terminal English run (strict / prose / not-target).", "",
            "## Per cell", "",
            md(cells, ["n_answered", "chars_median", "prose_compliance_mean", "commit_rate", "commit_any_matches_final", "commit_frac_median",
                       "commit_last_frac_median", "n_commit_median", "commit_span_frac_median", "pre_commit_chars_median", "post_commit_chars_median",
                       "switch_rate", "switch_prose_rate", "switch_nt_rate", "switch_nt_frac_median", "switch_nt_after_commit_rate", "n_switch_nt_and_commit",
                       "pre_commit_compliance_mean", "post_commit_compliance_mean", "pre_commit_en_share_mean", "post_commit_en_share_mean"]), "",
            "## Boundary validation: markers per 1,000 characters before / after the commitment", "",
            md(cells, ["n_answered", "verification_pre_per_1k", "verification_post_per_1k", "backtracking_pre_per_1k", "backtracking_post_per_1k",
                       "hedging_pre_per_1k", "hedging_post_per_1k", "commit_lang_en_share"]), "",
            "## Instruction arm vs prefix-only, paired by question (ratios = instruction / prefix-only)", "",
            md(kind(ic, "ratio"), ratio_cols, index=False), "", md(kind(ic, "diff"), diff_cols, index=False), "",
            "## English vs other trace language, paired by question (ratios = EN / other)", "",
            md(kind(lc, "ratio"), ratio_cols, index=False), "", md(kind(lc, "diff"), diff_cols, index=False), "",
            "## Naming prefix vs neutral opener, paired by question (ratios = zhao_long / neutral), no instruction", "",
            md(kind(pc, "ratio"), ratio_cols, index=False) if not pc.empty else "(no neutral cells)", "",
            "## Registered decision rules", "", md(verdicts, index=False), "",
            "## Marker density per decile (pooled per cell, per 1,000 characters)", "", md(dec, index=False), ""]
    (out / "PHASE-REPORT.md").write_text("\n".join(text), encoding="utf-8")
    print(f"[phase] wrote {out/'PHASE-REPORT.md'}", file=sys.stderr)
    for _, r in verdicts.iterrows():
        print("VERDICT " + " ".join(f"{k}={r[k]}" for k in verdicts.columns if k in r and pd.notna(r[k]) and not isinstance(r[k], tuple)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
