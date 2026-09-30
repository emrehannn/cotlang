#!/usr/bin/env python3
"""The paper's figures, built only from the committed CSVs under stageA/, stageB/ and probes/.

    uv run python scripts/figures.py                 # all seven figures, plus paper/figures/FIGURES.md
    uv run python scripts/figures.py --only F3       # re-render one figure; FIGURES.md is still rewritten whole
    uv run python scripts/figures.py --outdir DIR    # somewhere other than paper/figures

Nothing here reads runs/ (git-ignored generation output) or any judged trace: every number comes out of a committed
CSV, and paper/figures/FIGURES.md names the file, the row selector and the column behind each plotted value so a
reader can check every bar against its source. The one exception is F1's first series, which is Onyame et al.'s
published Table 4 and is hard-coded with a comment saying so.

The figures:
  F1  onyame_vs_ours         their per-language baseline-error rate against our per-thinking-language hint-following
  F2  two_by_two             hint-following by prompt language x thinking language, and the paired B - A differences
  F3  first_mention_position where the hint is first named, three ways, with the budget-matched control
  F4  length_by_forcing      trace length by forcing method: cell medians, within-language and English-over-other ratios
  F5  marker_deciles         verification / backtracking / hedging density across the trace
  F6  disclosure_summary     the paired cross-language disclosure contrasts (not used in the paper)
  F7  uptake_forest          paired uptake differences between thinking languages, by forcing method and prompt language
  F8  uptake_cells           uptake per cell with its interval, one dot per thinking language
  F9  length_cells           F4's panel (a) alone: median completion tokens per no-hint cell

The paper uses F8 as Figure 1 and F9 as Figure 2 (Appendix E); F8 replaced F7 and F9 replaced F4 on 2026-09-23, because
their small type and mixed encodings did not read at print size (F4's ratio panels became the paper's Table 7). F1-F7
are kept for the record and are not in the paper.

Style: white background, matplotlib only, one colour per language used identically in every figure, error bars
wherever the source CSV carries an interval, no title text inside the figure (the captions live in the paper).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent

LANG_COLOUR = {
    "en": "#1f5c99",
    "tr": "#c0522a",
    "zh": "#2f7d5d",
}
OTHER_COLOUR = {"sw": "#9a9a9a", "bn": "#5f5f5f"}
LANG_NAME = {"en": "English", "tr": "Turkish", "zh": "Chinese", "sw": "Swahili", "bn": "Bengali"}

GREY = "#555555"
LIGHT = "#dddddd"

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": LIGHT,
    "grid.linewidth": 0.6,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.6,
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.5,
    "svg.fonttype": "none",
    "svg.hashsalt": "cotlang-figures",
}



def read_csv(rel: str) -> list[dict]:
    path = ROOT / rel
    if not path.exists():
        raise SystemExit(f"figures.py: missing source CSV {rel} (looked in {path})")
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v: str | None) -> float | None:
    """A CSV cell as a float, or None when the cell is empty (several of these CSVs leave cells blank)."""
    if v is None or v.strip() == "":
        return None
    return float(v)


def one(rows: list[dict], where: str, **sel) -> dict:
    """The single row matching every key=value in sel. Refuses to guess when zero or several match."""
    hit = [r for r in rows if all(r.get(k, "") == v for k, v in sel.items())]
    if len(hit) != 1:
        raise SystemExit(f"figures.py: {where}: expected exactly 1 row for {sel}, found {len(hit)}")
    return hit[0]



@dataclass
class Row:
    """One plotted value, with where it came from."""
    panel: str
    label: str
    shown: str
    source: str


@dataclass
class Rec:
    key: str
    name: str
    what: str
    width_in: float
    height_in: float
    intended: str
    sources: list[str] = field(default_factory=list)
    values: list[Row] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def src(self, line: str) -> None:
        self.sources.append(line)

    def val(self, panel: str, label: str, shown: str, source: str) -> None:
        self.values.append(Row(panel, label, shown, source))

    def note(self, line: str) -> None:
        self.notes.append(line)

    @property
    def stem(self) -> str:
        return f"{self.key}_{self.name}"



def save(fig, outdir: Path, rec: Rec) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{rec.stem}.png"
    svg = outdir / f"{rec.stem}.svg"
    fig.savefig(png, dpi=300, format="png")
    fig.savefig(svg, format="svg", metadata={"Date": None})
    plt.close(fig)
    return [png, svg]


def lang_legend(fig, langs: list[str], *, extra: list = None, ncol: int = 3) -> None:
    handles = [Patch(facecolor=LANG_COLOUR[l], edgecolor="none", label=LANG_NAME[l]) for l in langs]
    if extra:
        handles += extra
    fig.legend(handles=handles, loc="outside upper center", ncol=ncol,
               handlelength=1.1, handleheight=0.9, columnspacing=1.1, borderaxespad=0.2)


def bar_values(ax, bars, vals, fmt="{:.2f}", fontsize=6.0, pad=1.4, rotation=0):
    ax.bar_label(bars, labels=[fmt.format(v) for v in vals], fontsize=fontsize, padding=pad,
                 color="#333333", rotation=rotation)


def err(v: float, lo: float, hi: float) -> list[list[float]]:
    """matplotlib wants [[below], [above]] distances; a CSV interval is absolute bounds."""
    return [[max(0.0, v - lo)], [max(0.0, hi - v)]]


def dot_rows(ax, rows: list[tuple], *, xline: float, row_labels: list[str], label_fs: float = 6.8):
    """A dot-and-interval panel: one row per label, points offset within the row, the label drawn above the row
    inside the axes (long row names would otherwise eat half the panel as y tick labels)."""
    n = len(row_labels)
    for label, lang, v, lo, hi, off in rows:
        y = (n - 1 - row_labels.index(label)) + off
        ax.errorbar(v, y, xerr=err(v, lo, hi), fmt="o", color=LANG_COLOUR[lang], ecolor=LANG_COLOUR[lang],
                    elinewidth=1.1, capsize=2.2, markersize=4.0, zorder=3)
    ax.axvline(xline, color=GREY, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_yticks([])
    ax.set_ylim(-0.55, n - 1 + 0.78)
    for i, label in enumerate(row_labels):
        ax.text(0.012, (n - 1 - i) + 0.30, label, transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=label_fs, color="#222222")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)



def fig1(outdir: Path, render: bool) -> Rec:
    rec = Rec("F1", "onyame_vs_ours",
              "Onyame et al.'s per-language baseline-error rate (whole prompt translated) beside our "
              "hint-following per thinking language with the English question and hint held fixed.",
              3.4, 4.8, "one column (3.3 in)")

    theirs = [("en", 65.4), ("zh", 70.4), ("sw", 83.0), ("bn", 87.9)]
    rec.src("Onyame et al. 2026 (arXiv 2605.27901), Table 4, Qwen3-8B, simple hint, whole prompt translated — "
            "**hard-coded in scripts/figures.py**, with a comment saying so; it is a published number, not ours.")

    mech = read_csv("stageA/mechanics_cells.csv")
    ours = []
    for lang in ("en", "tr", "zh"):
        key = f"qen_t{lang}_pzhao_long_cmetadata_clen"
        r = one(mech, "stageA/mechanics_cells.csv", cell_key=key, config="stageA_core")
        ours.append((lang, num(r["p_hint"]) * 100.0, key))
    rec.src("stageA/mechanics_cells.csv — column `p_hint` (x 100), rows `cell_key` = "
            "qen_ten/qen_ttr/qen_tzh_pzhao_long_cmetadata_clen, `config` = stageA_core "
            "(the core naming-prefix metadata cells, k = 3, 300 traces each).")

    for lang, v in theirs:
        rec.val("a (theirs)", LANG_NAME[lang], f"{v:.1f}%", "hard-coded, Onyame et al. Table 4")
    for lang, v, key in ours:
        rec.val("b (ours)", LANG_NAME[lang], f"{v:.1f}%", f"mechanics_cells.csv p_hint, {key}")

    rec.note("The two series do not share a denominator. Theirs: the share of hinted trials that select the hinted "
             "answer, with the question, the options and the hint all translated. Ours: the share of answered traces "
             "that choose the hinted letter, with the question and the hint fixed in English and only the planted "
             "first sentence of the trace changed. The axis labels say so and a note under the panels repeats it.")
    rec.note("Chinese keeps its colour in both panels; Swahili and Bengali are greys because they appear only in the "
             "published series and nowhere in our data.")

    if render:
        with plt.rc_context(RC):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(rec.width_in, rec.height_in), layout="constrained")
            for ax, series, xlab in (
                (ax1, theirs, "(a)  Onyame et al., Table 4 — question language\n(whole prompt translated)"),
                (ax2, [(l, v) for l, v, _ in ours], "(b)  this study — thinking language\n(English question and hint)"),
            ):
                langs = [l for l, _ in series]
                vals = [v for _, v in series]
                cols = [LANG_COLOUR.get(l) or OTHER_COLOUR[l] for l in langs]
                b = ax.bar(range(len(vals)), vals, width=0.62, color=cols, edgecolor="white")
                bar_values(ax, b, vals, fmt="{:.1f}")
                ax.set_xticks(range(len(vals)), [LANG_NAME[l] for l in langs])
                ax.set_ylim(0, 100)
                ax.set_yticks([0, 20, 40, 60, 80, 100])
                ax.grid(axis="y")
                ax.grid(axis="x", visible=False)
                ax.set_xlabel(xlab)
            ax1.set_ylabel("Hinted trials selecting\nthe hinted answer (%)")
            ax2.set_ylabel("Answered traces choosing\nthe hinted letter (%)")
            fig.supxlabel("Different denominators. Theirs: hinted trials, whole prompt\n"
                          "translated. Ours: answered traces, English question and hint\n"
                          "fixed. The two panels are not the same measurement.",
                          fontsize=6.2, color=GREY)
            save(fig, outdir, rec)
    return rec



STAGE_B = "Q=tr (Stage B)"
CUES = [("none", "no hint"), ("metadata", "metadata hint"), ("unethical", "unethical hint")]


def fig2(outdir: Path, render: bool) -> Rec:
    rec = Rec("F2", "two_by_two",
              "Hint-following in the four cells of prompt language x thinking language, for the two hints and the "
              "no-hint baseline, with the paired Stage B minus Stage A differences beside them.",
              6.9, 3.1, "full width (6.9 in)")

    mech = read_csv("stageA/mechanics_cells.csv")
    b2x2 = read_csv("stageB/cross_stage/cross_stage_2x2.csv")
    paired = read_csv("stageB/cross_stage/cross_stage_paired.csv")

    rec.src("stageA/mechanics_cells.csv — column `p_hint`, rows `cell_key` = qen_t{en,tr}_pzhao_long_c{none,"
            "metadata,unethical}_clen, `config` = stageA_core. English prompt, k = 3.")
    rec.src("stageB/cross_stage/cross_stage_2x2.csv — column `hint_follow`, rows `stage` = \"Q=tr (Stage B)\", "
            "`t_lang` in {en, tr}, `cue` in {none, metadata, unethical}. Turkish prompt, k = 1. "
            "(This file also carries the Stage A cells, and its Stage A `hint_follow` equals mechanics_cells.csv's "
            "`p_hint` to every digit; Stage A is read from mechanics_cells.csv all the same, as the registered source.)")
    rec.src("stageB/cross_stage/cross_stage_paired.csv — columns `estimate`, `ci_lo`, `ci_hi`, `n_items`, rows "
            "`block` = judge-free, `metric` = hint_follow, `kind` = rate. Paired by question, Stage B minus Stage A.")

    cellsA, cellsB = {}, {}
    for cue, _ in CUES:
        for lang in ("en", "tr"):
            key = f"qen_t{lang}_pzhao_long_c{cue}_clen"
            cellsA[(lang, cue)] = num(one(mech, "mechanics_cells.csv", cell_key=key, config="stageA_core")["p_hint"])
            cellsB[(lang, cue)] = num(one(b2x2, "cross_stage_2x2.csv", stage=STAGE_B, t_lang=lang, cue=cue)["hint_follow"])
            rec.val("a", f"English prompt, {LANG_NAME[lang]} thinking, {dict(CUES)[cue]}",
                    f"{cellsA[(lang, cue)]:.3f}", f"mechanics_cells.csv p_hint, {key}")
            rec.val("a", f"Turkish prompt, {LANG_NAME[lang]} thinking, {dict(CUES)[cue]}",
                    f"{cellsB[(lang, cue)]:.3f}", f"cross_stage_2x2.csv hint_follow, stage={STAGE_B} t_lang={lang} cue={cue}")

    diffs = []
    for cue, cue_name in CUES:
        for lang in ("en", "tr"):
            r = one(paired, "cross_stage_paired.csv", block="judge-free", metric="hint_follow", kind="rate",
                    t_lang=lang, cue=cue)
            d, lo, hi, n = num(r["estimate"]), num(r["ci_lo"]), num(r["ci_hi"]), int(r["n_items"])
            diffs.append((cue, cue_name, lang, d, lo, hi, n))
            rec.val("b", f"{cue_name}, {LANG_NAME[lang]} thinking (B - A)",
                    f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}], n = {n} questions",
                    f"cross_stage_paired.csv estimate/ci_lo/ci_hi, metric=hint_follow t_lang={lang} cue={cue}")

    rec.note("Stage A is k = 3 (three samples per question, 279 to 296 answered traces per cell); Stage B is k = 1 "
             "(97 to 100). Hatched bars are the Turkish prompt.")
    rec.note("Chinese thinking exists only under the English prompt, so it has no place in a 2 x 2 and is left out "
             "of this figure; its Stage A cells are in F1 and F4.")

    if render:
        with plt.rc_context(RC):
            fig, (axa, axb) = plt.subplots(1, 2, figsize=(rec.width_in, rec.height_in), layout="constrained",
                                           gridspec_kw={"width_ratios": [1.25, 1.0]})
            w = 0.20
            offs = {("en", "A"): -1.5 * w, ("tr", "A"): -0.5 * w, ("en", "B"): 0.5 * w, ("tr", "B"): 1.5 * w}
            for i, (cue, _) in enumerate(CUES):
                for lang in ("en", "tr"):
                    for stage, cells, hatch in (("A", cellsA, None), ("B", cellsB, "////")):
                        v = cells[(lang, cue)]
                        bb = axa.bar(i + offs[(lang, stage)], v, width=w, color=LANG_COLOUR[lang],
                                     edgecolor="white", hatch=hatch)
                        axa.bar_label(bb, labels=[f"{v:.2f}"], fontsize=5.6, padding=2.0,
                                      color="#333333", rotation=90)
            axa.set_xticks(range(len(CUES)), [n for _, n in CUES])
            axa.set_ylim(0, 1.0)
            axa.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            axa.set_ylabel("Share of answered traces choosing\nthe hinted letter")
            axa.set_xlabel("(a)  planted cue")
            axa.grid(axis="y"); axa.grid(axis="x", visible=False)

            rows = [(cue_name, lang, d, lo, hi, 0.17 if lang == "en" else -0.17)
                    for cue, cue_name, lang, d, lo, hi, n in diffs]
            dot_rows(axb, rows, xline=0.0, row_labels=[n for _, n in CUES])
            axb.set_xlim(-0.21, 0.24)
            axb.set_xlabel("(b)  Turkish prompt minus English prompt,\n"
                           "paired by question (marker colour is the\n"
                           "thinking language; 95% bootstrap interval)", fontsize=6.8)

            lang_legend(fig, ["en", "tr"],
                        extra=[Patch(facecolor="white", edgecolor="#777777", hatch="////",
                                     label="Turkish prompt (Stage B); plain = English prompt (Stage A)")],
                        ncol=3)
            save(fig, outdir, rec)
    return rec



def fig3(outdir: Path, render: bool) -> Rec:
    rec = Rec("F3", "first_mention_position",
              "Where the metadata hint is first named, for judged followers, measured three ways: as a fraction of "
              "the whole trace, as an absolute sentence index, and as a fraction of the derivation up to the first "
              "settlement. Top row the English prompt, bottom row the Turkish prompt.",
              6.9, 5.0, "full width (6.9 in)")

    a = read_csv("stageA/disclosure/ph5_cells.csv")
    b = read_csv("stageB/disclosure/ph5_cells.csv")
    bm = read_csv("stageA/length/budget_matching.csv")

    rec.src("stageA/disclosure/ph5_cells.csv — columns `first_mention_frac_median`, `first_mention_index_median`, "
            "`first_mention_deriv_frac_median` (and `n_follow`), rows `cell` = \"en metadata\", \"tr metadata\", "
            "\"zh metadata\". Cell medians over judged hint-following traces of the naming-prefix cells.")
    rec.src("stageB/disclosure/ph5_cells.csv — the same three columns, rows `cell` = \"qtr_ten metadata cue-tr\", "
            "\"qtr_ttr metadata cue-tr\".")
    rec.src("stageA/length/budget_matching.csv — columns `pos_en_median`, `pos_other_median`, `cap_value`, "
            "`pos_items`, rows `cue` = metadata, `instrument` = judge, `unit` = sentences, `cap` = p50 "
            "(the shorter language's median length), `pair` in {en-tr, en-zh}.")

    langsA = ["en", "tr", "zh"]
    A = {}
    for l in langsA:
        r = one(a, "stageA ph5_cells.csv", cell=f"{l} metadata")
        A[l] = (num(r["first_mention_frac_median"]), num(r["first_mention_index_median"]),
                num(r["first_mention_deriv_frac_median"]), int(float(r["n_follow"])))
    B = {}
    for l, cell in (("en", "qtr_ten metadata cue-tr"), ("tr", "qtr_ttr metadata cue-tr")):
        r = one(b, "stageB ph5_cells.csv", cell=cell)
        B[l] = (num(r["first_mention_frac_median"]), num(r["first_mention_index_median"]),
                num(r["first_mention_deriv_frac_median"]), int(float(r["n_follow"])))

    caps = {}
    for pair, other in (("en-tr", "tr"), ("en-zh", "zh")):
        r = one(bm, "budget_matching.csv", cue="metadata", instrument="judge", unit="sentences", cap="p50", pair=pair)
        caps[other] = (num(r["pos_en_median"]), num(r["pos_other_median"]), num(r["cap_value"]), int(r["pos_items"]))

    names = ("fraction of the trace", "sentence index", "fraction of the derivation")
    for l in langsA:
        for i, nm in enumerate(names):
            rec.val("top row (English prompt)", f"{LANG_NAME[l]} thinking, {nm}",
                    f"{A[l][i]:.3f}" if i != 1 else f"{A[l][i]:.1f}",
                    f"stageA ph5_cells.csv, cell='{l} metadata', n_follow = {A[l][3]}")
    for other, (en_v, ot_v, cap, n) in caps.items():
        rec.val("top row, budget-matched group", f"English at the {LANG_NAME[other]} median ({cap:.0f} sentences)",
                f"{en_v:.3f}", f"budget_matching.csv pos_en_median, pair=en-{other} cap=p50, n = {n} questions")
        rec.val("top row, budget-matched group", f"{LANG_NAME[other]} at its own median ({cap:.0f} sentences)",
                f"{ot_v:.3f}", f"budget_matching.csv pos_other_median, pair=en-{other} cap=p50, n = {n} questions")
    for l in ("en", "tr"):
        for i, nm in enumerate(names):
            rec.val("bottom row (Turkish prompt)", f"{LANG_NAME[l]} thinking, {nm}",
                    f"{B[l][i]:.3f}" if i != 1 else f"{B[l][i]:.1f}",
                    f"stageB ph5_cells.csv, cell='{'qtr_ten' if l == 'en' else 'qtr_ttr'} metadata cue-tr', "
                    f"n_follow = {B[l][3]}")

    rec.note("The hatched bars in the top-left panel are the budget-matched control: both languages truncated to the "
             "shorter language's median sentence count before the position is re-measured, so English is read at "
             "531 sentences against Turkish and at 896 sentences against Chinese. English therefore has two "
             "budget-matched values, one per pair, and they are not the same number.")
    rec.note("Chinese thinking has no Turkish-prompt cell, so the bottom row has two bars per panel rather than three.")
    rec.note("Stage B budget matching exists (stageB/length/budget_matching.csv) but is not plotted here; the brief "
             "asked for the Stage A control only.")

    if render:
        with plt.rc_context(RC):
            fig, axes = plt.subplots(2, 3, figsize=(rec.width_in, rec.height_in), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.65, 1.0, 1.0]})
            ax = axes[0][0]
            xs = [0, 1, 2, 3.6, 4.6, 6.2, 7.2]
            vals = [A["en"][0], A["tr"][0], A["zh"][0],
                    caps["tr"][0], caps["tr"][1], caps["zh"][0], caps["zh"][1]]
            cols = [LANG_COLOUR["en"], LANG_COLOUR["tr"], LANG_COLOUR["zh"],
                    LANG_COLOUR["en"], LANG_COLOUR["tr"], LANG_COLOUR["en"], LANG_COLOUR["zh"]]
            hatches = [None, None, None, "////", "////", "////", "////"]
            for x, v, c, h in zip(xs, vals, cols, hatches):
                bb = ax.bar(x, v, width=0.8, color=c, edgecolor="white", hatch=h)
                ax.bar_label(bb, labels=[f"{v:.2f}"], fontsize=5.8, padding=1.2, color="#333333")
            ax.set_xticks(xs, ["EN", "TR", "ZH", "EN", "TR", "EN", "ZH"])
            ax.set_xlim(-0.8, 8.0)
            ax.set_ylim(0, 0.42)
            ax.set_ylabel("English prompt (Stage A)\nfirst mention, fraction of trace")
            ax.set_xlabel("(a)  thinking language")
            ax.grid(axis="y"); ax.grid(axis="x", visible=False)
            for centre, txt in ((1.0, "full length"), (4.1, "at Turkish\nmedian"), (6.7, "at Chinese\nmedian")):
                ax.text(centre, 0.405, txt, ha="center", va="top", fontsize=6.2, color="#222222")

            for col, (idx, ylab, ylim, fmt) in enumerate(
                    ((1, "first mention, sentence index", (0, 210), "{:.0f}"),
                     (2, "first mention, fraction of derivation", (0, 1.0), "{:.2f}")), start=1):
                ax = axes[0][col]
                v = [A[l][idx] for l in langsA]
                bb = ax.bar(range(3), v, width=0.62, color=[LANG_COLOUR[l] for l in langsA], edgecolor="white")
                ax.bar_label(bb, labels=[fmt.format(x) for x in v], fontsize=6.0, padding=1.4, color="#333333")
                ax.set_xticks(range(3), ["EN", "TR", "ZH"])
                ax.set_ylim(*ylim)
                ax.set_ylabel(ylab)
                ax.set_xlabel(f"({'bc'[col - 1]})  thinking language")
                ax.grid(axis="y"); ax.grid(axis="x", visible=False)

            langsB = ["en", "tr"]
            for col, (idx, ylab, ylim, fmt) in enumerate(
                    ((0, "Turkish prompt (Stage B)\nfirst mention, fraction of trace", (0, 0.42), "{:.2f}"),
                     (1, "first mention, sentence index", (0, 210), "{:.0f}"),
                     (2, "first mention, fraction of derivation", (0, 1.0), "{:.2f}"))):
                ax = axes[1][col]
                v = [B[l][idx] for l in langsB]
                bb = ax.bar(range(2), v, width=0.5, color=[LANG_COLOUR[l] for l in langsB], edgecolor="white")
                ax.bar_label(bb, labels=[fmt.format(x) for x in v], fontsize=6.0, padding=1.4, color="#333333")
                ax.set_xticks(range(2), ["EN", "TR"])
                ax.set_xlim(-0.6, 1.6)
                ax.set_ylim(*ylim)
                ax.set_ylabel(ylab)
                ax.set_xlabel(f"({'def'[col]})  thinking language")
                ax.grid(axis="y"); ax.grid(axis="x", visible=False)

            lang_legend(fig, langsA,
                        extra=[Patch(facecolor="white", edgecolor="#777777", hatch="////",
                                     label="budget-matched: both truncated to the shorter language's median")],
                        ncol=4)
            save(fig, outdir, rec)
    return rec



def fig4(outdir: Path, render: bool) -> Rec:
    rec = Rec("F4", "length_by_forcing",
              "Trace length by forcing method, no-hint cells: median completion tokens per cell; the paired "
              "within-language ratios (naming prefix over neutral prefix, instruction over naming prefix alone); and the "
              "paired English-over-other token ratios under each forcing method. Was the paper's Figure 2 until 2026-09-23; now "
              "kept for the record (F9 is Figure 2, and the ratios of panels b and c are the paper's Table 7).",
              6.0, 3.0, "full width (6.0 in)")

    pa = read_csv("stageA/phase/phase_cells.csv")
    pb = read_csv("stageB/phase/phase_cells.csv")
    ca = read_csv("stageA/phase/phase_contrasts.csv")
    cb = read_csv("stageB/phase/phase_contrasts.csv")

    rec.src("stageA/phase/phase_cells.csv — column `tokens_median` (and `n_answered`), rows `cell_key` = "
            "qen_t{en,tr,zh}_pneutral_cnone_clen (neutral prefix), qen_t{en,tr,zh}_pzhao_long_cnone_clen (naming "
            "prefix), qen_t{en,tr,zh}_pzhao_long_cnone_clen_iexplicit (naming prefix + instruction). No-hint cells only.")
    rec.src("stageB/phase/phase_cells.csv — column `tokens_median`, rows `cell_key` = "
            "qtr_t{en,tr}_pzhao_long_cnone_cltr (Turkish prompt, naming prefix, no hint).")
    rec.src("stageA/phase/phase_contrasts.csv — columns `median_ratio`, `ci_lo`, `ci_hi`, `n_items`; panel (b) rows "
            "`family` = prefix (num zhao_long, den neutral) and `family` = instruction (num instruct, den prefix_only), "
            "`metric` = tokens, `kind` = ratio, `cue` = none, one per `t_lang`; panel (c) rows `family` = language, "
            "`num` = en, `den` = tr or zh, with `prefix`/`instruct` selecting the forcing method.")
    rec.src("stageB/phase/phase_contrasts.csv — the same columns, `cue` = none, en over tr (the only pair Stage B has).")

    arms = [("neutral\nprefix", "pneutral", ""),
            ("naming\nprefix", "pzhao_long", ""),
            ("naming +\ninstruction", "pzhao_long", "_iexplicit")]
    cells = {}
    for arm, pfx, suf in arms:
        for l in ("en", "tr", "zh"):
            key = f"qen_t{l}_{pfx}_cnone_clen{suf}"
            r = one(pa, "stageA phase_cells.csv", cell_key=key)
            cells[(arm, l)] = (num(r["tokens_median"]), int(r["n_answered"]), key, "stageA/phase/phase_cells.csv")
    tp = "Turkish\nprompt"
    for l in ("en", "tr"):
        key = f"qtr_t{l}_pzhao_long_cnone_cltr"
        r = one(pb, "stageB phase_cells.csv", cell_key=key)
        cells[(tp, l)] = (num(r["tokens_median"]), int(r["n_answered"]), key, "stageB/phase/phase_cells.csv")
    groups = [a for a, _, _ in arms] + [tp]
    for g in groups:
        for l in ("en", "tr", "zh"):
            if (g, l) in cells:
                v, n, key, f = cells[(g, l)]
                rec.val("a", f"{g.replace(chr(10), ' ')}, {LANG_NAME[l]} thinking",
                        f"{v:,.0f} tokens", f"{f} tokens_median, {key}, n_answered = {n}")

    within = []
    for lab, fam, nm, dn in (("naming prefix /\nneutral prefix", "prefix", "zhao_long", "neutral"),
                             ("+ instruction /\nnaming prefix", "instruction", "instruct", "prefix_only")):
        for l in ("en", "tr", "zh"):
            r = one(ca, "stageA phase_contrasts.csv", family=fam, metric="tokens", kind="ratio", cue="none",
                    t_lang=l, num=nm, den=dn)
            within.append((lab, l, num(r["median_ratio"]), num(r["ci_lo"]), num(r["ci_hi"]), int(r["n_items"])))
            rec.val("b", f"{lab.replace(chr(10), ' ')}, {LANG_NAME[l]} thinking",
                    f"{num(r['median_ratio']):.2f} [{num(r['ci_lo']):.2f}, {num(r['ci_hi']):.2f}], n = {r['n_items']}",
                    f"stageA/phase/phase_contrasts.csv family={fam} t_lang={l} num={nm} den={dn} (median_ratio, ci_lo, ci_hi)")

    ratios = []
    for lab, prefix, instruct in (("neutral\nprefix", "neutral", "none"), ("naming\nprefix", "zhao_long", "none"),
                                  ("naming +\ninstruction", "zhao_long", "explicit")):
        for den in ("tr", "zh"):
            r = one(ca, "stageA phase_contrasts.csv", family="language", metric="tokens", kind="ratio",
                    cue="none", num="en", den=den, prefix=prefix, instruct=instruct)
            ratios.append((lab, den, num(r["median_ratio"]), num(r["ci_lo"]), num(r["ci_hi"]), int(r["n_items"]),
                           f"stageA/phase/phase_contrasts.csv prefix={prefix} instruct={instruct} num=en den={den}"))
    r = one(cb, "stageB phase_contrasts.csv", family="language", metric="tokens", kind="ratio", cue="none",
            num="en", den="tr", prefix="zhao_long", instruct="none")
    ratios.append((tp, "tr", num(r["median_ratio"]), num(r["ci_lo"]), num(r["ci_hi"]),
                   int(r["n_items"]), "stageB/phase/phase_contrasts.csv prefix=zhao_long instruct=none num=en den=tr"))
    for lab, den, v, lo, hi, n, src in ratios:
        rec.val("c", f"{lab.replace(chr(10), ' ')}, English / {LANG_NAME[den]} tokens",
                f"{v:.2f} [{lo:.2f}, {hi:.2f}], n = {n} questions", src + " (median_ratio, ci_lo, ci_hi)")

    rec.note("Panels (b) and (c) use the paper's estimator: each question contributes the median of its samples, the "
             "ratio is formed within the question, and the statistic is the median of the per-question ratios with a "
             "95% bootstrap interval. They are therefore not quotients of the cell medians in panel (a).")
    rec.note("In panel (c) the numerator is English throughout, so the marker takes the colour of the denominator "
             "language. Both ratio panels use a log axis, so a halving and a doubling are the same distance from 1.")

    if render:
        with plt.rc_context(RC):
            fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(rec.width_in, rec.height_in), layout="constrained",
                                                gridspec_kw={"width_ratios": [1.25, 0.8, 0.95]})
            w = 0.25
            for i, g in enumerate(groups):
                present = [l for l in ("en", "tr", "zh") if (g, l) in cells]
                start = -(len(present) - 1) / 2.0
                for j, l in enumerate(present):
                    v = cells[(g, l)][0]
                    hatch = "////" if g == tp else None
                    bb = axa.bar(i + (start + j) * w, v / 1000.0, width=w, color=LANG_COLOUR[l],
                                 edgecolor="white", hatch=hatch)
                    axa.bar_label(bb, labels=[f"{v / 1000.0:.0f}"], fontsize=6.4, padding=1.0, color="#333333")
            axa.set_xticks(range(len(groups)), groups)
            axa.tick_params(axis="x", labelsize=6.4)
            axa.set_ylim(0, 25)
            axa.set_ylabel("Median completion tokens (thousands)")
            axa.set_xlabel("(a)  cell medians")
            axa.grid(axis="y"); axa.grid(axis="x", visible=False)

            import matplotlib.ticker as mt
            for ax in (axb, axc):
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(mt.FixedLocator([0.5, 0.7, 1, 1.5, 2, 3, 4]))
                ax.xaxis.set_major_formatter(mt.FixedFormatter(["0.5", "0.7", "1", "1.5", "2", "3", "4"]))
                ax.xaxis.set_minor_locator(mt.NullLocator())
            wl = ["naming prefix /\nneutral prefix", "+ instruction /\nnaming prefix"]
            off = {"en": 0.22, "tr": 0.0, "zh": -0.22}
            dot_rows(axb, [(lab.replace("\n", " "), l, v, lo, hi, off[l]) for lab, l, v, lo, hi, n in within],
                     xline=1.0, row_labels=[x.replace("\n", " ") for x in wl], label_fs=6.2)
            axb.set_xlim(0.5, 4.2)
            axb.set_xlabel("(b)  same language:\nforcing ratio")
            rows = [(lab.replace("\n", " "), den, v, lo, hi, 0.15 if den == "tr" else -0.15)
                    for lab, den, v, lo, hi, n, _ in ratios]
            dot_rows(axc, rows, xline=1.0, row_labels=[g.replace("\n", " ") for g in groups], label_fs=6.4)
            axc.set_xlim(0.5, 3.0)
            axc.set_xlabel("(c)  English over the other\nlanguage (colour: the other)")

            lang_legend(fig, ["en", "tr", "zh"],
                        extra=[Patch(facecolor="white", edgecolor="#777777", hatch="////",
                                     label="Turkish prompt (Stage B)")],
                        ncol=4)
            save(fig, outdir, rec)
    return rec



FORCING_COLOUR = {"naming": "#333333", "instruction": "#7b5aa6", "neutral": "#b8860b"}


def fig7(outdir: Path, render: bool) -> Rec:
    rec = Rec("F7", "uptake_forest",
              "Paired uptake differences between thinking languages with 95% bootstrap intervals: (a) English question, "
              "by forcing method, with the neutral-minus-naming interaction; (b) Turkish prompt minus English prompt, "
              "naming prefix, per thinking language and hint. Was the paper's Figure 1 until 2026-09-23; now kept for the "
              "record (F8 replaced it).",
              6.0, 3.4, "full width (6.0 in)")
    ua = read_csv("stageA/uptake_by_forcing.csv")
    ub = read_csv("stageB/cross_stage/cross_stage_uptake.csv")
    rec.src("stageA/uptake_by_forcing.csv (written by scripts/uptake_by_forcing.py) — columns `quantity`, "
            "`n_questions`, `estimate`, `ci_lo`, `ci_hi`; one row per plotted contrast, selected by `quantity`.")
    rec.src("stageB/cross_stage/cross_stage_uptake.csv — columns `t_lang`, `cue`, `n_items`, `b_minus_a`, `ci_lo`, "
            "`ci_hi` (uptake under the Turkish prompt minus uptake under the English prompt, paired by question).")

    a_rows = [
        ("naming, metadata, EN − TR", "uptake gap EN-TR, paired: naming prefix (k = 3)", "naming", "metadata"),
        ("naming, metadata, EN − ZH", "uptake gap EN-ZH, paired: naming prefix (k = 3), metadata", "naming", "metadata"),
        ("naming, metadata, TR − ZH", "uptake gap TR-ZH, paired: naming prefix (k = 3), metadata", "naming", "metadata"),
        ("naming, unethical, EN − TR", "uptake gap EN-TR, paired: naming prefix (k = 3), unethical", "naming", "unethical"),
        ("naming, unethical, EN − ZH", "uptake gap EN-ZH, paired: naming prefix (k = 3), unethical", "naming", "unethical"),
        ("naming, unethical, TR − ZH", "uptake gap TR-ZH, paired: naming prefix (k = 3), unethical", "naming", "unethical"),
        ("naming + instruction, metadata, EN − TR", "uptake gap EN-TR, paired: naming prefix + instruction (k = 1)",
         "instruction", "metadata"),
        ("neutral, metadata, EN − TR", "uptake gap EN-TR, paired: neutral prefix (k = 1)", "neutral", "metadata"),
        ("neutral gap − naming gap", "interaction: neutral gap minus naming gap, paired", "neutral", "interaction"),
    ]
    A = []
    for lab, q, forcing, hint in a_rows:
        r = one(ua, "stageA uptake_by_forcing.csv", quantity=q)
        k = "k = 3" if forcing == "naming" else ("k = 1 vs 3" if hint == "interaction" else "k = 1")
        v, lo, hi, n = num(r["estimate"]), num(r["ci_lo"]), num(r["ci_hi"]), int(r["n_questions"])
        A.append((lab, forcing, hint, v, lo, hi, n, k))
        rec.val("a", lab, f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}], n = {n}, {k}",
                f"stageA/uptake_by_forcing.csv quantity = \"{q}\"")
    B = []
    for l, cue in (("en", "metadata"), ("en", "unethical"), ("tr", "metadata"), ("tr", "unethical")):
        r = one(ub, "stageB cross_stage_uptake.csv", t_lang=l, cue=cue)
        v, lo, hi, n = num(r["b_minus_a"]), num(r["ci_lo"]), num(r["ci_hi"]), int(r["n_items"])
        lab = f"{LANG_NAME[l]}, {cue}"
        B.append((lab, l, cue, v, lo, hi, n))
        rec.val("b", lab, f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}], n = {n}, k = 1 against k = 3",
                f"stageB/cross_stage/cross_stage_uptake.csv t_lang = {l}, cue = {cue}")
    rec.note("Uptake is hint-following minus the no-hint rate of choosing the hinted letter, same thinking language "
             "and forcing method, per question; intervals are percentile bootstraps over questions, unadjusted.")
    rec.note("Panel (a) colour is the forcing method (dark grey naming prefix, purple naming prefix + instruction, "
             "ochre neutral prefix); marker shape is the hint (circle metadata, square unethical, diamond the "
             "interaction). Panel (b) colour is the thinking language, as in every other figure.")

    if render:
        marker = {"metadata": "o", "unethical": "s", "interaction": "D"}
        with plt.rc_context(RC):
            fig, (axa, axb) = plt.subplots(1, 2, figsize=(rec.width_in, rec.height_in), layout="constrained",
                                           gridspec_kw={"width_ratios": [1.3, 0.75]})
            n = len(A)
            ys, labs = [], []
            for i, (lab, forcing, hint, v, lo, hi, nq, k) in enumerate(A):
                y = n - 1 - i - (0.4 if i >= 6 else 0) - (0.4 if i >= 8 else 0)
                c = "#000000" if hint == "interaction" else FORCING_COLOUR[forcing]
                axa.errorbar(v, y, xerr=err(v, lo, hi), fmt=marker[hint], color=c, ecolor=c, elinewidth=1.1,
                             capsize=2.0, markersize=4.0, markerfacecolor=c if hint != "interaction" else "white",
                             zorder=3)
                ys.append(y); labs.append(f"{lab} ({nq} q)")
            axa.axvline(0, color=GREY, linestyle="--", linewidth=0.8, zorder=0)
            axa.set_xlim(-0.2, 0.36)
            axa.set_ylim(-1.3, n - 0.4)
            axa.set_yticks(ys, labs, fontsize=6.4)
            axa.tick_params(axis="y", length=0)
            axa.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
            axa.set_xlabel("(a)  English question, by forcing method\n(interaction: k = 1 vs 3)")
            axa.grid(axis="x"); axa.grid(axis="y", visible=False)

            m = len(B)
            for i, (lab, l, cue, v, lo, hi, nq) in enumerate(B):
                y = m - 1 - i
                axb.errorbar(v, y, xerr=err(v, lo, hi), fmt=marker[cue], color=LANG_COLOUR[l], ecolor=LANG_COLOUR[l],
                             elinewidth=1.1, capsize=2.0, markersize=4.0, zorder=3)
            axb.axvline(0, color=GREY, linestyle="--", linewidth=0.8, zorder=0)
            axb.set_xlim(-0.25, 0.25)
            axb.set_ylim(-0.7, m - 0.3)
            axb.set_yticks([m - 1 - i for i in range(m)], [f"{b[0]} ({b[6]} q)" for b in B], fontsize=6.4)
            axb.tick_params(axis="y", length=0)
            axb.set_xticks([-0.2, 0, 0.2])
            axb.set_xlabel("(b)  Turkish − English\nprompt (k = 1 vs 3)", loc="right")
            axb.grid(axis="x"); axb.grid(axis="y", visible=False)
            handles = [Line2D([], [], color=FORCING_COLOUR["naming"], marker="o", linestyle="", label="naming prefix (k = 3)"),
                       Line2D([], [], color=FORCING_COLOUR["instruction"], marker="o", linestyle="", label="naming prefix + instruction (k = 1)"),
                       Line2D([], [], color=FORCING_COLOUR["neutral"], marker="o", linestyle="", label="neutral prefix (k = 1)"),
                       Line2D([], [], color="#777777", marker="o", linestyle="", label="metadata"),
                       Line2D([], [], color="#777777", marker="s", linestyle="", label="unethical"),
                       Line2D([], [], color="#000000", marker="D", markerfacecolor="white", linestyle="", label="interaction")]
            fig.legend(handles=handles, loc="outside upper center", ncol=3, handlelength=1.0, columnspacing=1.0)
            save(fig, outdir, rec)
    return rec



RC_PRINT = {**RC, "font.size": 9.5, "axes.labelsize": 9.5, "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0, "legend.title_fontsize": 9.0}


def fig8(outdir: Path, render: bool) -> Rec:
    rec = Rec("F8", "uptake_cells",
              "Uptake per cell (hint-following minus the no-hint rate) with a 95% bootstrap interval over questions, one "
              "row per hint, forcing method and prompt language, one dot per thinking language. The paper's Figure 1.",
              6.0, 3.5, "full width (6.0 in)")
    ua = read_csv("stageA/uptake_cells.csv")
    ub = read_csv("stageB/uptake_cells.csv")
    rec.src("stageA/uptake_cells.csv (written by scripts/uptake_cells.py) — columns `uptake`, `ci_lo`, `ci_hi`, "
            "`answered`; rows selected by `cue`, `forcing` and `t_lang` (English prompt).")
    rec.src("stageB/uptake_cells.csv (the same script) — the same columns, Turkish prompt, naming prefix, English and "
            "Turkish thinking.")
    layout = [
        ("Metadata hint", None, None, None),
        ("naming prefix  (k = 3)", ua, "metadata", "naming prefix"),
        ("Turkish prompt, naming prefix  (k = 1)", ub, "metadata", "naming prefix"),
        ("naming prefix + instruction  (k = 1)", ua, "metadata", "naming prefix + instruction"),
        ("neutral prefix  (k = 1)", ua, "metadata", "neutral prefix"),
        ("Unethical hint", None, None, None),
        ("naming prefix  (k = 3)", ua, "unethical", "naming prefix"),
        ("Turkish prompt, naming prefix  (k = 1)", ub, "unethical", "naming prefix"),
        ("Sycophancy hint", None, None, None),
        ("naming prefix  (k = 1)", ua, "sycophancy", "naming prefix"),
    ]
    pts = []
    for y, (lab, rows, cue, forcing) in enumerate(layout):
        if cue is None:
            continue
        stage = "stageB" if rows is ub else "stageA"
        for l in ("en", "tr", "zh"):
            hit = [r for r in rows if r["cue"] == cue and r["forcing"] == forcing and r["t_lang"] == l]
            if not hit:
                continue
            r = one(rows, f"{stage} uptake_cells.csv", cue=cue, forcing=forcing, t_lang=l)
            v, lo, hi = num(r["uptake"]), num(r["ci_lo"]), num(r["ci_hi"])
            pts.append((y, l, v, lo, hi))
            rec.val("-", f"{cue}, {lab.split('  ')[0]}, {LANG_NAME[l]} thinking",
                    f"{v:.3f} [{lo:.3f}, {hi:.3f}], {r['answered']} answered traces",
                    f"{stage}/uptake_cells.csv cue={cue} forcing={forcing} t_lang={l} (uptake, ci_lo, ci_hi)")
    rec.note("Uptake is the cell's hint-following minus the no-hint rate of choosing the hinted letter in the cell with "
             "the same prompt, thinking language and forcing method (for sycophancy, the three-sample naming-prefix "
             "no-hint cell); the values are Table 3's. Intervals resample questions and are per cell: overlap between "
             "them is not a test, because the differences between languages are paired by question (uptake_by_forcing.csv, "
             "cross_stage_uptake.csv).")
    rec.note("Colour is the thinking language, as in every other figure; within a row English is drawn highest.")

    if render:
        from matplotlib.ticker import PercentFormatter
        with plt.rc_context(RC_PRINT):
            fig, ax = plt.subplots(figsize=(rec.width_in, rec.height_in), layout="constrained")
            dodge = {"en": -0.2, "tr": 0.0, "zh": 0.2}
            for y, l, v, lo, hi in pts:
                ax.errorbar(v, y + dodge[l], xerr=err(v, lo, hi), fmt="o", color=LANG_COLOUR[l], ecolor=LANG_COLOUR[l],
                            elinewidth=1.2, capsize=0, markersize=5.0, zorder=3)
            ax.set_yticks(range(len(layout)), [lab for lab, *_ in layout])
            for tl, (_, _, cue, _) in zip(ax.get_yticklabels(), layout):
                if cue is None:
                    tl.set_fontweight("bold")
            ax.tick_params(axis="y", length=0)
            ax.set_ylim(len(layout) - 0.4, -0.6)
            ax.set_xlim(0, 0.92)
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.set_xlabel("uptake (hint-following minus the no-hint rate)")
            ax.grid(axis="x"); ax.grid(axis="y", visible=False)
            handles = [Line2D([], [], marker="o", linestyle="", color=LANG_COLOUR[l], label=LANG_NAME[l])
                       for l in ("en", "tr", "zh")]
            fig.legend(handles=handles, loc="outside upper center", ncol=3, title="thinking language",
                       handletextpad=0.3, columnspacing=1.6)
            save(fig, outdir, rec)
    return rec



def fig9(outdir: Path, render: bool) -> Rec:
    rec = Rec("F9", "length_cells",
              "Median completion tokens per no-hint cell, by forcing method and thinking language, with the Turkish "
              "prompt hatched: panel (a) of F4 on its own, in print-size type. The paper's Figure 2 (Appendix E); the "
              "paired ratios of F4 (b) and (c) are the paper's Table 7.",
              6.0, 2.7, "full width (6.0 in)")
    pa = read_csv("stageA/phase/phase_cells.csv")
    pb = read_csv("stageB/phase/phase_cells.csv")
    rec.src("stageA/phase/phase_cells.csv — column `tokens_median` (and `n_answered`), rows `cell_key` = "
            "qen_t{en,tr,zh}_pneutral_cnone_clen, qen_t{en,tr,zh}_pzhao_long_cnone_clen and "
            "qen_t{en,tr,zh}_pzhao_long_cnone_clen_iexplicit.")
    rec.src("stageB/phase/phase_cells.csv — column `tokens_median`, rows `cell_key` = qtr_t{en,tr}_pzhao_long_cnone_cltr.")
    groups = [("neutral prefix", "stageA", "qen_t{l}_pneutral_cnone_clen", ("en", "tr", "zh")),
              ("naming prefix", "stageA", "qen_t{l}_pzhao_long_cnone_clen", ("en", "tr", "zh")),
              ("naming prefix\n+ instruction", "stageA", "qen_t{l}_pzhao_long_cnone_clen_iexplicit", ("en", "tr", "zh")),
              ("Turkish prompt,\nnaming prefix", "stageB", "qtr_t{l}_pzhao_long_cnone_cltr", ("en", "tr"))]
    bars = []
    for i, (g, stage, tmpl, langs) in enumerate(groups):
        for j, l in enumerate(langs):
            key = tmpl.format(l=l)
            r = one(pa if stage == "stageA" else pb, f"{stage} phase_cells.csv", cell_key=key)
            v = num(r["tokens_median"])
            bars.append((i, j, len(langs), l, v, stage == "stageB"))
            rec.val("-", f"{g.replace(chr(10), ' ')}, {LANG_NAME[l]} thinking", f"{v:,.0f} tokens",
                    f"{stage}/phase/phase_cells.csv tokens_median, {key}, n_answered = {r['n_answered']}")
    rec.note("Cell medians; the paired within-question ratios the paper reads (Table 7) are not quotients of these bars.")

    if render:
        with plt.rc_context(RC_PRINT):
            fig, ax = plt.subplots(figsize=(rec.width_in, rec.height_in), layout="constrained")
            w = 0.26
            for i, j, m, l, v, hatched in bars:
                x = i + (j - (m - 1) / 2.0) * w
                bb = ax.bar(x, v / 1000.0, width=w, color=LANG_COLOUR[l], edgecolor="white",
                            hatch="////" if hatched else None)
                ax.bar_label(bb, labels=[f"{v / 1000.0:.1f}"], fontsize=8.0, padding=1.5, color="#333333")
            ax.set_xticks(range(len(groups)), [g for g, *_ in groups])
            ax.tick_params(axis="x", length=0)
            ax.set_ylim(0, 25)
            ax.set_ylabel("median tokens (thousands)")
            ax.grid(axis="y"); ax.grid(axis="x", visible=False)
            lang_legend(fig, ["en", "tr", "zh"], ncol=3)
            save(fig, outdir, rec)
    return rec



def fig5(outdir: Path, render: bool) -> Rec:
    rec = Rec("F5", "marker_deciles",
              "Verification, backtracking and hedging marker density per 1,000 characters across the ten deciles of "
              "the trace, for the three no-hint naming-prefix cells and the neutral-opener English cell.",
              6.9, 2.9, "full width (6.9 in)")

    dec = read_csv("stageA/phase/phase_deciles.csv")
    rec.src("stageA/phase/phase_deciles.csv — columns `decile`, `verification_per_1k`, `backtracking_per_1k`, "
            "`hedging_per_1k`, rows `cell_key` = qen_t{en,tr,zh}_pzhao_long_cnone_clen (naming prefix, no hint) and "
            "qen_ten_pneutral_cnone_clen (neutral opener, English, no hint). Each decile pools the characters of "
            "every trace in the cell (column `chars`), so a decile's density is a cell-level rate, not a mean of "
            "per-trace rates; the CSV carries no interval, so there are no error bars here.")

    series = [("en", "naming prefix, English", "-", "o"),
              ("tr", "naming prefix, Turkish", "-", "s"),
              ("zh", "naming prefix, Chinese", "-", "^"),
              ("en", "neutral opener, English", "--", "o")]
    keys = ["qen_ten_pzhao_long_cnone_clen", "qen_ttr_pzhao_long_cnone_clen",
            "qen_tzh_pzhao_long_cnone_clen", "qen_ten_pneutral_cnone_clen"]
    families = [("verification_per_1k", "verification"), ("backtracking_per_1k", "backtracking"),
                ("hedging_per_1k", "hedging")]

    data = {}
    for key in keys:
        rows = sorted([r for r in dec if r["cell_key"] == key], key=lambda r: int(r["decile"]))
        if len(rows) != 10:
            raise SystemExit(f"figures.py: phase_deciles.csv: expected 10 deciles for {key}, found {len(rows)}")
        data[key] = rows

    for col, fam in families:
        for key, (_, lab, _, _) in zip(keys, series):
            vs = [num(r[col]) for r in data[key]]
            rec.val(fam, lab, " ".join(f"{v:.2f}" for v in vs) + "   (deciles 1 to 10)",
                    f"phase_deciles.csv {col}, cell_key = {key}")

    rec.note("The neutral-opener English cell is the same colour as English but dashed with open markers, because it "
             "differs from the solid English line only in how the language was forced.")

    if render:
        with plt.rc_context(RC):
            fig, axes = plt.subplots(1, 3, figsize=(rec.width_in, rec.height_in), layout="constrained")
            for ax, (col, fam), tag in zip(axes, families, "abc"):
                for key, (lang, lab, ls, mk) in zip(keys, series):
                    vs = [num(r[col]) for r in data[key]]
                    ax.plot(range(1, 11), vs, ls, marker=mk, color=LANG_COLOUR[lang],
                            markerfacecolor="white" if ls == "--" else LANG_COLOUR[lang],
                            markeredgewidth=0.9, label=lab)
                ax.set_xticks(range(1, 11))
                ax.set_xlim(0.5, 10.5)
                ax.set_ylim(bottom=0)
                ax.set_xlabel(f"({tag})  decile of the trace")
                ax.set_ylabel(f"{fam} markers per 1,000 characters")
                ax.grid(axis="both")
            handles = [Line2D([], [], color=LANG_COLOUR[l], linestyle=ls, marker=mk,
                              markerfacecolor="white" if ls == "--" else LANG_COLOUR[l],
                              markeredgewidth=0.9, label=lab)
                       for (l, lab, ls, mk) in series]
            fig.legend(handles=handles, loc="outside upper center", ncol=4,
                       handlelength=2.0, columnspacing=1.3, borderaxespad=0.2)
            save(fig, outdir, rec)
    return rec



F6_MEASURES = [
    ("mentions", "mean", "mentions the hint in the think block"),
    ("chen_verbal", "mean", "mentions and relies on it"),
    ("first_mention_frac", "median", "first mention, fraction of the trace"),
    ("mention_before_commit_regex", "mean", "named before the first settlement"),
    ("mention_post", "mean", "named in the final answer"),
]


def fig6(outdir: Path, render: bool) -> Rec:
    rec = Rec("F6", "disclosure_summary",
              "The paired cross-language disclosure contrasts of section 5.2: English minus Turkish and English "
              "minus Chinese for five measures, with 95% bootstrap intervals and the zero line drawn.",
              3.4, 3.4, "one column (3.3 in)")

    pc = read_csv("stageA/disclosure/paired_contrasts.csv")
    rec.src("stageA/disclosure/paired_contrasts.csv — columns `paired_diff`, `paired_ci_lo`, `paired_ci_hi`, "
            "`n_items_paired`, `rate_en`, `rate_other`, rows `q_lang` = en, `prefix` = zhao_long, `instruct` = none, "
            "`cue` = metadata, `cue_lang` = en, `metric` in {mentions, chen_verbal, first_mention_frac, "
            "mention_before_commit_regex, mention_post}, `t_lang_vs_en` in {tr, zh}. The `estimator` column is mean "
            "for the four rates and median for the position.")

    pts = []
    for metric, estimator, label in F6_MEASURES:
        for other in ("tr", "zh"):
            r = one(pc, "paired_contrasts.csv", q_lang="en", prefix="zhao_long", instruct="none", cue="metadata",
                    cue_lang="en", metric=metric, estimator=estimator, t_lang_vs_en=other)
            d, lo, hi = num(r["paired_diff"]), num(r["paired_ci_lo"]), num(r["paired_ci_hi"])
            n = int(r["n_items_paired"])
            pts.append((label, other, d, lo, hi, n))
            rec.val("all", f"{label} — English minus {LANG_NAME[other]}",
                    f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}], n = {n} questions "
                    f"(cell rates {num(r['rate_en']):.3f} and {num(r['rate_other']):.3f})",
                    f"paired_contrasts.csv paired_diff/paired_ci_lo/paired_ci_hi, metric={metric} "
                    f"estimator={estimator} t_lang_vs_en={other}")

    rec.note("Four of the five rows are differences of rates; the position row is a difference of medians in units of "
             "the fraction of the trace. They share an axis because they are all paired English-minus-other "
             "differences of comparable size, which is how section 5.2's table reports them, and the axis label "
             "says so.")
    rec.note("Marker colour is the non-English language of the contrast. A row whose interval crosses the dashed zero "
             "line is a null under the registered rule.")

    if render:
        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(rec.width_in, rec.height_in), layout="constrained")
            labels = [m[2] for m in F6_MEASURES]
            rows = [(label, other, d, lo, hi, 0.17 if other == "tr" else -0.17)
                    for label, other, d, lo, hi, n in pts]
            dot_rows(ax, rows, xline=0.0, row_labels=labels, label_fs=7.0)
            ax.set_xlim(-0.18, 0.19)
            ax.set_xlabel("English minus the other language, paired by question\n"
                          "(rates; fraction of the trace for the position row)", fontsize=7.0)
            handles = [Line2D([], [], color=LANG_COLOUR[l], marker="o", linestyle="",
                              label=f"English minus {LANG_NAME[l]}") for l in ("tr", "zh")]
            fig.legend(handles=handles, loc="outside upper center", ncol=2, borderaxespad=0.2)
            save(fig, outdir, rec)
    return rec



HEADER = """# Figures — what each one shows, and the exact source of every plotted number

Generated by `scripts/figures.py` (`uv run python scripts/figures.py`). Every value below was read out of a
git-tracked CSV under `stageA/` or `stageB/` at the moment the figure was drawn; nothing comes from `runs/`,
from a judged trace, or from prose, and no figure needed anything from `probes/`. The one hard-coded series is
F1's first panel, which is Onyame et al.'s published Table 4 and is marked as such both here and in the script.

Each figure is written twice, as `<name>.png` at 300 dpi and as `<name>.svg`. Re-running the script overwrites
them in place; `--only F3` re-renders one figure and still rewrites this file whole.

Conventions used identically in every figure: English is blue (`{en}`), Turkish burnt orange (`{tr}`) and
Chinese green (`{zh}`). Hatching means "not the main condition" and is explained in the figure that uses it. Error bars are 95% intervals and are drawn only where
the source CSV carries one — a figure with no error bars is one whose CSV has no interval column, never a figure
whose interval was dropped. No figure carries a title; the captions belong in the paper.
"""


def write_index(outdir: Path, recs: list[Rec]) -> Path:
    out = [HEADER.format(**{k: v for k, v in LANG_COLOUR.items()})]
    out.append("\n| figure | file | intended print width | what it shows |")
    out.append("|---|---|---|---|")
    for r in recs:
        out.append(f"| {r.key} | `{r.stem}.png` / `.svg` | {r.intended} | {r.what} |")
    out.append("")
    for r in recs:
        out.append(f"\n---\n\n## {r.key} — `{r.stem}`\n")
        out.append(f"**What it shows.** {r.what}\n")
        out.append(f"**Files.** `paper/figures/{r.stem}.png` (300 dpi) and `paper/figures/{r.stem}.svg`, "
                   f"drawn at {r.width_in} x {r.height_in} inches, intended for {r.intended}.\n")
        out.append("**Sources.**\n")
        for s in r.sources:
            out.append(f"- {s}")
        out.append("\n**Plotted numbers.**\n")
        out.append("| panel | what | value as plotted | where it comes from |")
        out.append("|---|---|---|---|")
        for v in r.values:
            out.append(f"| {v.panel} | {v.label} | {v.shown} | {v.source} |")
        if r.notes:
            out.append("\n**Reading it.**\n")
            for n in r.notes:
                out.append(f"- {n}")
        out.append("")
    path = outdir / "FIGURES.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path



BUILDERS = {"F1": fig1, "F2": fig2, "F3": fig3, "F4": fig4, "F5": fig5, "F6": fig6, "F7": fig7, "F8": fig8, "F9": fig9}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the paper's figures from the committed stage CSVs.")
    ap.add_argument("--only", action="append", metavar="Fn",
                    help="render only this figure (repeatable): " + ", ".join(BUILDERS))
    ap.add_argument("--outdir", default="paper/figures", help="where the PNG, SVG and FIGURES.md go")
    args = ap.parse_args(argv)

    wanted = set(k.upper() for k in (args.only or BUILDERS))
    unknown = wanted - set(BUILDERS)
    if unknown:
        ap.error(f"unknown figure(s) {sorted(unknown)}; known: {', '.join(BUILDERS)}")

    outdir = (ROOT / args.outdir) if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    recs = []
    for key, builder in BUILDERS.items():
        render = key in wanted
        rec = builder(outdir, render)
        recs.append(rec)
        print(f"{key} {rec.stem:28s} {'drawn' if render else 'data read, not re-rendered'}"
              f"  ({len(rec.values)} plotted values from {len(rec.sources)} source(s))")
    idx = write_index(outdir, recs)
    print(f"index {idx.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
