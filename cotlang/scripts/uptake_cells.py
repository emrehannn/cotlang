#!/usr/bin/env python3
"""Uptake per cell with a 95% bootstrap interval over questions: the paper's Figure 1 (figures.py F8).

    uv run python scripts/uptake_cells.py        # writes stageA/uptake_cells.csv and stageB/uptake_cells.csv

A cell's uptake is its hint-following minus the no-hint rate of choosing the hinted letter, both as cell-level rates over
answered traces, exactly as in the paper's Table 3: the no-hint cell is the one with the same prompt language, thinking
language and forcing method, and for the sycophancy cells (k = 1) the three-sample naming-prefix no-hint cell. The
interval resamples questions (4,000 percentile-bootstrap resamples, seed 20260906, as in scripts/uptake_by_forcing.py)
and recomputes both rates on the resampled questions, so the point estimate is the full-sample Table 3 value. These are
per-cell intervals: differences between thinking languages are tested paired by question in uptake_by_forcing.csv and
stageB/cross_stage/cross_stage_uptake.csv. Reads the committed per-trace records and only item_id and hint_letter from
the item file."""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_BOOT, SEED = 4000, 20260906
RECORDS = ["stageA/raw/gate_pilot_records.stageA_core.csv", "stageA/raw/gate_pilot_records.stageA_instruct.csv",
           "stageA/raw/gate_pilot_records.stageA_extras.csv", "stageB/raw/gate_pilot_records.stageB_tr.csv"]
CELLS = [
    ("A", "English", "metadata", "naming prefix", 3, "qen_t{t}_pzhao_long_cmetadata_clen", "qen_t{t}_pzhao_long_cnone_clen"),
    ("A", "English", "metadata", "naming prefix + instruction", 1, "qen_t{t}_pzhao_long_cmetadata_clen_iexplicit",
     "qen_t{t}_pzhao_long_cnone_clen_iexplicit"),
    ("A", "English", "metadata", "neutral prefix", 1, "qen_t{t}_pneutral_cmetadata_clen", "qen_t{t}_pneutral_cnone_clen"),
    ("A", "English", "unethical", "naming prefix", 3, "qen_t{t}_pzhao_long_cunethical_clen", "qen_t{t}_pzhao_long_cnone_clen"),
    ("A", "English", "sycophancy", "naming prefix", 1, "qen_t{t}_pzhao_long_csycophancy_clen", "qen_t{t}_pzhao_long_cnone_clen"),
    ("B", "Turkish", "metadata", "naming prefix", 1, "qtr_t{t}_pzhao_long_cmetadata_cltr", "qtr_t{t}_pzhao_long_cnone_cltr"),
    ("B", "Turkish", "unethical", "naming prefix", 1, "qtr_t{t}_pzhao_long_cunethical_cltr", "qtr_t{t}_pzhao_long_cnone_cltr"),
]

def main():
    hint = {}
    with open(ROOT / "data" / "items_gpqa.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hint[r["item_id"]] = r["hint_letter"]
    records = []
    for rel in RECORDS:
        with open(ROOT / rel, encoding="utf-8") as f:
            records += list(csv.DictReader(f))

    def counts(cell):
        d = {}
        for x in records:
            if x["cell_key"] == cell and x["answer"] in ("A", "B", "C", "D"):
                a, k = d.get(x["item_id"], (0, 0))
                d[x["item_id"]] = (a + 1, k + (x["answer"] == hint[x["item_id"]]))
        if not d:
            raise SystemExit(f"uptake_cells.py: no answered records for {cell}")
        return d

    out = {"A": [], "B": []}
    for stage, prompt, cue, forcing, k, hkey, nkey in CELLS:
        for t in (("en", "tr", "zh") if stage == "A" else ("en", "tr")):
            H, N = counts(hkey.format(t=t)), counts(nkey.format(t=t))
            qs = sorted(set(H) | set(N))
            h = np.array([H.get(q, (0, 0)) for q in qs], float)
            n = np.array([N.get(q, (0, 0)) for q in qs], float)
            follow, base = h[:, 1].sum() / h[:, 0].sum(), n[:, 1].sum() / n[:, 0].sum()
            rng = np.random.default_rng(SEED)
            idx = rng.integers(0, len(qs), size=(N_BOOT, len(qs)))
            hs, ns = h[idx].sum(axis=1), n[idx].sum(axis=1)
            lo, hi = np.percentile(hs[:, 1] / hs[:, 0] - ns[:, 1] / ns[:, 0], [2.5, 97.5])
            out[stage].append({"prompt": prompt, "cue": cue, "forcing": forcing, "k": k, "t_lang": t,
                               "hint_cell": hkey.format(t=t), "nohint_cell": nkey.format(t=t),
                               "answered": int(h[:, 0].sum()), "hint_follow": round(float(follow), 4),
                               "nohint_rate": round(float(base), 4), "uptake": round(float(follow - base), 4),
                               "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4), "n_questions": len(qs)})
    for stage, rows in out.items():
        path = ROOT / f"stage{stage}" / "uptake_cells.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path.relative_to(ROOT)} ({len(rows)} cells)")
        for r in rows:
            print(f"  {r['prompt']:7s} {r['cue']:10s} {r['forcing']:28s} k={r['k']} {r['t_lang']}  "
                  f"follow {r['hint_follow']:.3f} - no-hint {r['nohint_rate']:.3f} = {r['uptake']:+.3f} "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")

if __name__ == "__main__":
    main()
