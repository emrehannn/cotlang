#!/usr/bin/env python3
"""English-minus-Turkish metadata uptake, paired by question, under each forcing method of Stage A (Table 4), and every
pairwise uptake contrast under the naming prefix (Section 4.1).

    uv run python scripts/uptake_by_forcing.py        # writes stageA/uptake_by_forcing.csv

A question's uptake in a thinking language is its metadata-cell hint-following minus its no-hint rate of choosing the
hinted letter, both over answered traces (the mean of its samples where k = 3). The EN - TR gap is paired within the
question over the questions answered in all four cells of a forcing method. The interaction row is the neutral-prefix
gap minus the naming-prefix gap, paired over the questions answered in all eight cells. The hint-following row for the
neutral prefix counts the questions that followed in English only and in Turkish only (k = 1) and gives the exact
two-sided binomial p of that split. 4,000 percentile-bootstrap resamples of questions, seed 20260906, as in
cotlang/phase.py. Reads only item_id and hint_letter from the item file. Replaces scripts/neutral_gap.py, whose two
rows are reproduced here."""
import csv, json, sys
from math import comb
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "stageA" / "raw"
N_BOOT, SEED = 4000, 20260906
METHODS = {
    "naming prefix (k = 3)": ("core", "qen_t{t}_pzhao_long_c{c}_clen"),
    "naming prefix + instruction (k = 1)": ("instruct", "qen_t{t}_pzhao_long_c{c}_clen_iexplicit"),
    "neutral prefix (k = 1)": ("extras", "qen_t{t}_pneutral_c{c}_clen"),
}

def main():
    hint = {}
    with open(ROOT / "data" / "items_gpqa.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hint[r["item_id"]] = r["hint_letter"]
    records = {s: list(csv.DictReader(open(RAW / f"gate_pilot_records.stageA_{s}.csv", encoding="utf-8")))
               for s in ("core", "instruct", "extras")}

    def follow(src, cell):
        acc = {}
        for x in records[src]:
            if x["cell_key"] == cell and x["answer"] in ("A", "B", "C", "D"):
                acc.setdefault(x["item_id"], []).append(float(x["answer"] == hint[x["item_id"]]))
        return {q: float(np.mean(v)) for q, v in acc.items()}

    rng = np.random.default_rng(SEED)
    def boot(d):
        b = np.mean(d[rng.integers(0, len(d), size=(N_BOOT, len(d)))], axis=1)
        return np.percentile(b, [2.5, 97.5])

    F = {m: {(t, c): follow(src, key.format(t=t, c=c)) for t in ("en", "tr") for c in ("none", "metadata")}
         for m, (src, key) in METHODS.items()}
    def gap(m, q):
        f = F[m]
        return (f[("en", "metadata")][q] - f[("en", "none")][q]) - (f[("tr", "metadata")][q] - f[("tr", "none")][q])

    out = []
    def row(quantity, qs, d, **extra):
        lo, hi = boot(d)
        out.append({"quantity": quantity, "n_questions": len(qs), "estimate": round(float(d.mean()), 4),
                    "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                    "en_only": extra.get("en_only", ""), "tr_only": extra.get("tr_only", ""),
                    "exact_p_two_sided": extra.get("p", "")})

    common = {}
    for m in METHODS:
        qs = sorted(set.intersection(*(set(v) for v in F[m].values())))
        common[m] = set(qs)
        row(f"uptake gap EN-TR, paired: {m}", qs, np.array([gap(m, q) for q in qs]))
    nam, neu = "naming prefix (k = 3)", "neutral prefix (k = 1)"
    qs = sorted(common[nam] & common[neu])
    row("interaction: neutral gap minus naming gap, paired", qs, np.array([gap(neu, q) - gap(nam, q) for q in qs]))

    for cue in ("metadata", "unethical"):
        G = {t: (follow("core", f"qen_t{t}_pzhao_long_c{cue}_clen"), follow("core", f"qen_t{t}_pzhao_long_cnone_clen"))
             for t in ("en", "tr", "zh")}
        for a_, b_ in (("en", "tr"), ("en", "zh"), ("tr", "zh")):
            if (cue, a_, b_) == ("metadata", "en", "tr"):
                continue
            qs = sorted(set(G[a_][0]) & set(G[a_][1]) & set(G[b_][0]) & set(G[b_][1]))
            d = np.array([(G[a_][0][q] - G[a_][1][q]) - (G[b_][0][q] - G[b_][1][q]) for q in qs])
            row(f"uptake gap {a_.upper()}-{b_.upper()}, paired: naming prefix (k = 3), {cue}", qs, d)

    f = F[neu]
    q2 = sorted(set(f[("en", "metadata")]) & set(f[("tr", "metadata")]))
    d2 = np.array([f[("en", "metadata")][q] - f[("tr", "metadata")][q] for q in q2])
    a, b = int((d2 > 0).sum()), int((d2 < 0).sum())
    n, k = a + b, min(a, b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    row(f"hint-following gap EN-TR, paired: {neu}", q2, d2, en_only=a, tr_only=b, p=round(p, 4))

    dst = ROOT / "stageA" / "uptake_by_forcing.csv"
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    for r in out:
        print(r)
    print(f"wrote {dst.relative_to(ROOT)}", file=sys.stderr)

if __name__ == "__main__":
    main()
