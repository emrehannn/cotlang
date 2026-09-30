#!/usr/bin/env python3
"""Judge test-retest reliability (docs/ROADMAP.md step 5a; PLAN twentieth and twenty-fourth addenda): buy a second and
third reading of a seeded, stratified subsample of the Stage A labels with the cache bypassed, and report per-field
agreement.

    uv run --locked python scripts/judge_retest.py runs/stageA_gpqa --config configs/stageA_core_judge.yaml \
        --n 60 --readings 2 --out runs/stageA_gpqa_retest            # selects, writes selection.json, projects; spends nothing
    ... --go                                                          # spends; emre's call, never the default
    ... --analyze-only                                                # re-derive the tables from readings.jsonl

Nothing in cotlang/ is modified: the prompt is rebuilt with the same functions judge_all uses (`_user_prompt`,
`control_framing`, `hint_text`, `build_user_message`), the same Judge class and resolver, so a reading here is the
same call the main pass made, minus the sqlite cache (verified 2026-09-15: all 60 rebuilt prompts hit the main pass's
cache key). Raw replies land in <out>/judge_raw/ so scripts/deepseek_cost.py prices the run like any workdir. The
original label from judged.jsonl is reading 0; the new ones are 1..readings.

Selection: among valid labels whose judge_context matches the current judge, `n` traces balanced over the six strata
(trace language x hint / no-hint), sampled with `--seed` from the labels PRESENT AT SELECTION TIME, written to
<out>/selection.json before any call. Once that file exists it is the selection: a later run reuses it and refuses a
different --seed or --n (delete the file to re-select). Select only after the last judge twin has finished, so the
pool holds every cell.

Report (RETEST.md): per field, the share of traces on which every reading agrees (`all_agree`, over traces) and the
agreement over reading PAIRS (`pair_agree`); for the three index fields, over non-null pairs, the median absolute
difference AS A FRACTION OF THE THINK BLOCK'S SENTENCES (the roadmap's unit, not the index) and the share within 2
sentences; and the share of pairs where one reading found the event and the other did not. Per stratum and pooled.

Adversarial pass 2026-09-15 15:45 applied: selection persisted and reused; futures cancelled on any exception and
after 8 consecutive errors (as judge_all); a cut last line closed before appending; reading 0 written before any call;
a field missing from every record is an error, not perfect agreement; pair statistics pooled over pairs; an unknown
balance stops the run; the item file's registered hash and the sentence counts are checked against reading 0.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cotlang import items as items_mod
from cotlang.generate import close_partial_last_line
from cotlang.judge import (JUDGE_CONTEXT, MAX_CONSECUTIVE_ERRORS, SCHEMA_HASH, Judge, _user_prompt,
                           control_framing, resolve_label)
from cotlang.prompts import build_user_message, hint_text

CATEGORICAL = ("mentions_hint", "relies_on_hint", "restated_as_own", "acknowledges_but_denies", "mention_in_final_answer",
               "mention_type", "mention_language")
INDEX = ("first_mention_sentence", "first_reliance_sentence", "first_commitment_sentence")


def balance() -> tuple[float | None, str | None]:
    import urllib.request
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None, None
    try:
        req = urllib.request.Request("https://api.deepseek.com/user/balance", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            info = json.load(r)["balance_infos"][0]
            return float(info["total_balance"]), info.get("currency")
    except Exception:
        return None, None


def is_peak() -> bool:
    """DeepSeek peak: 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday (CLAUDE.md §5)."""
    t = time.gmtime()
    return t.tm_wday < 5 and (1 <= t.tm_hour < 4 or 6 <= t.tm_hour < 10)


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def stratum_of(rec: dict) -> str:
    cell = rec["gen_id"].split("|")[1]
    t_lang = next(tok[1:] for tok in cell.split("_") if tok.startswith("t") and tok[1:] in ("en", "tr", "zh"))
    return f"{t_lang} {'no hint' if rec.get('is_control') else 'hint'}"


def select(work: Path, ctx: str, n: int, seed: int) -> list[dict]:
    """The last record per gen_id from judged.jsonl under the current judge context (an error last record drops the
    gen_id, as labelled_ids does), then n // 6 per stratum, seeded."""
    last: dict[str, dict] = {}
    for r in read_jsonl(work / "judged.jsonl"):
        if "error" in r:
            last.pop(r["gen_id"], None)
        elif r.get("judge_context") == ctx:
            last[r["gen_id"]] = r
    if not last:
        sys.exit(f"[retest] no valid label under judge context {ctx} in {work/'judged.jsonl'}")
    strata: dict[str, list[dict]] = defaultdict(list)
    for r in last.values():
        strata[stratum_of(r)].append(r)
    rng = np.random.default_rng(seed)
    per = n // len(strata)
    if per * len(strata) != n:
        print(f"[retest] WARNING: --n {n} is not a multiple of {len(strata)} strata; selecting {per * len(strata)}", file=sys.stderr)
    chosen = []
    for key in sorted(strata):
        pool = sorted(strata[key], key=lambda r: r["gen_id"])
        if len(pool) < per:
            print(f"[retest] WARNING: stratum '{key}' has only {len(pool)} labels, wanted {per}", file=sys.stderr)
        take = rng.choice(len(pool), size=min(per, len(pool)), replace=False)
        chosen += [dict(pool[i], stratum=key) for i in sorted(take)]
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir"); ap.add_argument("--config", default="configs/stageA_core_judge.yaml")
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--readings", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260915); ap.add_argument("--out", default=None)
    ap.add_argument("--per-label", type=float, default=0.0084, help="projected USD per label off-peak, from the main pass")
    ap.add_argument("--reserve", type=float, default=1.0, help="USD that must remain after the projection")
    ap.add_argument("--go", action="store_true", help="actually spend; without it the script only selects and projects")
    ap.add_argument("--allow-unknown-balance", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    work = Path(a.workdir)
    if (work / ".blind").exists():
        sys.exit(f"[blind] {work} is blind; the judge has not run here")
    out = Path(a.out) if a.out else Path(str(work) + "_retest")
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(a.config).read_text())
    jcfg = cfg["judge"]
    readings_path = out / "readings.jsonl"
    sel_path = out / "selection.json"

    if not a.analyze_only:
        items_path = Path(cfg["paths"]["items"]); want_sha = cfg["items"].get("sha256")
        if want_sha and hashlib.sha256(items_path.read_bytes()).hexdigest() != want_sha:
            sys.exit(f"[retest] {items_path} does not match the registered sha256 in {a.config}")
        judge = Judge(jcfg, out / "retest_cache.sqlite")
        ctx = f"{JUDGE_CONTEXT[judge.backend]}|{judge.model}|{SCHEMA_HASH}"
        if sel_path.exists():
            sel = json.loads(sel_path.read_text())
            if sel["seed"] != a.seed or sel["n"] != a.n or sel["context"] != ctx:
                sys.exit(f"[retest] {sel_path} was made with seed {sel['seed']}, n {sel['n']}, context {sel['context']}; "
                         "delete it to re-select")
            last = {r["gen_id"]: r for r in read_jsonl(work / "judged.jsonl") if "error" not in r and r.get("judge_context") == ctx}
            chosen = [dict(last[g], stratum=st) for g, st in zip(sel["gen_ids"], sel["strata"])]
            print(f"[retest] reusing the {len(chosen)} traces in {sel_path}", file=sys.stderr)
        else:
            chosen = select(work, ctx, a.n, a.seed)
            sel_path.write_text(json.dumps({"seed": a.seed, "n": a.n, "readings": a.readings, "context": ctx,
                                            "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                            "pool_note": "labels present in judged.jsonl at selection time",
                                            "gen_ids": [c["gen_id"] for c in chosen], "strata": [c["stratum"] for c in chosen]}, indent=2))
        counts = pd.Series([c["stratum"] for c in chosen]).value_counts().sort_index()
        print(f"[retest] {len(chosen)} traces under context {ctx}:\n{counts.to_string()}", file=sys.stderr)
        n_calls = len(chosen) * a.readings
        rate = a.per_label * (2 if is_peak() else 1)
        proj = n_calls * rate
        bal, cur = balance()
        print(f"[retest] {n_calls} uncached calls projected at ${proj:.2f} ({'PEAK' if is_peak() else 'off-peak'} rate; input mostly "
              f"prefix-cache hits, so likely less; an unparseable reply is paid twice); balance "
              f"{'unknown' if bal is None else f'{bal:.2f} {cur}'}; reserve ${a.reserve:.2f}", file=sys.stderr)
        if not a.go:
            print("[retest] dry run: pass --go to spend (emre's call)", file=sys.stderr)
            return 0
        if bal is None:
            if not a.allow_unknown_balance:
                sys.exit("[retest] STOP: balance unknown (pass --allow-unknown-balance to override)")
        else:
            if cur not in (None, "USD"):
                sys.exit(f"[retest] STOP: balance currency is {cur}, projection is in USD")
            if bal < proj + a.reserve:
                sys.exit(f"[retest] STOP: balance ${bal:.2f} below projection ${proj:.2f} + reserve ${a.reserve:.2f}")
        items = {i["item_id"]: i for i in items_mod.load_items(items_path)}
        want = {c["gen_id"] for c in chosen}
        ext = {}
        for line in (work / "extracted.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r["gen_id"] in want:
                ext[r["gen_id"]] = r
                if len(ext) == len(want):
                    break
        missing = want - set(ext)
        if missing:
            sys.exit(f"[retest] {len(missing)} selected gen_ids have no extracted record")
        for c in chosen:
            if len(ext[c["gen_id"]]["sentences"]) != c.get("think_sentences_total"):
                sys.exit(f"[retest] {c['gen_id']}: extracted.jsonl has {len(ext[c['gen_id']]['sentences'])} sentences, "
                         f"the label saw {c.get('think_sentences_total')}; extraction changed since the pass")
        close_partial_last_line(readings_path) if readings_path.exists() else None
        existing = [r for r in read_jsonl(readings_path) if "error" not in r]
        have0 = {r["gen_id"] for r in existing if r.get("reading") == 0}
        with readings_path.open("a", encoding="utf-8") as f:
            for c in chosen:
                if c["gen_id"] not in have0:
                    f.write(json.dumps({**{k: v for k, v in c.items() if not k.startswith("_")}, "reading": 0}, ensure_ascii=False) + "\n")
        done = {(r["gen_id"], r["reading"]) for r in existing if r.get("reading", 0) > 0}
        judge.raw_dir = out / "judge_raw"; judge.raw_dir.mkdir(exist_ok=True)
        max_in = jcfg.get("max_input_tokens"); max_in = None if max_in is None else int(max_in)

        def work_one(c: dict, reading: int) -> dict:
            """The same call judge_all makes for this trace, cache bypassed. Mirrors judge_all.work on purpose:
            cotlang/judge.py is not edited while a judge pass may still import it. Never raises: any failure becomes
            an error record, which a rerun retries."""
            base = {"gen_id": c["gen_id"], "reading": reading, "stratum": c["stratum"]}
            try:
                e = ext[c["gen_id"]]; cell = e["cell"]; it = items[e["item_id"]]
                is_control = cell["cue"] == "none"
                cue = control_framing(e["gen_id"]) if is_control else cell["cue"]
                hint = hint_text(cue, cell["cue_lang"], it["hint_letter"], e["item_id"])
                question = build_user_message(it, cell["q_lang"], "none", cell["cue_lang"], cell.get("instruct", "none"), cell["t_lang"])
                budget = max_in
                prompt, shown, total = _user_prompt(question, cue, hint, it["hint_letter"], e["sentences"], e["post"], budget)
                for _ in range(3):
                    try:
                        lab = judge.label(prompt, use_cache=False)
                    except Exception as err:
                        lab = {"error": repr(err)}
                    msg = str(lab.get("error", "")).lower()
                    if "error" in lab and any(k in msg for k in ("context", "too long", "maximum context", "length limit")):
                        budget = int((budget or 120_000) * 0.45)
                        prompt, shown, total = _user_prompt(question, cue, hint, it["hint_letter"], e["sentences"], e["post"], budget)
                        continue
                    break
                for fld in INDEX:
                    v = lab.get(fld)
                    if isinstance(v, int) and v > shown:
                        lab[fld] = None
                        lab["index_out_of_range"] = sorted(set(lab.get("index_out_of_range", []) + [fld]))
                resolve_label(lab, e["sentences"][:shown])
                judge.save_raw(c["gen_id"], lab)
                for k in ("_raw", "_thinking", "_discarded"):
                    lab.pop(k, None)
                return {**base, "is_control": is_control, "control_framing": cue if is_control else None,
                        "think_sentences_shown": shown, "think_sentences_total": total, "input_truncated": shown < total, **lab}
            except Exception as err:
                return {**base, "error": repr(err)}

        jobs = [(c, k) for c in chosen for k in range(1, a.readings + 1) if (c["gen_id"], k) not in done]
        print(f"[retest] {len(jobs)} calls to make ({len(done)} already on disk)", file=sys.stderr)
        n_err = consecutive = 0
        with ThreadPoolExecutor(max_workers=int(jcfg.get("max_concurrency", 4))) as ex, readings_path.open("a", encoding="utf-8") as f:
            futs = [ex.submit(work_one, c, k) for c, k in jobs]
            try:
                for fut in as_completed(futs):
                    rec = fut.result()
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                    if "error" in rec:
                        n_err += 1; consecutive += 1
                        if consecutive >= MAX_CONSECUTIVE_ERRORS:
                            raise RuntimeError(f"[retest] {MAX_CONSECUTIVE_ERRORS} consecutive errors; last: {rec['error']}. Rerun resumes.")
                    else:
                        consecutive = 0
            except BaseException:
                for other in futs:
                    other.cancel()
                raise
        print(f"[retest] done, {n_err} error records (rerun resumes them)", file=sys.stderr)

    if not readings_path.exists():
        sys.exit(f"[retest] {readings_path} does not exist yet: run with --go first")
    recs = [r for r in read_jsonl(readings_path) if "error" not in r]
    if not recs:
        sys.exit("[retest] no readings to analyse")
    df = pd.DataFrame(recs)
    for fld in CATEGORICAL + INDEX:
        if fld not in df:
            sys.exit(f"[retest] field {fld} is absent from every reading; the schema has changed, nothing to compare")
    ctxs = set(df.get("judge_context", pd.Series(dtype=str)).dropna())
    if len(ctxs) > 1:
        sys.exit(f"[retest] readings come from more than one judge context: {ctxs}")
    reading_sets = df.groupby("gen_id").reading.agg(lambda s: tuple(sorted(set(s))))
    no0 = [g for g, s in reading_sets.items() if 0 not in s]
    if no0:
        sys.exit(f"[retest] {len(no0)} traces have no reading 0 (the main-pass label); run without --analyze-only to append it")
    trace_rows, pair_rows = [], []

    def norm(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

    for gid, d in df.groupby("gen_id"):
        d = d.sort_values("reading")
        if d.reading.nunique() < 2:
            continue
        total = float(d.think_sentences_total.iloc[0] or np.nan)
        stratum = d.stratum.iloc[0]
        for fld in CATEGORICAL + INDEX:
            vals = [norm(v) for v in d[fld].tolist()]
            trace_rows.append({"gen_id": gid, "stratum": stratum, "field": fld, "n_readings": len(vals),
                               "all_agree": float(all(v == vals[0] for v in vals))})
            for (ra, x), (rb, y) in combinations(zip(d.reading.tolist(), vals), 2):
                row = {"gen_id": gid, "stratum": stratum, "field": fld, "pair": f"{ra}-{rb}", "agree": float(x == y)}
                if fld in INDEX:
                    row["both_null"] = float(x is None and y is None)
                    row["one_null"] = float((x is None) != (y is None))
                    if x is not None and y is not None:
                        row["abs_diff_frac"] = abs(x - y) / total if total else np.nan
                        row["within_2"] = float(abs(x - y) <= 2)
                pair_rows.append(row)
    per_trace, per_pair = pd.DataFrame(trace_rows), pd.DataFrame(pair_rows)
    per_trace.to_csv(out / "retest_per_trace.csv", index=False); per_pair.to_csv(out / "retest_per_pair.csv", index=False)

    def table(keys):
        t = per_trace.groupby(keys).agg(n_traces=("gen_id", "size"), all_agree=("all_agree", "mean"))
        p = per_pair.groupby(keys).agg(n_pairs=("gen_id", "size"), pair_agree=("agree", "mean"))
        cols = {c: (c, "mean") for c in ("both_null", "one_null", "within_2") if c in per_pair}
        if "abs_diff_frac" in per_pair:
            cols["abs_diff_frac_median"] = ("abs_diff_frac", "median")
            cols["n_nonnull_pairs"] = ("abs_diff_frac", "count")
        return t.join(p).join(per_pair.groupby(keys).agg(**cols)) if cols else t.join(p)

    pooled, strat = table(["field"]), table(["field", "stratum"])
    lines = [f"# Judge test-retest — {work}", "",
             f"traces: {per_trace.gen_id.nunique()}  reading sets: {dict(Counter(reading_sets))}  "
             f"judge context: {next(iter(ctxs)) if ctxs else '?'}", "",
             "`all_agree`: share of traces on which every reading gives the same value (null counts as a value). The other columns "
             "are over reading PAIRS (reading 0 is the main pass's label): `pair_agree` the share of agreeing pairs; for the index "
             "fields `both_null` / `one_null` the share of pairs where neither / exactly one reading found the event, "
             "`abs_diff_frac_median` the median over non-null pairs of the absolute difference as a fraction of the think block's "
             "sentences, `within_2` the share of non-null pairs within 2 sentences.", "",
             "## Pooled", "", pooled.round(3).to_markdown(), "",
             "## By stratum (trace language x hint)", "", strat.round(3).to_markdown(), ""]
    (out / "RETEST.md").write_text("\n".join(lines), encoding="utf-8")
    pooled.to_csv(out / "retest_pooled.csv"); strat.to_csv(out / "retest_by_stratum.csv")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
