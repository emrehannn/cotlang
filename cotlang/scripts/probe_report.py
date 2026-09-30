#!/usr/bin/env python3
"""Compute the probe's registered decision, exactly as docs/PLAN.md §6 (2026-09-13 addendum) defines it.

    uv run --locked python scripts/probe_report.py runs/probe_pp15 runs/probe_pp0

The decision is made from numbers, on the clock, with a rented GPU running -- so the arithmetic is written here
BEFORE the data exist, and the estimators are the registered ones. Nothing in this script looks at hint disclosure;
it reads answers, language labels and lengths only.

Registered rules, in order:

  PRIMARY   mean Turkish prose compliance (sentence-weighted) at presence_penalty 0 minus the same at 1.5.
            Adopt 0 if the difference is positive and its 95% interval excludes zero.
  GUARD     adopt 0 only if median completion tokens rise by less than 25% and nothing truncates in either arm.
  CHINESE   the Chinese cells run in Stage A only if Chinese clears the same bars as Turkish:
            mean prose compliance >= 0.85 and share of traces at or above 0.8 >= 0.80.
  LENGTH    Chinese trace length against English, paired by question -- the claim that the EN/ZH contrast is
            length-matched rests on this.
  DESCRIBE  within-trace growth of the English share, first quarter against last quarter, pooled across traces and
            weighted by sentence count. Descriptive only: at 40 traces per arm it cannot separate "flat" from
            "still growing".
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cotlang.gate import COMPLIANT_MIN, PILOT_TR_COMPLIANCE_MIN, PILOT_TR_COMPLIANT_SHARE_MIN, prose_compliance_sentences

CI = 1.96


def load(workdir: str) -> list[dict]:
    p = Path(workdir) / "extracted.jsonl"
    if not p.exists():
        sys.exit(f"no extracted traces at {p} — run --stage extract first")
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def mean_ci(xs: list[float]) -> tuple[float, float]:
    """Mean and the half-width of its 95% interval."""
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    return st.mean(xs), CI * st.stdev(xs) / math.sqrt(len(xs))


DECIDING_CUES = ("none", "metadata")


def compliance(recs: list[dict], lang: str) -> list[float]:
    return [prose_compliance_sentences(r) or 0.0 for r in recs if r["cell"]["t_lang"] == lang and r["cell"]["cue"] in DECIDING_CUES]


def tokens(recs: list[dict], lang: str) -> list[int]:
    return [r.get("completion_tokens") or 0 for r in recs if r["cell"]["t_lang"] == lang and r["cell"]["cue"] in DECIDING_CUES]


def quarter_shares(recs: list[dict], lang: str) -> tuple[float, float]:
    """English share of sentences in the first and last quarter, pooled over traces, sentence-weighted."""
    f = l = fn = ln = 0
    for r in recs:
        if r["cell"]["t_lang"] != lang or r["cell"]["cue"] not in DECIDING_CUES:
            continue
        langs = r.get("sentence_langs") or []
        if len(langs) < 8:
            continue
        q = len(langs) // 4
        f += sum(1 for x in langs[:q] if x == "en"); fn += q
        l += sum(1 for x in langs[-q:] if x == "en"); ln += q
    return (f / fn if fn else float("nan")), (l / ln if ln else float("nan"))


def wall_seconds(workdir: str) -> float | None:
    """Wall-clock seconds the launcher spent inside `generate` for this workdir, summed over attempts, from the
    timestamps scripts/stage.sh writes to <runs>/logs/stage-*.log. None if the log has no completed attempt."""
    import re
    logdir = Path(workdir).parent / "logs"
    logs = sorted(logdir.glob("stage-*.log")) if logdir.exists() else []
    if not logs:
        return None
    ts = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] (.*)$")
    cur, t_start, total = None, None, 0.0
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            m = ts.match(line)
            if not m:
                continue
            t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
            msg = m[4]
            if msg.startswith("workdir "):
                cur = msg.split(None, 1)[1].strip()
            elif msg.startswith("attempt ") and "running generate" in msg and cur and Path(cur).resolve() == Path(workdir).resolve():
                t_start = t
            elif msg.startswith("generate exited") and t_start is not None and cur and Path(cur).resolve() == Path(workdir).resolve():
                total += (t - t_start) % 86400
                t_start = None
    return total or None


def main(a_dir: str, b_dir: str) -> int:
    A, B = load(a_dir), load(b_dir)
    print(f"arm A {a_dir}: {len(A)} traces      arm B {b_dir}: {len(B)} traces\n")

    print("PRIMARY — Turkish prose compliance (sentence-weighted), the registered bar is 0.85")
    ca, cb = compliance(A, "tr"), compliance(B, "tr")
    ma, ha = mean_ci(ca)
    mb, hb = mean_ci(cb)
    diff = mb - ma
    hd = math.sqrt(ha ** 2 + hb ** 2) if not (math.isnan(ha) or math.isnan(hb)) else float("nan")
    print(f"   penalty 1.5: mean {ma:.3f} +-{ha:.3f}  (n={len(ca)}, share>=0.8: {sum(x >= COMPLIANT_MIN for x in ca)/len(ca):.2f})")
    print(f"   penalty 0.0: mean {mb:.3f} +-{hb:.3f}  (n={len(cb)}, share>=0.8: {sum(x >= COMPLIANT_MIN for x in cb)/len(cb):.2f})")
    print(f"   difference : {diff:+.3f}  95% CI [{diff-hd:+.3f}, {diff+hd:+.3f}]")
    primary = diff > 0 and (diff - hd) > 0
    print(f"   -> primary rule {'MET' if primary else 'NOT met'}: penalty 0 {'is' if primary else 'is not'} better beyond noise")

    print("\nGUARD — length and truncation (adopt 0 only if median tokens rise <25% and nothing truncates)")
    guard = True
    for lang in ("en", "tr", "zh"):
        ta, tb = tokens(A, lang), tokens(B, lang)
        if not ta or not tb:
            continue
        mra, mrb = st.median(ta), st.median(tb)
        rise = (mrb - mra) / mra if mra else float("nan")
        tra = sum(1 for r in A if r["cell"]["t_lang"] == lang and r.get("truncated"))
        trb = sum(1 for r in B if r["cell"]["t_lang"] == lang and r.get("truncated"))
        ok = rise < 0.25 and tra == 0 and trb == 0
        guard &= ok
        print(f"   {lang}: median {mra:>7,.0f} -> {mrb:>7,.0f} ({rise:+.1%}), truncated {tra}/{trb}  {'ok' if ok else 'FAILS'}")
    print(f"   -> guard {'MET' if guard else 'NOT met'}")
    print(f"\n=> ADOPT presence_penalty 0 for Stage A: {'YES' if (primary and guard) else 'NO, keep 1.5'}")

    adopted = B if (primary and guard) else A
    label = "penalty 0" if (primary and guard) else "penalty 1.5"
    print(f"\nCHINESE — go/no-go, measured in the adopted arm ({label}); same bars as Turkish")
    cz = compliance(adopted, "zh")
    if not cz:
        print("   no Chinese traces found")
    else:
        mz, hz = mean_ci(cz)
        share = sum(x >= COMPLIANT_MIN for x in cz) / len(cz)
        ok = mz >= PILOT_TR_COMPLIANCE_MIN and share >= PILOT_TR_COMPLIANT_SHARE_MIN
        print(f"   mean {mz:.3f} +-{hz:.3f} (bar {PILOT_TR_COMPLIANCE_MIN}), share>=0.8 {share:.2f} (bar {PILOT_TR_COMPLIANT_SHARE_MIN}), n={len(cz)}")
        print(f"   -> Chinese cells {'RUN' if ok else 'are DROPPED from Stage A'}")

    print(f"\nLENGTH — is Chinese near English? (paired by question, in the adopted arm)")
    by_item: dict[str, dict[str, int]] = {}
    for r in adopted:
        if r["cell"]["cue"] != "none":
            continue
        by_item.setdefault(r["item_id"], {})[r["cell"]["t_lang"]] = r.get("completion_tokens") or 0
    for other in ("zh", "tr"):
        pairs = [(v["en"], v[other]) for v in by_item.values() if "en" in v and other in v]
        if len(pairs) < 2:
            continue
        d = [e - o for e, o in pairs]
        m, h = mean_ci(d)
        ratio = st.median([e for e, _ in pairs]) / st.median([o for _, o in pairs])
        print(f"   en - {other}: mean {m:+,.0f} tokens 95% CI [{m-h:+,.0f}, {m+h:+,.0f}] over {len(pairs)} questions; "
              f"median ratio en/{other} = {ratio:.2f}x")

    print(f"\nTHROUGHPUT — how long Stage A actually takes on THIS instance")
    par = 64
    for arm, name in ((A, a_dir), (B, b_dir)):
        gen = Path(name) / "generations.jsonl"
        recs = [json.loads(l) for l in gen.open(encoding="utf-8") if l.strip()] if gen.exists() else []
        el = [r.get("elapsed_s") or 0 for r in recs]
        tk = [r.get("completion_tokens") or 0 for r in recs]
        if not el or not sum(el):
            continue
        est = sum(tk) / (sum(el) / par) * 0.9
        mean_tok = sum(tk) / len(tk)
        wall = wall_seconds(name)
        rate = sum(tk) / wall if wall else est
        if not wall:
            print("   WARNING: no launcher log found for this workdir; the rate below is the latency ESTIMATE, which assumes\n"
                  f"            {par} busy streams and read 2.6x too high on run 4. Do not size Stage A from it.")
        src = f"wall clock {wall/60:.1f} min in generate" if wall else "latency estimate (no launcher log found)"
        print(f"   {name}: {len(recs)} traces, mean {mean_tok:,.0f} tokens; {sum(tk)/1e6:.2f}M tokens over {src}")
        print(f"      -> {rate:,.0f} tok/s measured{f' (latency estimate {est:,.0f})' if wall else ''} at {par} client streams")
        for n, label in ((3600, "k=3  (2,700 core + 900 extras)"), (1800, "k=1  (900 core + 900 extras)")):
            print(f"      Stage A {label}: {n * mean_tok / rate / 3600:5.1f} h  (the long run saturates the tail, so"
                  f" expect this or better)")
    print("   (run 4 measured 969 tok/s at 24 streams. Decide k from the k=3 hours above: if it does not fit the")
    print("    session, set sampling.k: 1 in configs/stageA_core.yaml before starting Stage A.)")

    print(f"\nDESCRIPTIVE — within-trace growth of the English share (pooled, sentence-weighted)")
    for name, arm in ((f"penalty 1.5", A), (f"penalty 0.0", B)):
        q1, q4 = quarter_shares(arm, "tr")
        print(f"   Turkish, {name}: first quarter {q1:.3f} -> last quarter {q4:.3f}  (growth {q4-q1:+.3f})")
    print("   (run 4 measured 0.145 -> 0.593. Descriptive only: 40 traces per arm cannot separate flat from growing.)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
