from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from . import analysis, annotate, extract, gate, generate, items as items_mod, judge, server
from .prompts import cell_key, hint_text, normalize_cell


def load_cfg(path: str) -> dict:
    """Read the YAML config, expand ~ in cache_dir, validate every cell."""
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["paths"]["cache_dir"] = str(Path(os.path.expanduser(cfg["paths"]["cache_dir"])))
    cfg["cells"] = [normalize_cell(c) for c in cfg["cells"]]
    return cfg


def dump_traces(ext_path: Path, gen_path: Path, items: list[dict], out_path: Path, n_per_cell: int) -> None:
    """Write the first n traces of every cell as readable markdown (from extracted.jsonl if it exists, else raw generations)."""
    recs = analysis.load_jsonl(ext_path) or analysis.load_jsonl(gen_path)
    by_item = {i["item_id"]: i for i in items}
    seen: dict[str, int] = {}
    lines = ["# Trace dump", ""]
    for r in recs:
        ck = r.get("cell_key", "?")
        if seen.get(ck, 0) >= n_per_cell:
            continue
        seen[ck] = seen.get(ck, 0) + 1
        it = by_item.get(r.get("item_id"), {})
        c = r.get("cell", {})
        lines += [f"## {ck}  |  item {r.get('item_id')}  |  sample {r.get('sample_idx')}", ""]
        if "error" in r:
            lines += [f"ERROR: {r['error']}", ""]
            continue
        if it:
            lines += [f"gold={it['answer']}  hinted={it['hint_letter']}  model answer={r.get('answer')}  "
                      f"compliance={r.get('compliance')}  tokens={r.get('completion_tokens')}  truncated={r.get('truncated')}", "",
                      f"QUESTION (en): {it['text']['en']['question']}", ""]
            if c.get("cue", "none") != "none":
                lines += [f"HINT: {hint_text(c['cue'], c['cue_lang'], it['hint_letter'], it['item_id'])}", ""]
        think = r.get("think") if "think" in r else r.get("text", "")
        lines += ["THINK:", "", think or "(empty)", ""]
        if r.get("post"):
            lines += ["FINAL:", "", r["post"], ""]
        lines += ["---", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[dump] {sum(seen.values())} traces from {len(seen)} cells -> {out_path}", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["all", "items", "generate", "extract", "judge", "analyze"])
    ap.add_argument("--no-langid", action="store_true", help="skip GlotLID (faster extract, no compliance numbers)")
    ap.add_argument("--rebuild-items", action="store_true", help="rebuild the item file even if it exists")
    ap.add_argument("--export-items-review", metavar="LANG", help="write a side-by-side EN/LANG item sheet and exit")
    ap.add_argument("--dump-traces", type=int, metavar="N_PER_CELL", help="write a readable dump of N traces per cell and exit")
    ap.add_argument("--annotate-export", type=int, metavar="N_PER_LANG")
    ap.add_argument("--annotate-langs", default="tr,zh")
    ap.add_argument("--kappa", metavar="FILLED_CSV")
    ap.add_argument("--gate", choices=["smoke", "pilot"], help="pilot checks from extracted.jsonl only (cotlang/gate.py); exit")
    ap.add_argument("--unblind", action="store_true", help="remove the workdir's .blind marker (Stage A only; see header)")
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config)
    safe = (args.gate or args.export_items_review or args.stage in ("items", "generate", "extract")) and \
        not (args.dump_traces or args.annotate_export or args.kappa)
    marker = Path(cfg["paths"]["workdir"]) / ".blind"
    if cfg.get("blind"):
        if not safe or args.no_langid or args.unblind:
            raise SystemExit(f"[blind] {args.config} is a blind config (docs/PLAN.md, pilot design): only --stage items, "
                             "--stage generate, --stage extract, --gate and --export-items-review are allowed; no "
                             "--dump-traces, --annotate-export, --kappa, --no-langid or --unblind")
    elif marker.exists() and not safe and not args.unblind:
        raise SystemExit(f"[blind] {marker} exists: this workdir holds blind pilot traces. Judging, analysis and trace "
                         "exports need --unblind, which removes the marker; only the Stage A brief may do that")
    seed = int(cfg["seed"])
    work = Path(cfg["paths"]["workdir"]); work.mkdir(parents=True, exist_ok=True)
    if cfg.get("blind"):
        marker.touch()
    cache = Path(cfg["paths"]["cache_dir"]); cache.mkdir(parents=True, exist_ok=True)
    items_path = Path(cfg["paths"]["items"])
    exclude_path = Path(cfg["paths"].get("items_exclude", items_path.parent / "items_exclude.txt"))
    gen_path, ext_path, jud_path = work / "generations.jsonl", work / "extracted.jsonl", work / "judged.jsonl"
    (work / f"config.resolved.{cfg['name']}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    stages = ["items", "generate", "extract", "judge", "analyze"] if args.stage == "all" else [args.stage]

    if "items" in stages or not items_path.exists():
        if items_path.exists() and not args.rebuild_items:
            print(f"[items] reusing {items_path} (use --rebuild-items to redo it)", file=sys.stderr)
        else:
            its = items_mod.build_items(cfg, items_path, seed, exclude_path)
            print(f"[items] built {len(its)} items -> {items_path}", file=sys.stderr)
    want = cfg["items"].get("sha256")
    if want:
        import hashlib
        got = hashlib.sha256(items_path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"[items] {items_path} has sha256 {got}, the config requires {want}: rebuild it with "
                             "--stage items --rebuild-items and check the exclusion list")
    if args.unblind and not cfg.get("blind") and marker.exists():
        marker.unlink()
        print(f"[blind] removed {marker}: this workdir is unblinded from now on", file=sys.stderr)
    items = items_mod.load_items(items_path)
    limit = cfg["items"].get("limit")
    gen_items = items[: int(limit)] if limit else items

    if args.export_items_review:
        items_mod.export_review_csv(items, args.export_items_review, work / f"items_review_{args.export_items_review}.csv")
        return 0
    if args.dump_traces:
        dump_traces(ext_path, gen_path, items, work / "traces_dump.md", args.dump_traces)
        return 0
    if args.annotate_export:
        annotate.export(ext_path, jud_path, items, work / "annotations_blind.csv", args.annotate_langs.split(","), args.annotate_export, seed)
        return 0
    if args.kappa:
        annotate.kappa(Path(args.kappa), jud_path)
        return 0
    if args.gate:
        planned = [j["gen_id"] for j in generate.plan(gen_items, cfg["cells"], cfg["sampling"]["k"], seed)]
        try:
            recs = gate.planned_records(analysis.load_jsonl(ext_path), planned)
        except gate.GateInputError as e:
            raise SystemExit(f"[gate] {e}")
        cells = [cell_key(c) for c in cfg["cells"]]
        res = gate.smoke(items, recs, cells) if args.gate == "smoke" else gate.pilot(items, recs, cells)
        gate.write(res, work, recs)
        return 0

    n_gen = len(gen_items) * len(cfg["cells"]) * cfg["sampling"]["k"]
    print(f"[plan] {len(gen_items)} items x {len(cfg['cells'])} cells x k={cfg['sampling']['k']} = {n_gen} generations", file=sys.stderr)
    sc = cfg.get("server") or {}
    if "ctx_per_slot" in sc:
        print(f"[plan] budget: {sc.get('parallel')} slots x {sc['ctx_per_slot']} ctx, max_tokens "
              f"{cfg['sampling'].get('max_tokens')}; a trace gets min(max_tokens, ctx_per_slot - prompt - 8)",
              file=sys.stderr)

    if "generate" in stages:
        with server.LlamaServer(cfg["server"], cache):
            generate.generate(gen_items, cfg, gen_path)

    if "extract" in stages:
        extract.extract_all(gen_path, ext_path, cache, use_langid=not args.no_langid)

    if "judge" in stages and cfg["judge"].get("enabled", True):
        mine = {cell_key(normalize_cell(c)) for c in cfg["cells"]}
        judge.judge_all([e for e in analysis.load_jsonl(ext_path) if e.get("cell_key") in mine],
                        {i["item_id"]: i for i in items}, cfg["judge"], jud_path, cache, seed)

    if "analyze" in stages:
        headline = analysis.write_report(items, analysis.load_jsonl(ext_path), analysis.load_jsonl(jud_path), work / "report", hint_aware=True)
        (work / "headline.json").write_text(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
