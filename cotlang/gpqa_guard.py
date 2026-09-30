from __future__ import annotations

import csv
import json
import lzma
import re
import subprocess
import sys
from pathlib import Path

SHINGLE = 10
MIN_OPTION = 40
WS = re.compile(r"\s+")
WORD = re.compile(r"\w+", re.UNICODE)
FIELDS = ("Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")


def _norm(s: str) -> str:
    return WS.sub(" ", s).strip().lower()


def _shingles(text: str) -> set[tuple[str, ...]]:
    w = WORD.findall(text.lower())
    return {tuple(w[i:i + SHINGLE]) for i in range(len(w) - SHINGLE + 1)}


def needles_from_rows(rows: list[dict]) -> dict:
    """{"shingles": {10-word tuple: record id}, "strings": [(option text, record id)]} for the given GPQA rows."""
    sh, strings = {}, []
    for r in rows:
        rid = r.get("Record ID", "?").strip()
        for k in FIELDS:
            t = r.get(k, "") or ""
            for s in _shingles(t):
                sh.setdefault(s, rid)
            if k != "Question" and len(_norm(t)) >= MIN_OPTION and len(WORD.findall(t)) < SHINGLE:
                strings.append((_norm(t), rid))
    return {"shingles": sh, "strings": strings}


def load_gpqa_rows(dataset: str = "Idavidrein/gpqa", configs: tuple[str, ...] = ("gpqa_diamond", "gpqa_main")) -> list[dict]:
    """Rows of the GPQA CSVs (gated: needs a valid Hugging Face login with GPQA's terms accepted, or a cached copy)."""
    from huggingface_hub import hf_hub_download
    rows = []
    for c in configs:
        with open(hf_hub_download(dataset, f"{c}.csv", repo_type="dataset"), encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    return rows


def _json_strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in list(obj.keys()) + list(obj.values()) for s in _json_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _json_strings(v)]
    return []


def to_text(name: str, data: bytes) -> str | None:
    """The text to scan for one file's bytes, or None if it cannot be read as UTF-8 text."""
    if name.endswith(".xz"):
        try:
            data = lzma.decompress(data)
        except lzma.LZMAError:
            return None
        name = name[:-3]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if name.endswith((".json", ".jsonl")):
        extra = []
        for chunk in ([text] if name.endswith(".json") else text.splitlines()):
            try:
                extra += _json_strings(json.loads(chunk))
            except json.JSONDecodeError:
                continue
        text = text + "\n" + "\n".join(extra)
    return text


def scan_texts(texts: dict[str, str], needles: dict) -> list[tuple[str, str]]:
    """(name, record id) for every text that contains a needle."""
    hits = []
    for name, raw in texts.items():
        norm = _norm(raw)
        found = {needles["shingles"][s] for s in _shingles(raw) if s in needles["shingles"]}
        found |= {rid for n, rid in needles["strings"] if n in norm}
        hits += [(name, rid) for rid in sorted(found)]
    return hits


def read_paths(paths: list[str]) -> tuple[dict[str, str], list[str], list[str]]:
    """(texts, missing, unreadable) for files on disk."""
    texts, missing, unreadable = {}, [], []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            missing.append(p)
            continue
        t = to_text(p, path.read_bytes())
        (unreadable.append(p) if t is None else texts.__setitem__(p, t))
    return texts, missing, unreadable


def read_staged(cwd: str | None = None) -> tuple[dict[str, str], list[str], list[str]]:
    """(texts, missing, unreadable) for the staged version of every added, copied, modified or renamed file."""
    names = subprocess.run(["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"], cwd=cwd,
                           capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    texts, missing, unreadable = {}, [], []
    for n in filter(None, names):
        r = subprocess.run(["git", "show", f":{n}"], cwd=cwd, capture_output=True)
        if r.returncode != 0:
            missing.append(n)
            continue
        t = to_text(n, r.stdout)
        (unreadable.append(n) if t is None else texts.__setitem__(n, t))
    return texts, missing, unreadable


def main(argv: list[str]) -> int:
    if argv == ["--staged"]:
        texts, missing, unreadable = read_staged()
    elif argv and "--staged" not in argv:
        texts, missing, unreadable = read_paths(argv)
    else:
        print("usage: python -m cotlang.gpqa_guard --staged | FILE...", file=sys.stderr)
        return 2
    hits = scan_texts(texts, needles_from_rows(load_gpqa_rows())) if texts else []
    for name, rid in hits:
        print(f"[gpqa_guard] GPQA text found in {name} (record {rid})")
    for name in missing:
        print(f"[gpqa_guard] ERROR: {name} is missing (or not staged); nothing was checked for it")
    for name in unreadable:
        print(f"[gpqa_guard] ERROR: {name} is not readable UTF-8 text; it cannot be checked, do not commit it unchecked")
    print(f"[gpqa_guard] {len(texts)} files scanned, {len(missing)} missing, {len(unreadable)} unreadable, {len(hits)} hits")
    return 2 if (missing or unreadable) else (1 if hits else 0)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:
        print(f"[gpqa_guard] ERROR: the guard could not run: {e!r}", file=sys.stderr)
        sys.exit(3)
