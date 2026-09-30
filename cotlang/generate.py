from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from tqdm import tqdm

from .prompts import STOP, build_raw_prompt, cell_key, normalize_cell, stable_seed

CTX_MARGIN = 8
MIN_COMPLETION = 1024
MAX_CONSECUTIVE_FAILURES = 6
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty")


def plan(items: list[dict], cells: list[dict], k: int, seed: int) -> list[dict]:
    """All jobs: one per item x cell x sample index, each with a stable id, seed and prompt."""
    jobs = []
    for cell in cells:
        c = normalize_cell(cell)
        ck = cell_key(c)
        for it in items:
            for i in range(k):
                jobs.append({
                    "gen_id": f"{it['item_id']}|{ck}|{i}",
                    "item_id": it["item_id"],
                    "cell_key": ck,
                    "cell": c,
                    "sample_idx": i,
                    "seed": stable_seed(seed, it["item_id"], ck, i),
                    "prompt": build_raw_prompt(it, c),
                })
    return jobs


def load_done(path: Path) -> set[str]:
    """gen_ids already present in the output file."""
    done = set()
    if path.exists():
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["gen_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


async def _count_tokens(client: httpx.AsyncClient, base_url: str, prompt: str) -> int | None:
    """Ask the server how many tokens the prompt is (None if the endpoint is unavailable).
    The two servers we use disagree on the field name: llama.cpp's /tokenize takes {"content": ...} and vLLM's takes
    {"prompt": ...}, and the wrong one is a 400 that would silently fall back to a character estimate, losing the
    prompt-token count for every record. Try both, and count either response shape."""
    for field in ("prompt", "content"):
        try:
            r = await client.post(f"{base_url}/tokenize", json={field: prompt}, timeout=60)
            r.raise_for_status()
            d = r.json()
            toks = d.get("tokens") if isinstance(d, dict) else None
            if toks is None and isinstance(d, dict):
                toks = d.get("count")
            if isinstance(toks, int):
                return toks
            if toks is not None:
                return len(toks)
        except (httpx.HTTPError, KeyError, ValueError):
            continue
    return None


async def _healthy(client: httpx.AsyncClient, base_url: str) -> bool:
    """True if the server answers 200 on /health within 10 s."""
    try:
        return (await client.get(f"{base_url}/health", timeout=10)).status_code == 200
    except httpx.HTTPError:
        return False


async def _one(client: httpx.AsyncClient, sem: asyncio.Semaphore, base_url: str, job: dict,
               samp: dict, ctx_per_slot: int, timeout_s: float) -> dict:
    """Run one job; return the record to write. Raises only on a malformed 200 response (no JSON, no choices), which
    stops the run with a traceback and records nothing."""
    meta = {k: v for k, v in job.items() if k != "prompt"}
    async with sem:
        t0 = time.time()
        n_prompt = await _count_tokens(client, base_url, job["prompt"])
        est = n_prompt if n_prompt is not None else int(len(job["prompt"]) / 2.5)
        room = ctx_per_slot - est - CTX_MARGIN
        if room < MIN_COMPLETION:
            return {**meta, "n_prompt": n_prompt, "error": f"prompt too long for slot: {est} tokens, {room} left"}
        max_tokens = min(int(samp["max_tokens"]), room)
        body = {"prompt": job["prompt"], "max_tokens": max_tokens, "seed": job["seed"], "stop": STOP, "stream": False}
        body.update({k: samp[k] for k in SAMPLING_KEYS if k in samp})
        for attempt in range(3):
            try:
                r = await client.post(f"{base_url}/v1/completions", json=body,
                                      timeout=httpx.Timeout(timeout_s, connect=15, pool=15))
                r.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    return {**meta, "n_prompt": n_prompt, "transient": True,
                            "error": f"HTTP {e.response.status_code} (rejected): {e.response.text[:300]}"}
                err = e
            except httpx.HTTPError as e:
                err = e
            if attempt == 2 or not await _healthy(client, base_url):
                return {**meta, "n_prompt": n_prompt, "error": repr(err), "transient": True}
            await asyncio.sleep(2 * (attempt + 1))
        data = r.json()
        ch = data["choices"][0]
        usage = data.get("usage", {})
        return {
            **meta,
            "text": ch.get("text", ""),
            "finish_reason": ch.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens", n_prompt),
            "n_prompt": n_prompt,
            "max_tokens_used": max_tokens,
            "model": data.get("model"),
            "sampling": {k: samp[k] for k in SAMPLING_KEYS if k in samp},
            "elapsed_s": round(time.time() - t0, 2),
        }


async def _run(jobs: list[dict], out_path: Path, base_url: str, samp: dict, concurrency: int,
               ctx_per_slot: int, timeout_s: float) -> None:
    """Run all jobs concurrently and append each record as soon as it finishes."""
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(_one(client, sem, base_url, j, samp, ctx_per_slot, timeout_s)) for j in jobs]
        n_err, n_fail, consecutive = 0, 0, 0
        with out_path.open("a", encoding="utf-8") as f, tqdm(total=len(tasks), desc="generate", file=sys.stderr) as bar:
            for coro in asyncio.as_completed(tasks):
                rec = await coro
                bar.update(1)
                if rec.pop("transient", False):
                    n_fail += 1
                    consecutive += 1
                    print(f"[generate] request failed, not recorded, retried on the next run: {rec['gen_id']}: {rec['error']}",
                          file=sys.stderr)
                    if consecutive >= MAX_CONSECUTIVE_FAILURES or not await _healthy(client, base_url):
                        for t in tasks:
                            t.cancel()
                        raise RuntimeError(f"[generate] {consecutive} request(s) failed in a row or the server stopped answering "
                                           "/health; stopping. Nothing was recorded for them; rerun the same command to resume.")
                    continue
                consecutive = 0
                if "error" in rec:
                    n_err += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
        if n_err:
            print(f"[generate] {n_err} of {len(tasks)} requests ended in an error record", file=sys.stderr)
        if n_fail:
            raise RuntimeError(f"[generate] {n_fail} of {len(tasks)} requests failed and were not recorded; "
                               "rerun the same command to retry them.")


def close_partial_last_line(path: Path) -> bool:
    """If the file's last line has no newline (a run killed mid-write), add one so the next record starts on its own
    line. The partial line stays in the file, unreadable; load_done and extract skip it. Returns True if it repaired."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        f.seek(-1, 2)
        if f.read(1) == b"\n":
            return False
    with path.open("ab") as f:
        f.write(b"\n")
    print(f"[generate] WARNING: the last line of {path} was incomplete (a killed run?); closed it with a newline", file=sys.stderr)
    return True


def generate(items: list[dict], cfg: dict, out_path: Path) -> None:
    """Entry point: plan, skip what is done, run the rest."""
    jobs = plan(items, cfg["cells"], cfg["sampling"]["k"], cfg["seed"])
    close_partial_last_line(out_path)
    done = load_done(out_path)
    todo = [j for j in jobs if j["gen_id"] not in done]
    print(f"[generate] {len(jobs)} planned, {len(done)} done, {len(todo)} to run", file=sys.stderr)
    if not todo:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scfg = cfg["server"]
    asyncio.run(_run(todo, out_path, scfg["base_url"].rstrip("/"), cfg["sampling"], int(scfg["parallel"]),
                     int(scfg["ctx_per_slot"]), float(cfg["sampling"].get("request_timeout_s", 1800))))
