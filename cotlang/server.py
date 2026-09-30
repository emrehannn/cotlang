from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download

VRAM_WARN_MIB = 1024


def resolve_model(scfg: dict, cache_dir: Path) -> Path:
    """Path of the GGUF; download it into cache_dir/models if it is not there yet."""
    local = cache_dir / "models" / scfg["model_file"]
    if local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(scfg["model_repo"], scfg["model_file"], local_dir=str(local.parent))
    return Path(p)


def gpu_memory_used_mib() -> int | None:
    """GPU memory in use right now, in MiB (None if nvidia-smi is not available)."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return sum(int(x.strip()) for x in out.strip().splitlines() if x.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


class LlamaServer:
    """Context manager: `with LlamaServer(cfg, cache): ...` starts the server before and stops it after."""

    def __init__(self, scfg: dict, cache_dir: Path):
        self.cfg = scfg
        self.cache_dir = cache_dir
        self.base_url = scfg["base_url"].rstrip("/")
        self.proc: subprocess.Popen | None = None

    def healthy(self) -> bool:
        """True if something answers 200 on /health."""
        try:
            return httpx.get(f"{self.base_url}/health", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def command(self, model: Path) -> list[str]:
        """The llama-server command line."""
        port = self.base_url.rsplit(":", 1)[-1]
        par = int(self.cfg["parallel"])
        ctx = par * int(self.cfg["ctx_per_slot"])
        cmd = [self.cfg["binary"], "-m", str(model), "--host", "127.0.0.1", "--port", port,
               "-c", str(ctx), "-np", str(par), *map(str, self.cfg.get("extra_args", []))]
        cmd += ["--cache-ram", str(int(self.cfg.get("cache_ram_mib", 0)))]
        kvq = self.cfg.get("kv_quant")
        if kvq:
            cmd += ["-ctk", str(kvq), "-ctv", str(kvq)]
        return cmd

    def start(self) -> None:
        if self.cfg.get("external"):
            if not self.healthy():
                raise RuntimeError(f"server.external=true but nothing healthy at {self.base_url}")
            return
        if self.healthy():
            raise RuntimeError(f"something is already listening at {self.base_url}; set server.external or stop it")
        binary = self.cfg["binary"]
        if shutil.which(binary) is None and not Path(binary).exists():
            raise RuntimeError(f"{binary} not found; install llama.cpp or set server.binary")
        used = gpu_memory_used_mib()
        if used is not None and used > VRAM_WARN_MIB:
            print(f"[server] WARNING: {used} MiB of GPU memory already in use; the model + KV cache need most of the card", file=sys.stderr)
        model = resolve_model(self.cfg, self.cache_dir)
        cmd = self.command(model)
        log = (self.cache_dir / "llama-server.log").open("a")
        print("[server] launching:", " ".join(cmd), file=sys.stderr)
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        t0 = time.time()
        while time.time() - t0 < float(self.cfg.get("startup_timeout_s", 300)):
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early (code {self.proc.returncode}); see {self.cache_dir/'llama-server.log'}")
            if self.healthy():
                print(f"[server] healthy after {time.time()-t0:.0f}s", file=sys.stderr)
                return
            time.sleep(2)
        self.stop()
        raise RuntimeError("llama-server did not become healthy in time")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
