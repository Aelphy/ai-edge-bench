"""LiteRT C++ Tensor API driver wrapper for the AI Edge Bench harness.

Drives the compiled C++ `gemma4_tensor_driver_{xnnpack,ynnpack}` binary over the
AI Edge Bench JSON-lines protocol (`ready`, `tokenize`, `generate`, `quit`).

Two execution modes are supported:
  1. Direct exec mode (when a SentencePiece `tokenizer.model` is present):
     Replaces the current Python process in-place via `os.execvp` with the C++
     `gemma4_tensor_driver_{xnnpack,ynnpack}` binary so `aeb.monitor` tracks the
     native C++ process directly with zero wrapper overhead.
  2. HuggingFace `tokenizer.json` proxy mode (when only `tokenizer.json` is
     staged in the model directory):
     Loads `tokenizers.Tokenizer.from_file("<model>/tokenizer.json")` in Python,
     spawns `gemma4_tensor_driver_{xnnpack,ynnpack}` as a child process with
     pre-tokenized `prompt_ids`, and decodes `gen_ids` into `text`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BOS_TOKEN_ID = 2


def emit(obj: dict) -> None:
  sys.stdout.write(json.dumps(obj) + "\n")
  sys.stdout.flush()


def resolve_driver_binary(use_ynnpack: bool, explicit_bin: str = "") -> str:
  """Locate `gemma4_tensor_driver_xnnpack` or `gemma4_tensor_driver_ynnpack`."""
  if explicit_bin:
    p = Path(explicit_bin)
    if p.is_file() and os.access(p, os.X_OK):
      return str(p)
    raise FileNotFoundError(
        f"explicit --driver-bin not found or not executable: {explicit_bin}"
    )

  env_bin = os.environ.get("AEB_LITERT_TENSOR_DRIVER_BIN", "")
  if env_bin:
    p = Path(env_bin)
    if p.is_file() and os.access(p, os.X_OK):
      return str(p)

  bin_name = (
      "gemma4_tensor_driver_ynnpack"
      if use_ynnpack
      else "gemma4_tensor_driver_xnnpack"
  )
  candidates = [
      shutil.which(bin_name),
      f"/opt/litert-tensor/bin/{bin_name}",
      shutil.which("gemma4_tensor_driver"),
      "/opt/litert-tensor/bin/gemma4_tensor_driver",
  ]
  for cand in candidates:
    if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
      return str(cand)

  raise FileNotFoundError(
      f"Could not find LiteRT Tensor API driver binary ({bin_name!r}). "
      "Ensure /opt/litert-tensor/bin is on PATH or pass --driver-bin."
  )


def resolve_tokenizer_paths(
    model_path: str, explicit_tokenizer: str = ""
) -> tuple[str | None, str | None]:
  """Return `(sp_model_path, hf_tokenizer_json_path)` if found on disk."""
  if explicit_tokenizer:
    p = Path(explicit_tokenizer)
    if p.is_file():
      if p.name.endswith(".json"):
        return None, str(p)
      return str(p), None

  mpath = Path(model_path)
  base_dir = mpath if mpath.is_dir() else mpath.parent
  sp_path = base_dir / "tokenizer.model"
  hf_path = base_dir / "tokenizer.json"
  return (
      str(sp_path) if sp_path.is_file() else None,
      str(hf_path) if hf_path.is_file() else None,
  )


def build_cpp_cmd(
    bin_path: str, args: argparse.Namespace, sp_tokenizer_path: str | None
) -> list[str]:
  """Build the CLI command line for the C++ `gemma4_tensor_driver` binary."""
  backend_name = (
      "ynnpack"
      if (args.ynnpack or args.backend == "ynnpack")
      else "xnnpack"
  )
  cmd = [
      bin_path,
      f"--weights={args.model}",
      f"--num_threads={args.threads}",
      f"--ctx={args.ctx}",
      f"--backend_name={backend_name}",
      f"--dtype={args.dtype}",
  ]
  if sp_tokenizer_path:
    cmd.append(f"--tokenizer={sp_tokenizer_path}")
  if args.cache_dir:
    cmd.append(f"--cache_dir={args.cache_dir}")
  if args.weight_cache:
    cmd.append(f"--weight_cache={args.weight_cache}")
  if args.cache_prefix:
    cmd.append("--cache_prefix=true")
  return cmd


class HfTokenizerProxy:
  """Wraps `tokenizers.Tokenizer` + C++ `gemma4_tensor_driver` subprocess."""

  def __init__(self, cmd: list[str], hf_tokenizer_json: str | None):
    self.cmd = cmd
    self.tokenizer = None
    if hf_tokenizer_json:
      from tokenizers import Tokenizer  # pylint: disable=g-import-not-at-top

      self.tokenizer = Tokenizer.from_file(hf_tokenizer_json)
    self.proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        bufsize=1,
        text=True,
    )

  def encode(self, text: str, add_bos: bool = True) -> list[int]:
    if self.tokenizer is None:
      raise RuntimeError(
          "No tokenizer available (neither tokenizer.model nor tokenizer.json)"
      )
    ids = list(self.tokenizer.encode(text, add_special_tokens=False).ids)
    if add_bos and (not ids or ids[0] != BOS_TOKEN_ID):
      return [BOS_TOKEN_ID] + ids
    return ids

  def decode(self, token_ids: list[int]) -> str:
    if self.tokenizer is None or not token_ids:
      return ""
    return self.tokenizer.decode(token_ids, skip_special_tokens=False)

  def _read_cpp(self) -> dict:
    assert self.proc.stdout is not None
    while True:
      line = self.proc.stdout.readline()
      if not line:
        rc = self.proc.wait()
        raise RuntimeError(f"gemma4_tensor_driver exited with code {rc}")
      line = line.strip()
      if not line:
        continue
      try:
        return json.loads(line)
      except json.JSONDecodeError:
        sys.stderr.write(f"[cpp-stdout] {line}\n")
        sys.stderr.flush()

  def _send_cpp(self, payload: dict) -> dict:
    assert self.proc.stdin is not None
    self.proc.stdin.write(json.dumps(payload) + "\n")
    self.proc.stdin.flush()
    return self._read_cpp()

  def serve(self) -> int:
    ready = self._read_cpp()
    emit(ready)
    if ready.get("event") != "ready":
      return 1

    try:
      for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
          continue
        try:
          req = json.loads(line)
        except json.JSONDecodeError as e:
          emit({"ok": False, "error": f"bad json: {e}"})
          continue

        op = req.get("op", "")
        rid = req.get("id")
        try:
          if op == "quit":
            resp = self._send_cpp({"op": "quit", "id": rid})
            emit(resp)
            break
          elif op == "tokenize":
            if self.tokenizer is not None:
              ids = self.encode(
                  req.get("text", ""), add_bos=bool(req.get("add_bos", True))
              )
              emit({"id": rid, "ok": True, "ids": ids})
            else:
              emit(self._send_cpp(req))
          elif op == "generate":
            cpp_req = dict(req)
            if (
                "prompt_ids" not in cpp_req
                and "prompt" in cpp_req
                and self.tokenizer is not None
            ):
              cpp_req["prompt_ids"] = self.encode(
                  cpp_req["prompt"], add_bos=True
              )
            resp = self._send_cpp(cpp_req)
            if (
                resp.get("ok")
                and bool(req.get("return_text", True))
                and not resp.get("text")
                and resp.get("gen_ids")
                and self.tokenizer is not None
            ):
              resp["text"] = self.decode(resp["gen_ids"])
            emit(resp)
          else:
            emit(self._send_cpp(req))
        except Exception as e:  # pylint: disable=broad-except
          emit({"id": rid, "ok": False, "error": str(e)})
    finally:
      if self.proc.poll() is None:
        try:
          self.proc.terminate()
          self.proc.wait(timeout=5)
        except Exception:  # pylint: disable=broad-except
          self.proc.kill()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", required=True, help="Safetensors dir or file")
  ap.add_argument("--tokenizer", default="", help="Optional tokenizer path")
  ap.add_argument("--threads", type=int, default=4)
  ap.add_argument("--ctx", type=int, default=4096)
  ap.add_argument("--cache-dir", default="", help="Weight cache directory")
  ap.add_argument("--weight-cache", default="", help="Weight cache file path")
  ap.add_argument("--ynnpack", action="store_true", help="Use YNNPACK binary")
  ap.add_argument(
      "--backend",
      choices=("xnnpack", "ynnpack", "cpu"),
      default="xnnpack",
      help="CPU backend selection",
  )
  ap.add_argument("--dtype", default="fp32", help="Execution dtype (fp32)")
  ap.add_argument(
      "--cache-prefix",
      action="store_true",
      help="Enable KV cache prefix reuse across requests",
  )
  ap.add_argument(
      "--driver-bin",
      default="",
      help="Explicit path to gemma4_tensor_driver binary",
  )
  ap.add_argument(
      "--no-exec",
      action="store_true",
      help="Always use subprocess proxy mode instead of os.execvp",
  )
  return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  use_ynnpack = bool(args.ynnpack or args.backend == "ynnpack")
  try:
    bin_path = resolve_driver_binary(use_ynnpack, args.driver_bin)
    sp_tok, hf_tok = resolve_tokenizer_paths(args.model, args.tokenizer)
    cmd = build_cpp_cmd(bin_path, args, sp_tok)
    if sp_tok and not args.no_exec:
      os.execvp(bin_path, cmd)
    proxy = HfTokenizerProxy(cmd, hf_tok)
    return proxy.serve()
  except Exception as e:  # pylint: disable=broad-except
    emit({"event": "error", "error": str(e)})
    return 1


if __name__ == "__main__":
  sys.exit(main())
