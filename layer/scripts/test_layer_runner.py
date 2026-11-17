"""Tests for the Level 2 layer catalog, scout, Roofline, and benchmark runner.

Runner tests drive `layer.main` end to end against `FakeLayerBench`, an
in-process stand-in for the C++ `layer_bench` binary, and write into a
temporary run directory, so they need neither a benchmark binary nor a
container. Every JSON artifact is re-read with a parser that rejects the
`NaN`/`Infinity` tokens RFC 8259 forbids.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import contextlib
import copy
import dataclasses
import functools
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO_ROOT / "model" / "docker" / "harness"
if str(HARNESS_DIR) not in sys.path:
  sys.path.insert(0, str(HARNESS_DIR))

# pylint: disable=g-import-not-at-top,wrong-import-position
from aeb import catalog
from aeb import layer

# pylint: enable=g-import-not-at-top,wrong-import-position

LAYER_ROOT = REPO_ROOT / "layer"
SUITES_DIR = LAYER_ROOT / "suites"
REPO_CATALOG = LAYER_ROOT / "catalog" / "gemma_qwen_edge.json"
SAPPHIRE_RAPIDS_CEILINGS = (
    REPO_ROOT
    / "kernel"
    / "reports"
    / "2026-11-17-gemma4-31b-sapphire-rapids"
    / "ceilings_sapphire_rapids.json"
)

# One (preset, family, batch size, sequence length) parity case, small enough
# that tensor dumps stay tiny.
CASE_FLAGS = (
    "--presets",
    "Gemma4-E2B",
    "--families",
    "BM_RmsNorm",
    "--batch-size",
    "2",
    "--seq-len",
    "16",
    "--threads",
    "4",
)
CASE_DUMP_NAME = "Gemma4-E2B__BM_RmsNorm__B2_S16.lmlb"
FIXED_UTC_NOW = "2026-01-02T03:04:05Z"


class FakeLayerBench:
  """In-process stand-in for the C++ `layer_bench` binary.

  Implements the command-line contract the runner relies on: it registers one
  benchmark per (family, preset, batch size, sequence length, thread count) of
  `--suite_file`, keeps those matching `--benchmark_filter`, writes Google
  Benchmark JSON to `--benchmark_out`, and writes a `[batch, seq_q, width]`
  LMLB tensor to `--dump_parity_tensors`.

  Attributes:
    commands: Every command line received, in call order.
  """

  def __init__(
      self,
      *,
      drift: float = 0.0,
      exit_code: int = 0,
      write_dumps: bool = True,
      tensor_width: int = 4,
      first_tensor_value: float | None = None,
      counters: Mapping[str, Any] | None = None,
      report_text: str | None = None,
  ):
    """Initializes the instance.

    Args:
      drift: Offset added to every dumped tensor element. A non-zero drift also
        fails benchmarks the binary compares with its strict gate.
      exit_code: Process exit status; non-zero simulates a crash before any
        output is written.
      write_dumps: Whether requested tensor dumps are written.
      tensor_width: Trailing dimension of dumped tensors.
      first_tensor_value: Replaces the first element of every dumped tensor;
        `None` keeps the generated value.
      counters: Fields merged into every benchmark entry; `None` adds none.
      report_text: Text written to `--benchmark_out` instead of the generated
        report; `None` writes the generated report.
    """
    self.commands: list[list[str]] = []
    self._drift = drift
    self._exit_code = exit_code
    self._write_dumps = write_dumps
    self._tensor_width = tensor_width
    self._first_tensor_value = first_tensor_value
    self._counters = dict(counters or {})
    self._report_text = report_text

  def __call__(
      self, command: Sequence[str]
  ) -> subprocess.CompletedProcess[str]:
    """Runs one `layer_bench` command line.

    Args:
      command: Executable path followed by `--flag=value` arguments.

    Returns:
      The completed process with captured stdout and stderr.
    """
    self.commands.append(list(command))
    if self._exit_code:
      return subprocess.CompletedProcess(
          command, self._exit_code, "", "simulated layer_bench crash\n"
      )
    flags = dict(arg[2:].split("=", 1) for arg in command[1:])
    suite = json.loads(Path(flags["suite_file"]).read_text(encoding="utf-8"))
    name_filter = re.compile(flags["benchmark_filter"])
    # The binary's own gate is strict allclose unless a distribution gate is
    # forwarded; any drift then fails the benchmark.
    strict_gate = "compare_parity_tensors" in flags and not (
        "parity_min_cosine" in flags or "parity_max_l2_rel_err" in flags
    )
    benchmarks = []
    for family, preset, batch_size, seq_len, threads in itertools.product(
        suite["families"],
        [catalog.preset_name(preset) for preset in suite["presets"]],
        suite["batch_sizes"],
        suite["seq_lengths"],
        suite["thread_counts"],
    ):
      name = (
          f"{family}/{preset}/B:{batch_size}/S:{seq_len}/Threads:{threads}"
          "/real_time"
      )
      if not name_filter.search(name):
        continue
      seq_q = 1 if family in catalog.DECODE_ONLY_FAMILIES else seq_len
      failed = strict_gate and self._drift != 0.0
      benchmarks.append(self._benchmark_entry(name, seq_q, seq_len, failed))
      if "dump_parity_tensors" in flags and self._write_dumps:
        self._write_dump(
            Path(flags["dump_parity_tensors"]),
            [batch_size, seq_q, self._tensor_width],
        )
    self._write_report(Path(flags["benchmark_out"]), benchmarks)
    return subprocess.CompletedProcess(command, 0, "fake stdout\n", "")

  def _benchmark_entry(
      self, name: str, seq_q: int, seq_len: int, failed: bool
  ) -> dict[str, Any]:
    """Returns the Google Benchmark entry of one benchmark."""
    return {
        "name": name,
        "real_time": 1.25,
        "cpu_time": 1.20,
        "time_unit": "ms",
        "iterations": 10,
        "GFLOP/s": 1024.0,
        "GB/s": 128.0,
        "ArithmeticIntensity": 8.0,
        "Theoretical_FLOPs": 1.28e9,
        "Theoretical_Bytes": 1.6e8,
        "SeqLenQ": float(seq_q),
        "SeqLenKV": float(seq_len),
        "PeakMem_MB": 12.5,
        "AllocDelta_MB": 4.0,
        "error_occurred": failed,
        "error_message": "strict allclose failed" if failed else "",
        **self._counters,
    }

  def _write_report(self, path: Path, benchmarks: Sequence[Any]) -> None:
    """Writes the Google Benchmark JSON report (or the configured text)."""
    report = json.dumps({
        "context": {"host_name": "test-host", "num_cpus": 32},
        "benchmarks": benchmarks,
    })
    path.write_text(
        report if self._report_text is None else self._report_text,
        encoding="utf-8",
    )

  def _write_dump(self, path: Path, shape: Sequence[int]) -> None:
    """Writes a deterministic ramp tensor of `shape` as an LMLB dump."""
    values = [0.05 * index + self._drift for index in range(math.prod(shape))]
    if self._first_tensor_value is not None:
      values[0] = self._first_tensor_value
    layer.write_parity_dump(path, shape, values)


class TickingBench:
  """Wraps a bench so that a clock advances whenever the binary runs."""

  def __init__(
      self, bench: Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
  ):
    """Initializes the instance.

    Args:
      bench: Stand-in for the `layer_bench` binary to delegate to.
    """
    self._bench = bench
    self._ticks = 0

  def run_command(
      self, command: Sequence[str]
  ) -> subprocess.CompletedProcess[str]:
    """Advances the clock, then runs `command` on the wrapped bench."""
    self._ticks += 1
    return self._bench(command)

  def utc_now(self) -> str:
    """Returns `tick-<n>` after `n` binary invocations."""
    return f"tick-{self._ticks}"


def silent_layer_bench(
    command: Sequence[str],
) -> subprocess.CompletedProcess[str]:
  """Stands in for a `layer_bench` that succeeds without writing any output."""
  return subprocess.CompletedProcess(command, 0, "", "")


def corrupt_dump_layer_bench(
    command: Sequence[str],
) -> subprocess.CompletedProcess[str]:
  """Stands in for a `layer_bench` whose tensor dump is not an LMLB file."""
  flags = dict(arg[2:].split("=", 1) for arg in command[1:])
  Path(flags["dump_parity_tensors"]).write_bytes(b"not an lmlb dump")
  return subprocess.CompletedProcess(command, 0, "", "")


@dataclasses.dataclass(frozen=True)
class RunResult:
  """Outcome of one hermetic `layer.main` invocation.

  Attributes:
    exit_code: Process exit code returned by `layer.main`.
    stderr: Text the runner printed to stderr.
    run_dir: Run directory the injected factory hands out; it exists only if the
      runner got past flag and suite validation.
  """

  exit_code: int
  stderr: str
  run_dir: Path

  def artifact(self, name: str) -> Any:
    """Parses a JSON artifact of the run with a strict RFC 8259 parser."""
    return load_strict_json(self.run_dir / name)


def reject_json_constant(token: str) -> float:
  """Fails on the `NaN`/`Infinity` tokens that RFC 8259 does not allow."""
  raise ValueError(f"non-RFC-8259 JSON token: {token}")


def load_strict_json(path: Path) -> Any:
  """Parses a JSON file, rejecting `NaN` and `Infinity` tokens."""
  return json.loads(
      path.read_text(encoding="utf-8"), parse_constant=reject_json_constant
  )


def write_executable(path: Path) -> Path:
  """Writes an executable stub script at `path` and returns `path`."""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("#!/bin/sh\n", encoding="utf-8")
  path.chmod(0o755)
  return path


def write_stub_sysinfo(path: Path) -> None:
  """Writes a canned host description in place of the `aeb-sysinfo` tool."""
  path.write_text(
      json.dumps({"cpu": "test-cpu", "cores": 32}) + "\n", encoding="utf-8"
  )


def fixed_utc_now() -> str:
  """Returns the constant timestamp `FIXED_UTC_NOW` in place of the clock."""
  return FIXED_UTC_NOW


def stub_build_info() -> dict[str, Any]:
  """Returns canned build provenance in place of hashing the harness."""
  return {"harness_sha256": "test-digest", "repo_rev": "test-rev"}


def new_test_run_dir(root: Path, kind: str, framework: str, label: str) -> Path:
  """Creates the run directory `<root>/<kind>-<framework>-<label>`."""
  run_dir = root / f"{kind}-{framework}-{label}"
  run_dir.mkdir(parents=True)
  return run_dir


def run_layer(
    root: Path,
    argv: Sequence[str],
    bench: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    environ: Mapping[str, str] | None = None,
    utc_now: Callable[[], str] = fixed_utc_now,
) -> RunResult:
  """Runs `layer.main(["--label", "run", *argv])` without touching the host.

  Args:
    root: Scratch directory of the test; the run directory is created in it.
    argv: Runner flags after the fixed `--label run`.
    bench: Stand-in for the `layer_bench` binary; `None` executes the `--binary`
      given in `argv` as a real subprocess.
    environ: Environment variables visible to the runner in addition to an empty
      `PATH`, which keeps binary discovery away from the host; `None` adds none.
    utc_now: Clock of the runner; defaults to the constant `FIXED_UTC_NOW`.

  Returns:
    The exit code, captured stderr, and run directory.
  """
  overrides = {} if bench is None else {"run_command": bench}
  hooks = layer.RunnerHooks(
      new_run_dir=functools.partial(new_test_run_dir, root),
      write_sysinfo=write_stub_sysinfo,
      environ={"PATH": "", **(environ or {})},
      install_bin_dir=root / "no_installed_binaries",
      utc_now=utc_now,
      build_info=stub_build_info,
      **overrides,
  )
  stderr = io.StringIO()
  with contextlib.redirect_stderr(stderr):
    exit_code = layer.main(["--label", "run", *argv], hooks=hooks)
  return RunResult(
      exit_code=exit_code,
      stderr=stderr.getvalue(),
      run_dir=root / "layer-litert-tensor-run",
  )


def with_value(document: Any, path: Sequence[Any], value: Any) -> Any:
  """Returns a deep copy of `document` with `value` stored at `path`.

  Args:
    document: Nested dictionaries and lists.
    path: Keys and list indices leading to the element to replace.
    value: Replacement value.
  """
  mutated = copy.deepcopy(document)
  target = mutated
  for key in path[:-1]:
    target = target[key]
  target[path[-1]] = value
  return mutated


def tiny_catalog() -> dict[str, Any]:
  """Returns a minimal valid catalog: model `Tiny` built from two blocks."""
  return {
      "schema": "aeb.layer.catalog/1",
      "catalog_id": "tiny-catalog",
      "models": {
          "Tiny": {
              "family": "test",
              "total_layers": 3,
              "embed_dim": 64,
              "block_composition": [
                  {"layer_id": "block.Tiny.Local", "count": 2},
                  {"layer_id": "block.Tiny.Global", "count": 1},
              ],
          },
      },
      "composite_blocks": [
          {
              "id": "block.Tiny.Local",
              "preset_name": "Tiny",
              "layer_type": "TransformerBlock",
              "params": {
                  "embed_dim": 64,
                  "hidden_dim": 128,
                  "num_heads": 4,
                  "num_kv_heads": 2,
                  "head_dim": 16,
                  "rotary_dim": 16,
                  "sliding_window": 32,
                  "eps": 1e-6,
              },
          },
          {
              "id": "block.Tiny.Global",
              "preset_name": "Tiny-Global",
              "layer_type": "TransformerBlock",
              "params": {
                  "embed_dim": 64,
                  "hidden_dim": 128,
                  "num_heads": 4,
                  "num_kv_heads": 4,
                  "head_dim": 16,
              },
          },
      ],
  }


def tiny_hyperparams() -> dict[str, Any]:
  """Returns a minimal valid registry with one profile of every kind."""
  return {
      "schema": "aeb.layer.hyperparams/1",
      "regimes": {
          "short": {"mode": "both", "batch_sizes": [1], "seq_lengths": [64]},
          "decode_only": {
              "mode": "decode",
              "batch_sizes": [1],
              "seq_lengths": [8],
          },
      },
      "quant_schemes": {
          "fp32": {
              "weight_dtype": "fp32",
              "quality_gate": {"rtol": 1e-4, "atol": 1e-5},
          },
      },
      "cliff_sweeps": {
          "none": {"deltas": [0], "target_dims": []},
          "plus_minus_8": {"deltas": [-8, 0, 8], "target_dims": ["embed_dim"]},
      },
      "machine_threading": {
          "box": {"thread_counts": [1, 4], "physical_cores_per_domain": 4},
      },
  }


def write_tiny_registry(
    root: Path, catalog_document: Mapping[str, Any] | None = None
) -> Path:
  """Writes a catalog and the tiny registry where suites look by default.

  Args:
    root: Directory receiving `catalog/gemma_qwen_edge.json`,
      `catalog/hyperparams.json`, and an empty `suites` directory.
    catalog_document: Catalog to write; `None` writes `tiny_catalog()`.

  Returns:
    The `suites` directory, for use as the suite directory of a resolution.
  """
  catalog_dir = root / "catalog"
  catalog_dir.mkdir()
  (catalog_dir / "gemma_qwen_edge.json").write_text(
      json.dumps(catalog_document or tiny_catalog()), encoding="utf-8"
  )
  (catalog_dir / "hyperparams.json").write_text(
      json.dumps(tiny_hyperparams()), encoding="utf-8"
  )
  suites_dir = root / "suites"
  suites_dir.mkdir()
  return suites_dir


def write_tiny_suite(root: Path, **fields: Any) -> Path:
  """Writes the tiny registry and a suite benchmarking model `Tiny`.

  Args:
    root: Directory receiving the registry and `suites/tiny.json`.
    **fields: Suite fields added to `{"name": "tiny", "models": ["Tiny"]}`.

  Returns:
    Path of the written suite file.
  """
  suite_path = write_tiny_registry(root) / "tiny.json"
  suite_path.write_text(
      json.dumps({"name": "tiny", "models": ["Tiny"], **fields}),
      encoding="utf-8",
  )
  return suite_path


class TempDirTestCase(unittest.TestCase):
  """Gives every test a scratch directory `self.tmp` removed on teardown."""

  def setUp(self):
    super().setUp()
    temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(temp_dir.cleanup)
    self.tmp = Path(temp_dir.name)


class ParityDumpTest(TempDirTestCase):

  def test_write_parity_dump_emits_little_endian_lmlb_layout(self):
    dump_path = self.tmp / "tensor.lmlb"

    layer.write_parity_dump(dump_path, [2], [1.0, -2.0])

    expected = bytes.fromhex(
        "424c4d4c"  # magic 0x4C4D4C42, little-endian
        "01000000"  # rank 1 (uint32)
        "0200000000000000"  # dimension 2 (int64)
        "0000803f"  # 1.0 (float32)
        "000000c0"  # -2.0 (float32)
    )
    self.assertEqual(dump_path.read_bytes(), expected)

  def test_read_parity_dump_round_trips_written_tensor(self):
    dump_path = self.tmp / "tensor.lmlb"
    layer.write_parity_dump(dump_path, [1, 4], [3.0, 0.0, -4.0, 0.5])

    shape, values = layer.read_parity_dump(dump_path)

    self.assertEqual(shape, [1, 4])
    self.assertEqual(values, [3.0, 0.0, -4.0, 0.5])

  def test_read_parity_dump_header_reports_shape_and_element_count(self):
    dump_path = self.tmp / "tensor.lmlb"
    layer.write_parity_dump(dump_path, [2, 3], [0.0] * 6)

    shape, n_elems = layer.read_parity_dump_header(dump_path)

    self.assertEqual(shape, [2, 3])
    self.assertEqual(n_elems, 6)

  def test_write_parity_dump_creates_missing_parent_directories(self):
    dump_path = self.tmp / "nested" / "dir" / "tensor.lmlb"

    layer.write_parity_dump(dump_path, [1], [7.0])

    self.assertEqual(layer.read_parity_dump(dump_path), ([1], [7.0]))

  def test_read_parity_dump_header_rejects_malformed_files(self):
    magic = "424c4d4c"
    cases = (
        ("too_short", "424c", "too short"),
        ("bad_magic", "58585858" + "01000000", "Invalid LMLB magic"),
        ("rank_zero", magic + "00000000", "Invalid tensor rank"),
        ("rank_above_16", magic + "11000000", "Invalid tensor rank"),
        (
            "truncated_shape",
            magic + "03000000" + "0100000000000000",
            "Truncated shape header",
        ),
        (
            "negative_dim",
            magic + "01000000" + "feffffffffffffff",
            "Non-positive dimension -2",
        ),
        (
            "payload_too_small",
            magic + "01000000" + "0400000000000000" + "0000803f",
            "Payload size mismatch",
        ),
        (
            "payload_too_large",
            magic + "01000000" + "0100000000000000" + "0000803f" + "0000803f",
            "Payload size mismatch",
        ),
    )
    for name, hex_bytes, message in cases:
      with self.subTest(case=name):
        dump_path = self.tmp / f"{name}.lmlb"
        dump_path.write_bytes(bytes.fromhex(hex_bytes))

        with self.assertRaisesRegex(ValueError, message):
          layer.read_parity_dump_header(dump_path)

  def test_read_parity_dump_rejects_malformed_file(self):
    dump_path = self.tmp / "garbage.lmlb"
    dump_path.write_bytes(b"not an lmlb dump")

    with self.assertRaisesRegex(ValueError, "Invalid LMLB magic"):
      layer.read_parity_dump(dump_path)

  def test_read_parity_dump_header_raises_os_error_for_missing_file(self):
    with self.assertRaisesRegex(FileNotFoundError, "absent.lmlb"):
      layer.read_parity_dump_header(self.tmp / "absent.lmlb")

  def test_write_parity_dump_rejects_invalid_shape_or_payload(self):
    cases = (
        ("rank_zero", [], [], "Invalid rank 0"),
        ("rank_above_16", [1] * 17, [0.0], "Invalid rank 17"),
        ("zero_dim", [2, 0], [], "Non-positive dimension 0"),
        ("negative_dim", [2, -1], [1.0], "Non-positive dimension -1"),
        ("too_few_values", [2, 2], [1.0, 2.0], "shape product 4"),
    )
    for name, shape, values, message in cases:
      with self.subTest(case=name):
        with self.assertRaisesRegex(ValueError, message):
          layer.write_parity_dump(self.tmp / f"{name}.lmlb", shape, values)


class CompareParityDumpsTest(TempDirTestCase):

  def compare(
      self,
      reference: Sequence[float],
      candidate: Sequence[float],
      candidate_shape: Sequence[int] = (1, 4),
      **gates: float,
  ) -> dict[str, Any]:
    """Compares a `[1, 4]` reference dump against a candidate dump."""
    ref_path = self.tmp / "reference.lmlb"
    cand_path = self.tmp / "candidate.lmlb"
    layer.write_parity_dump(ref_path, [1, 4], reference)
    layer.write_parity_dump(cand_path, candidate_shape, candidate)
    return layer.compare_parity_dumps(ref_path, cand_path, **gates)

  def test_candidate_within_tolerance_passes_strict_allclose(self):
    result = self.compare(
        [3.0, 0.0, -4.0, 0.0], [3.0 + 1e-6, 0.0, -4.0 - 1e-6, 0.0]
    )

    self.assertTrue(result["passed"])
    self.assertEqual(result["details"], "PASSED")
    self.assertEqual(result["mismatch_count"], 0)
    self.assertAlmostEqual(result["cosine_similarity"], 1.0, places=9)
    self.assertLess(result["l2_relative_error"], 1e-6)
    self.assertLess(result["kl_divergence"], 1e-10)

  def test_elements_outside_tolerance_are_counted_as_mismatches(self):
    result = self.compare([1.0, 2.0, 3.0, 4.0], [1.0, 2.5, 3.0, 4.0])

    self.assertFalse(result["passed"])
    self.assertEqual(result["mismatch_count"], 1)
    self.assertEqual(result["total_elements"], 4)
    self.assertAlmostEqual(result["max_abs_error"], 0.5, places=6)
    self.assertAlmostEqual(result["max_rel_error"], 0.25, places=6)

  def test_scaled_candidate_fails_strict_allclose(self):
    result = self.compare([3.0, 0.0, -4.0, 0.0], [4.5, 0.0, -6.0, 0.0])

    self.assertFalse(result["passed"])
    self.assertAlmostEqual(result["cosine_similarity"], 1.0, places=9)
    self.assertAlmostEqual(result["l2_relative_error"], 0.5, places=6)

  def test_scaled_candidate_passes_enabled_distribution_gates(self):
    result = self.compare(
        [3.0, 0.0, -4.0, 0.0],
        [4.5, 0.0, -6.0, 0.0],
        min_cosine=0.999,
        max_l2_rel_err=0.6,
        max_kl_divergence=0.25,
    )

    self.assertTrue(result["passed"])

  def test_kl_gate_rejects_scaled_candidate(self):
    result = self.compare(
        [3.0, 0.0, -4.0, 0.0], [4.5, 0.0, -6.0, 0.0], max_kl_divergence=1e-4
    )

    self.assertFalse(result["passed"])
    self.assertGreater(result["kl_divergence"], 1e-4)

  def test_kl_divergence_is_mean_row_kl_of_candidate_from_reference(self):
    ref_path = self.tmp / "reference.lmlb"
    cand_path = self.tmp / "candidate.lmlb"
    layer.write_parity_dump(ref_path, [2, 2], [1.0, 0.0, 0.0, 0.0])
    layer.write_parity_dump(cand_path, [2, 2], [0.0, 0.0, 0.0, 0.0])

    result = layer.compare_parity_dumps(ref_path, cand_path)

    # Row 0: KL(softmax([1, 0]) || [1/2, 1/2]) = ln 2 - H(softmax([1, 0])) =
    # 0.110944072. Row 1 compares equal rows and adds 0, so the mean over the
    # two rows is 0.055472036. KL of the reference from the candidate would
    # instead give 0.060057253.
    self.assertAlmostEqual(result["kl_divergence"], 0.055472036, places=8)

  def test_l2_gate_rejects_scaled_candidate(self):
    result = self.compare(
        [3.0, 0.0, -4.0, 0.0], [4.5, 0.0, -6.0, 0.0], max_l2_rel_err=0.4
    )

    self.assertFalse(result["passed"])

  def test_cosine_gate_rejects_orthogonal_candidate(self):
    result = self.compare(
        [3.0, 0.0, -4.0, 0.0], [0.0, 3.0, 0.0, 4.0], min_cosine=0.95
    )

    self.assertFalse(result["passed"])
    self.assertAlmostEqual(result["cosine_similarity"], 0.0, places=9)
    self.assertAlmostEqual(result["l2_relative_error"], 1.41421356, places=6)

  def test_non_finite_candidate_fails_even_with_loose_gates(self):
    result = self.compare(
        [3.0, 0.0, -4.0, 0.0],
        [3.0, float("nan"), -4.0, float("inf")],
        min_cosine=0.5,
        max_l2_rel_err=10.0,
        max_kl_divergence=10.0,
    )

    self.assertFalse(result["passed"])
    self.assertEqual(result["mismatch_count"], 2)
    self.assertTrue(math.isnan(result["cosine_similarity"]))
    self.assertTrue(math.isinf(result["l2_relative_error"]))
    self.assertTrue(math.isinf(result["kl_divergence"]))
    self.assertTrue(math.isinf(result["max_abs_error"]))
    self.assertIn("2 element pair(s) hold NaN or Inf", result["details"])

  def test_non_zero_candidate_against_zero_reference_fails(self):
    result = self.compare([0.0, 0.0, 0.0, 0.0], [1e-4, 0.0, 0.0, 0.0])

    self.assertFalse(result["passed"])
    self.assertEqual(result["max_rel_error"], 0.0)
    self.assertEqual(result["cosine_similarity"], 0.0)
    self.assertTrue(math.isinf(result["l2_relative_error"]))
    self.assertIn("reference tensor is all zeros", result["details"])

  def test_two_zero_tensors_pass_with_unit_cosine(self):
    result = self.compare([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])

    self.assertTrue(result["passed"])
    self.assertEqual(result["cosine_similarity"], 1.0)
    self.assertEqual(result["l2_relative_error"], 0.0)

  def test_shape_mismatch_fails_and_names_both_shapes(self):
    result = self.compare(
        [3.0, 0.0, -4.0, 0.0],
        [3.0, 0.0, -4.0, 0.0],
        candidate_shape=[2, 2],
        min_cosine=0.9,
        max_l2_rel_err=0.5,
    )

    self.assertFalse(result["passed"])
    self.assertTrue(math.isnan(result["cosine_similarity"]))
    self.assertEqual(result["details"], "Shape mismatch: [1, 4] vs [2, 2]")


def probe_ceilings() -> dict[str, Any]:
  """Returns hardware ceilings measured at 1 and 8 threads."""
  return {
      "threads_numa_node0": 8,
      "compute_ceilings_gops": {
          "avx512_fp32_fma": {"single_core": 100.0, "numa_node0_8c": 800.0},
          "amx_int8_tdpbssd": {"single_core": 400.0, "numa_node0_8c": 3200.0},
          "amx_bf16_tdpbf16ps": {"single_core": 200.0, "numa_node0_8c": 1600.0},
      },
      "bandwidth_ceilings_gbps": {
          "dram": {
              "stream_triad": {"single_core": 10.0, "numa_node0_8c": 45.0},
          },
      },
  }


class PracticalCeilingsTest(unittest.TestCase):

  def test_ceilings_interpolate_linearly_between_measured_thread_counts(self):
    # 4 threads sit 3/7 of the way from the 1-thread to the 8-thread knot.
    cases = (
        ("at_single_core", 1, 100.0, 10.0),
        ("between_knots", 4, 400.0, 25.0),
        ("at_last_knot", 8, 800.0, 45.0),
        ("beyond_last_knot", 16, 800.0, 45.0),
    )
    for name, threads, expected_gops, expected_gbps in cases:
      with self.subTest(case=name):
        gops, gbps = layer.select_practical_ceilings(
            probe_ceilings(), threads=threads
        )

        self.assertAlmostEqual(gops, expected_gops, places=6)
        self.assertAlmostEqual(gbps, expected_gbps, places=6)

  def test_compute_ceiling_follows_dtype_instruction_set(self):
    cases = (
        ("fp32", 100.0),
        ("bf16", 200.0),
        ("fp16", 200.0),
        ("int8_channel", 400.0),
        ("int4_block", 400.0),
        ("int2_block", 400.0),
        ("qd8_dynamic", 400.0),
    )
    for dtype, expected_gops in cases:
      with self.subTest(dtype=dtype):
        gops, _ = layer.select_practical_ceilings(
            probe_ceilings(), threads=1, dtype=dtype
        )

        self.assertEqual(gops, expected_gops)

  def test_quantized_dtype_falls_back_to_fp32_probe_when_int8_is_missing(self):
    ceilings = probe_ceilings()
    del ceilings["compute_ceilings_gops"]["amx_int8_tdpbssd"]

    gops, _ = layer.select_practical_ceilings(
        ceilings, threads=1, dtype="int8_channel"
    )

    self.assertEqual(gops, 100.0)

  def test_unsuffixed_numa_key_is_placed_at_numa_thread_count(self):
    ceilings = {
        "threads_numa_node0": 4,
        "compute_ceilings_gops": {
            "avx512_fp32_fma": {"single_core": 10.0, "numa_node0": 40.0},
        },
    }

    gops_at_numa, _ = layer.select_practical_ceilings(ceilings, threads=4)
    gops_halfway, _ = layer.select_practical_ceilings(ceilings, threads=2)

    self.assertEqual(gops_at_numa, 40.0)
    self.assertAlmostEqual(gops_halfway, 20.0, places=6)

  def test_missing_probes_yield_zero_ceilings(self):
    self.assertEqual(layer.select_practical_ceilings({}, threads=8), (0.0, 0.0))

  def test_repo_sapphire_rapids_probe_interpolates_across_numa_knots(self):
    ceilings = layer.load_hardware_ceilings(SAPPHIRE_RAPIDS_CEILINGS)
    cases = (
        (1, 126.73, 14.70),
        (14, 1765.86, 125.27),
        (28, 3531.07, 244.34),
        (42, 5301.80, 241.41),
        (56, 7072.53, 238.47),
    )
    for threads, expected_gops, expected_gbps in cases:
      with self.subTest(threads=threads):
        gops, gbps = layer.select_practical_ceilings(
            ceilings, threads=threads, dtype="fp32"
        )

        self.assertAlmostEqual(gops, expected_gops, places=2)
        self.assertAlmostEqual(gbps, expected_gbps, places=2)

  def test_repo_sapphire_rapids_probe_has_faster_quantized_ceilings(self):
    ceilings = layer.load_hardware_ceilings(SAPPHIRE_RAPIDS_CEILINGS)
    cases = (
        ("int8_channel", 112594.40),
        ("int4_block", 112594.40),
        ("int2_block", 112594.40),
        ("bf16", 55750.15),
    )
    for dtype, expected_gops in cases:
      with self.subTest(dtype=dtype):
        gops, _ = layer.select_practical_ceilings(
            ceilings, threads=28, dtype=dtype
        )

        self.assertAlmostEqual(gops, expected_gops, places=2)


class RooflineInputFilesTest(TempDirTestCase):

  def test_load_hardware_ceilings_rejects_non_object_json(self):
    path = self.tmp / "ceilings.json"
    path.write_text("[1, 2]", encoding="utf-8")

    with self.assertRaisesRegex(ValueError, "expected a JSON object"):
      layer.load_hardware_ceilings(path)

  def test_load_kernel_timings_returns_milliseconds_by_key(self):
    path = self.tmp / "timings.json"
    path.write_text('{"Tiny/BM_RmsNorm": 1.5, "RoPE": 2}', encoding="utf-8")

    timings = layer.load_kernel_timings(path)

    self.assertEqual(timings, {"Tiny/BM_RmsNorm": 1.5, "RoPE": 2.0})

  def test_load_kernel_timings_rejects_invalid_documents(self):
    cases = (
        ("not_an_object", "[1.5]", "expected a JSON object"),
        ("malformed", "{", "invalid JSON"),
        ("string_value", '{"RoPE": "fast"}', "must be a finite number"),
        ("boolean_value", '{"RoPE": true}', "must be a finite number"),
        ("nan_value", '{"RoPE": NaN}', "must be a finite number"),
    )
    for name, text, message in cases:
      with self.subTest(case=name):
        path = self.tmp / f"{name}.json"
        path.write_text(text, encoding="utf-8")

        with self.assertRaisesRegex(ValueError, message):
          layer.load_kernel_timings(path)


class SubgraphRooflineTest(unittest.TestCase):

  def test_compute_bound_layer_reports_compute_regime(self):
    # 1e8 FLOPs at 100 GOPS take 1.0 ms; 5e6 bytes at 10 GB/s take 0.5 ms.
    record = {
        "threads": 1,
        "real_time_ms": 2.0,
        "theoretical_flops": 1e8,
        "theoretical_bytes": 5e6,
    }

    roofline = layer.compute_subgraph_roofline(record, probe_ceilings())

    self.assertEqual(roofline["bound_regime"], "compute")
    self.assertAlmostEqual(roofline["t_roofline_ms"], 1.0, places=9)
    self.assertAlmostEqual(
        roofline["whole_subgraph_roofline_pct"], 50.0, places=6
    )
    self.assertEqual(roofline["c_prac_gops"], 100.0)
    self.assertEqual(roofline["bw_prac_gbps"], 10.0)

  def test_memory_bound_layer_reports_memory_regime(self):
    # 1e7 FLOPs at 100 GOPS take 0.1 ms; 1.5e7 bytes at 10 GB/s take 1.5 ms.
    record = {
        "threads": 1,
        "real_time_ms": 2.0,
        "theoretical_flops": 1e7,
        "theoretical_bytes": 1.5e7,
    }

    roofline = layer.compute_subgraph_roofline(record, probe_ceilings())

    self.assertEqual(roofline["bound_regime"], "memory")
    self.assertAlmostEqual(roofline["t_roofline_ms"], 1.5, places=9)
    self.assertAlmostEqual(
        roofline["whole_subgraph_roofline_pct"], 75.0, places=6
    )

  def test_missing_ceilings_leave_roofline_fields_unset(self):
    record = {"threads": 1, "real_time_ms": 2.0, "theoretical_flops": 1e8}

    roofline = layer.compute_subgraph_roofline(record, {})

    self.assertEqual(
        roofline,
        {
            "c_prac_gops": 0.0,
            "bw_prac_gbps": 0.0,
            "t_roofline_ms": None,
            "whole_subgraph_roofline_pct": None,
            "bound_regime": None,
            "delta_t_compose_ms": None,
        },
    )

  def test_preset_scoped_kernel_timing_wins_over_family_timing(self):
    record = {
        "real_time_ms": 2.0,
        "family": "BM_RmsNorm",
        "layer": "RmsNorm",
        "preset": "Tiny",
    }

    roofline = layer.compute_subgraph_roofline(
        record,
        probe_ceilings(),
        kernel_timings={"Tiny/BM_RmsNorm": 1.5, "BM_RmsNorm": 0.25},
    )

    self.assertEqual(roofline["kernel_sum_ms"], 1.5)
    self.assertEqual(roofline["delta_t_compose_ms"], 0.5)

  def test_layer_name_kernel_timing_is_used_as_last_resort(self):
    record = {
        "real_time_ms": 2.0,
        "family": "BM_RmsNorm",
        "layer": "RmsNorm",
        "preset": "Tiny",
    }

    roofline = layer.compute_subgraph_roofline(
        record, probe_ceilings(), kernel_timings={"RmsNorm": 0.25}
    )

    self.assertEqual(roofline["delta_t_compose_ms"], 1.75)

  def test_unmatched_kernel_timings_leave_compose_delta_unset(self):
    record = {"real_time_ms": 2.0, "family": "BM_RmsNorm", "preset": "Tiny"}

    roofline = layer.compute_subgraph_roofline(
        record, probe_ceilings(), kernel_timings={"BM_RoPE": 0.25}
    )

    self.assertIsNone(roofline["delta_t_compose_ms"])
    self.assertNotIn("kernel_sum_ms", roofline)

  def test_repo_probe_roofline_of_measured_gated_mlp_prefill(self):
    ceilings = layer.load_hardware_ceilings(SAPPHIRE_RAPIDS_CEILINGS)
    record = layer.parse_benchmark_entry({
        "name": "BM_GatedMlp_GeGLU/Gemma4-E2B/B:1/S:1024/Threads:28/real_time",
        "real_time": 23.586,
        "cpu_time": 23.586,
        "time_unit": "ms",
        "iterations": 10,
        "GFLOP/s": 2459.14,
        "GB/s": 5.335,
        "ArithmeticIntensity": 460.95,
        "PeakMem_MB": 120.0,
    })

    roofline = layer.compute_subgraph_roofline(
        record,
        ceilings,
        dtype="fp32",
        kernel_timings={"Gemma4-E2B/BM_GatedMlp_GeGLU": 21.586},
    )

    self.assertEqual(roofline["bound_regime"], "compute")
    self.assertAlmostEqual(roofline["c_prac_gops"], 3531.07, places=2)
    self.assertAlmostEqual(
        roofline["whole_subgraph_roofline_pct"], 69.64, places=1
    )
    self.assertAlmostEqual(roofline["delta_t_compose_ms"], 2.0, places=3)


class ParseBenchmarkEntryTest(unittest.TestCase):

  def test_name_components_populate_identity_and_operating_point(self):
    record = layer.parse_benchmark_entry({
        "name": "BM_GatedMlp_GeGLU/Gemma4-E2B/B:2/S:1024/Threads:28/real_time",
    })

    self.assertEqual(record["family"], "BM_GatedMlp_GeGLU")
    self.assertEqual(record["layer"], "GatedMlp_GeGLU")
    self.assertEqual(record["preset"], "Gemma4-E2B")
    self.assertEqual(record["batch_size"], 2)
    self.assertEqual(record["seq_len"], 1024)
    self.assertEqual(record["threads"], 28)

  def test_counters_supply_operating_point_missing_from_name(self):
    record = layer.parse_benchmark_entry({
        "name": "BM_RmsNorm/Tiny",
        "BatchSize": 4,
        "SeqLenKV": 64,
        "Threads": 8,
    })

    self.assertEqual(record["batch_size"], 4)
    self.assertEqual(record["seq_len"], 64)
    self.assertEqual(record["threads"], 8)

  def test_decode_family_defaults_query_length_to_one(self):
    record = layer.parse_benchmark_entry({
        "name": "BM_StandardAttention_Decode/Gemma4-E2B/B:1/S:1024/Threads:28",
        "real_time": 0.1,
        "cpu_time": 0.1,
        "time_unit": "ms",
        "iterations": 100,
    })

    self.assertEqual(record["seq_len"], 1024)
    self.assertEqual(record["seq_len_q"], 1)

  def test_prefill_family_defaults_query_length_to_sequence_length(self):
    record = layer.parse_benchmark_entry({
        "name": "BM_StandardAttention_Prefill/Gemma4-E2B/B:1/S:1024/Threads:2",
    })

    self.assertEqual(record["seq_len_q"], 1024)

  def test_reported_query_length_overrides_family_default(self):
    record = layer.parse_benchmark_entry({
        "name": "BM_StandardAttention_Prefill/Gemma4-E2B/B:1/S:1024/Threads:2",
        "SeqLenQ": 16.0,
    })

    self.assertEqual(record["seq_len_q"], 16)

  def test_time_units_are_normalized_to_milliseconds(self):
    cases = (
        ("ns", 2500.0, 0.0025),
        ("us", 2500.0, 2.5),
        ("ms", 2500.0, 2500.0),
        ("s", 2.5, 2500.0),
        ("unknown_unit", 2500.0, 2500.0),
    )
    for time_unit, real_time, expected_ms in cases:
      with self.subTest(time_unit=time_unit):
        record = layer.parse_benchmark_entry({
            "name": "BM_RmsNorm/Tiny",
            "real_time": real_time,
            "cpu_time": real_time,
            "time_unit": time_unit,
        })

        self.assertAlmostEqual(record["real_time_ms"], expected_ms, places=9)
        self.assertAlmostEqual(record["cpu_time_ms"], expected_ms, places=9)

  def test_throughput_counters_fall_back_to_unscaled_keys(self):
    cases = (
        ("giga_flops", {"GFLOP/s": 7.0, "FLOPS": 2e9}, "gflops", 7.0),
        ("flops", {"FLOPS": 2e9, "FLOP/s": 3e9}, "gflops", 2.0),
        ("flop_per_s", {"FLOP/s": 3e9}, "gflops", 3.0),
        ("no_flops", {}, "gflops", 0.0),
        ("giga_bytes", {"GB/s": 9.0, "Bytes/s": 4e9}, "gbps", 9.0),
        ("bytes_per_s", {"Bytes/s": 4e9, "B/s": 5e9}, "gbps", 4.0),
        ("b_per_s", {"B/s": 5e9}, "gbps", 5.0),
    )
    for name, counters, field, expected in cases:
      with self.subTest(case=name):
        record = layer.parse_benchmark_entry({"name": "BM_RmsNorm", **counters})

        self.assertEqual(record[field], expected)

  def test_theoretical_work_is_back_computed_from_rate_and_time(self):
    # 2 GFLOP/s and 4 GB/s sustained for 0.5 s.
    record = layer.parse_benchmark_entry({
        "name": "BM_RmsNorm/Tiny",
        "real_time": 500.0,
        "time_unit": "ms",
        "GFLOP/s": 2.0,
        "GB/s": 4.0,
    })

    self.assertEqual(record["theoretical_flops"], 1e9)
    self.assertEqual(record["theoretical_bytes"], 2e9)

  def test_reported_theoretical_work_is_kept(self):
    record = layer.parse_benchmark_entry({
        "name": "BM_RmsNorm/Tiny",
        "real_time": 500.0,
        "GFLOP/s": 2.0,
        "GB/s": 4.0,
        "Theoretical_FLOPs": 123.0,
        "Theoretical_Bytes": 456.0,
    })

    self.assertEqual(record["theoretical_flops"], 123.0)
    self.assertEqual(record["theoretical_bytes"], 456.0)

  def test_workspace_peak_falls_back_to_peak_memory_megabytes(self):
    cases = (
        (
            "peak_bytes",
            {"Workspace_Peak_Bytes": 1024.0, "PeakMem_MB": 2.0},
            1024.0,
        ),
        (
            "legacy_peak_bytes",
            {"WorkspacePeakBytes": 2048.0, "PeakMem_MB": 2.0},
            2048.0,
        ),
        ("peak_megabytes", {"PeakMem_MB": 2.0}, 2097152.0),
    )
    for name, counters, expected_bytes in cases:
      with self.subTest(case=name):
        record = layer.parse_benchmark_entry({"name": "BM_RmsNorm", **counters})

        self.assertEqual(record["workspace_peak_bytes"], expected_bytes)

  def test_parity_counters_are_reported_when_binary_compared_tensors(self):
    record = layer.parse_benchmark_entry({
        "name": "BM_RmsNorm/Tiny",
        "Parity_Passed": 1.0,
        "Parity_MaxAbsErr": 0.001,
        "Parity_MaxRelErr": 0.002,
        "Parity_CosineSim": 0.999,
        "Parity_L2RelErr": 0.003,
        "Parity_Mismatches": 3.0,
    })

    self.assertIs(record["parity_passed"], True)
    self.assertEqual(record["parity_max_abs_err"], 0.001)
    self.assertEqual(record["parity_max_rel_err"], 0.002)
    self.assertEqual(record["parity_cosine_sim"], 0.999)
    self.assertEqual(record["parity_l2_rel_err"], 0.003)
    self.assertEqual(record["parity_mismatches"], 3)

  def test_failed_parity_counter_is_reported_as_false(self):
    record = layer.parse_benchmark_entry({
        "name": "BM_RmsNorm/Tiny",
        "Parity_Passed": 0.0,
    })

    self.assertIs(record["parity_passed"], False)

  def test_parity_fields_are_none_without_parity_counters(self):
    record = layer.parse_benchmark_entry({"name": "BM_RmsNorm/Tiny"})

    self.assertIsNone(record["parity_passed"])
    self.assertIsNone(record["parity_max_abs_err"])
    self.assertIsNone(record["parity_mismatches"])

  def test_benchmark_error_is_propagated(self):
    record = layer.parse_benchmark_entry({
        "name": "BM_RmsNorm/Tiny",
        "error_occurred": True,
        "error_message": "boom",
    })

    self.assertIs(record["error_occurred"], True)
    self.assertEqual(record["error_message"], "boom")


class ResolveSweepPointsTest(unittest.TestCase):

  def test_cli_override_is_used_and_registered_in_suite(self):
    suite = {"batch_sizes": [1, 4], "seq_lengths": [128]}

    batch_sizes, seq_lengths, _ = layer.resolve_sweep_points(
        suite, batch_size_arg=2, seq_len_arg=64
    )

    self.assertEqual(batch_sizes, [2])
    self.assertEqual(seq_lengths, [64])
    self.assertEqual(suite["batch_sizes"], [1, 2, 4])
    self.assertEqual(suite["seq_lengths"], [64, 128])

  def test_suite_values_are_swept_without_overrides(self):
    suite = {"batch_sizes": [4, 1], "seq_lengths": [2048, 64]}

    batch_sizes, seq_lengths, _ = layer.resolve_sweep_points(suite)

    self.assertEqual(batch_sizes, [4, 1])
    self.assertEqual(seq_lengths, [2048, 64])

  def test_repeated_suite_values_are_swept_once(self):
    suite = {"batch_sizes": [2, 1, 2], "seq_lengths": [64, 64]}

    batch_sizes, seq_lengths, _ = layer.resolve_sweep_points(suite)

    self.assertEqual(batch_sizes, [2, 1])
    self.assertEqual(seq_lengths, [64])

  def test_empty_suite_gets_default_operating_point(self):
    suite = {}

    points = layer.resolve_sweep_points(suite)

    self.assertEqual(points, ([1], [1024], 32))
    self.assertEqual(
        suite,
        {"batch_sizes": [1], "seq_lengths": [1024], "thread_counts": [32]},
    )

  def test_threads_default_to_machine_profile_physical_cores(self):
    suite = {
        "thread_counts": [1, 4, 8],
        "machine_profile": {"physical_cores_per_domain": 24},
    }

    _, _, threads = layer.resolve_sweep_points(suite)

    self.assertEqual(threads, 24)
    self.assertEqual(suite["thread_counts"], [1, 4, 8, 24])

  def test_threads_default_to_last_suite_thread_count(self):
    suite = {"thread_counts": [1, 4, 8]}

    _, _, threads = layer.resolve_sweep_points(suite)

    self.assertEqual(threads, 8)

  def test_threads_override_beats_machine_profile(self):
    suite = {"machine_profile": {"physical_cores_per_domain": 24}}

    _, _, threads = layer.resolve_sweep_points(suite, threads_arg=6)

    self.assertEqual(threads, 6)
    self.assertEqual(suite["thread_counts"], [6])


class DefaultBinaryTest(TempDirTestCase):

  def test_backend_specific_variable_takes_precedence(self):
    env = {
        "AEB_LAYER_BIN_YNNPACK": "/custom/layer_bench_ynn",
        "AEB_LAYER_BIN": "/custom/layer_bench",
    }

    self.assertEqual(
        layer.default_binary("ynnpack", env), "/custom/layer_bench_ynn"
    )

  def test_generic_variable_applies_to_other_backends(self):
    env = {
        "AEB_LAYER_BIN_YNNPACK": "/custom/layer_bench_ynn",
        "AEB_LAYER_BIN": "/custom/layer_bench",
    }

    self.assertEqual(
        layer.default_binary("xnnpack", env), "/custom/layer_bench"
    )

  def test_unconfigured_binary_falls_back_to_backend_install_path(self):
    install_dir = self.tmp / "install"

    resolved = layer.default_binary(
        "xnnpack", {"PATH": str(self.tmp)}, install_dir
    )

    self.assertEqual(resolved, str(install_dir / "layer_bench_xnnpack"))

  def test_installed_backend_binary_outranks_path(self):
    install_dir = self.tmp / "install"
    path_dir = self.tmp / "path"
    installed = write_executable(install_dir / "layer_bench_ynnpack")
    write_executable(path_dir / "layer_bench_ynnpack")

    resolved = layer.default_binary(
        "ynnpack", {"PATH": str(path_dir)}, install_dir
    )

    self.assertEqual(resolved, str(installed))

  def test_installed_generic_binary_serves_any_backend(self):
    install_dir = self.tmp / "install"
    installed = write_executable(install_dir / "layer_bench")

    resolved = layer.default_binary("xnnpack", {"PATH": ""}, install_dir)

    self.assertEqual(resolved, str(installed))

  def test_backend_binary_is_discovered_on_path(self):
    path_dir = self.tmp / "path"
    binary = write_executable(path_dir / "layer_bench_ynnpack")

    resolved = layer.default_binary(
        "ynnpack", {"PATH": str(path_dir)}, self.tmp / "install"
    )

    self.assertEqual(resolved, str(binary))

  def test_generic_binary_on_path_serves_any_backend(self):
    path_dir = self.tmp / "path"
    binary = write_executable(path_dir / "layer_bench")

    resolved = layer.default_binary(
        "xnnpack", {"PATH": str(path_dir)}, self.tmp / "install"
    )

    self.assertEqual(resolved, str(binary))

  def test_process_environment_is_consulted_by_default(self):
    variables = {"AEB_LAYER_BIN_XNNPACK": "/process/layer_bench"}

    with mock.patch.dict(os.environ, variables):
      resolved = layer.default_binary("xnnpack")

    self.assertEqual(resolved, "/process/layer_bench")


class RunnerHooksTest(unittest.TestCase):

  def test_default_hooks_read_the_process_environment(self):
    hooks = layer.RunnerHooks()

    self.assertIs(hooks.environ, os.environ)


def block_record(
    preset: str, real_time_ms: float, **fields: Any
) -> dict[str, Any]:
  """Returns a decode TransformerBlock layer record at B=1, S=512, T=16."""
  return {
      "family": "BM_TransformerBlock_Decode",
      "preset": preset,
      "batch_size": 1,
      "seq_len": 512,
      "threads": 16,
      "real_time_ms": real_time_ms,
      "error_occurred": False,
      **fields,
  }


class EstimateModelBlocksLatencyTest(unittest.TestCase):

  def test_fully_measured_model_sums_count_weighted_block_latencies(self):
    weights = {
        "Tiny": [
            {"layer_id": "block.Tiny.Local", "preset": "Tiny", "count": 2},
            {
                "layer_id": "block.Tiny.Global",
                "preset": "Tiny-Global",
                "count": 1,
                "approximated_features": ["k_eq_v"],
            },
        ],
    }
    layers = [block_record("Tiny", 1.5), block_record("Tiny-Global", 4.0)]

    estimates = layer.estimate_model_blocks_latency(layers, weights)

    # 2 x 1.5 ms + 1 x 4.0 ms.
    self.assertEqual(
        estimates,
        [{
            "model": "Tiny",
            "mode": "decode",
            "batch_size": 1,
            "seq_len": 512,
            "threads": 16,
            "summed_blocks_ms": 7.0,
            "covered_layers": 3,
            "total_layers": 3,
            "complete": True,
            "missing_presets": [],
            "approximated_presets": {"Tiny-Global": ["k_eq_v"]},
        }],
    )

  def test_unsupported_block_makes_estimate_incomplete(self):
    weights = {
        "Hybrid": [
            {"layer_id": "block.Hybrid.Full", "preset": "Hybrid", "count": 2},
            {
                "layer_id": "block.Hybrid.Linear",
                "preset": None,
                "count": 5,
                "unsupported": True,
            },
        ],
    }

    estimates = layer.estimate_model_blocks_latency(
        [block_record("Hybrid", 3.0)], weights
    )

    self.assertEqual(len(estimates), 1)
    self.assertFalse(estimates[0]["complete"])
    self.assertEqual(estimates[0]["covered_layers"], 2)
    self.assertEqual(estimates[0]["total_layers"], 7)
    self.assertEqual(estimates[0]["summed_blocks_ms"], 6.0)
    self.assertEqual(estimates[0]["missing_presets"], ["block.Hybrid.Linear"])

  def test_unmeasured_preset_is_listed_as_missing(self):
    weights = {
        "Tiny": [
            {"layer_id": "block.Tiny.Local", "preset": "Tiny", "count": 2},
            {
                "layer_id": "block.Tiny.Global",
                "preset": "Tiny-Global",
                "count": 1,
            },
        ],
    }

    estimates = layer.estimate_model_blocks_latency(
        [block_record("Tiny", 1.5)], weights
    )

    self.assertFalse(estimates[0]["complete"])
    self.assertEqual(estimates[0]["missing_presets"], ["Tiny-Global"])

  def test_prefill_and_decode_points_are_estimated_separately(self):
    weights = {
        "Tiny": [{"layer_id": "block.Tiny", "preset": "Tiny", "count": 2}],
    }
    layers = [
        block_record("Tiny", 1.5),
        block_record("Tiny", 9.0, family="BM_TransformerBlock_Prefill"),
    ]

    estimates = layer.estimate_model_blocks_latency(layers, weights)

    self.assertEqual(
        [(est["mode"], est["summed_blocks_ms"]) for est in estimates],
        [("decode", 3.0), ("prefill", 18.0)],
    )

  def test_failed_and_non_block_benchmarks_are_ignored(self):
    weights = {
        "Tiny": [{"layer_id": "block.Tiny", "preset": "Tiny", "count": 2}],
    }
    layers = [
        block_record("Tiny", 1.5, error_occurred=True),
        block_record("Tiny", 1.5, family="BM_RmsNorm"),
    ]

    self.assertEqual(layer.estimate_model_blocks_latency(layers, weights), [])

  def test_model_without_any_measured_block_is_omitted(self):
    weights = {
        "Tiny": [{"layer_id": "block.Tiny", "preset": "Tiny", "count": 2}],
    }

    estimates = layer.estimate_model_blocks_latency(
        [block_record("Other", 1.5)], weights
    )

    self.assertEqual(estimates, [])

  def test_no_layer_records_yield_no_estimates(self):
    weights = {
        "Tiny": [{"layer_id": "block.Tiny", "preset": "Tiny", "count": 2}],
    }

    self.assertEqual(layer.estimate_model_blocks_latency([], weights), [])

  def test_no_model_weights_yield_no_estimates(self):
    layers = [block_record("Tiny", 1.5)]

    self.assertEqual(layer.estimate_model_blocks_latency(layers, {}), [])


class FilterFamiliesForModeTest(unittest.TestCase):

  def test_families_are_kept_when_they_run_in_the_mode(self):
    families = (
        "BM_RmsNorm",
        "BM_TransformerBlock_Prefill",
        "BM_TransformerBlock_Decode",
        "BM_Custom",
    )
    cases = (
        ("prefill", ["BM_RmsNorm", "BM_TransformerBlock_Prefill"]),
        ("decode", ["BM_TransformerBlock_Decode"]),
        (
            "both",
            [
                "BM_RmsNorm",
                "BM_TransformerBlock_Prefill",
                "BM_TransformerBlock_Decode",
                "BM_Custom",
            ],
        ),
    )
    for mode, expected_families in cases:
      with self.subTest(mode=mode):
        self.assertEqual(
            catalog.filter_families_for_mode(families, mode), expected_families
        )


class ValidateCatalogTest(unittest.TestCase):

  def test_minimal_catalog_is_valid(self):
    self.assertIsNone(catalog.validate_catalog(tiny_catalog()))

  def test_null_rotary_dim_defaults_to_head_dim(self):
    document = with_value(
        tiny_catalog(), ("composite_blocks", 0, "params", "rotary_dim"), None
    )

    self.assertIsNone(catalog.validate_catalog(document))

  def test_block_without_attention_heads_skips_attention_checks(self):
    document = with_value(
        tiny_catalog(), ("composite_blocks", 1, "params", "num_heads"), None
    )

    self.assertIsNone(catalog.validate_catalog(document))

  def test_invalid_top_level_fields_are_rejected(self):
    cases = (
        ("wrong_schema", ("schema",), "aeb.layer.catalog/2", "expected schema"),
        ("empty_catalog_id", ("catalog_id",), "", "non-empty 'catalog_id'"),
        ("table_not_a_list", ("mlp_layers",), {}, "'mlp_layers' must be a"),
        ("empty_models", ("models",), {}, "'models' must be a non-empty"),
    )
    for name, path, value, message in cases:
      with self.subTest(case=name):
        document = with_value(tiny_catalog(), path, value)

        with self.assertRaisesRegex(ValueError, message):
          catalog.validate_catalog(document)

  def test_invalid_layer_entries_are_rejected(self):
    block = ("composite_blocks", 0)
    cases = (
        ("entry_not_an_object", block, "block", "entry must be an object"),
        ("empty_id", (*block, "id"), "", "missing non-empty 'id'"),
        (
            "duplicate_id",
            (*block, "id"),
            "block.Tiny.Global",
            "duplicate layer id 'block.Tiny.Global'",
        ),
        (
            "unsupported_layer_type",
            (*block, "layer_type"),
            "InvalidBlockType",
            "unsupported layer_type 'InvalidBlockType'",
        ),
        ("params_not_an_object", (*block, "params"), [], "missing 'params'"),
        ("slash_in_preset", (*block, "preset_name"), "a/b", "must not contain"),
        (
            "numeric_preset",
            (*block, "preset_name"),
            7,
            "preset_name must be a string",
        ),
        (
            "duplicate_preset",
            (*block, "preset_name"),
            "Tiny-Global",
            "duplicate preset_name 'Tiny-Global'",
        ),
    )
    for name, path, value, message in cases:
      with self.subTest(case=name):
        document = with_value(tiny_catalog(), path, value)

        with self.assertRaisesRegex(ValueError, message):
          catalog.validate_catalog(document)

  def test_invalid_block_dimensions_are_rejected(self):
    params = ("composite_blocks", 0, "params")
    cases = (
        ("zero_embed_dim", "embed_dim", 0, "requires positive embed_dim"),
        ("bool_hidden_dim", "hidden_dim", True, "requires positive hidden_dim"),
        ("nan_eps", "eps", float("nan"), "requires positive finite eps"),
        (
            "null_kv_heads",
            "num_kv_heads",
            None,
            "requires positive num_heads, num_kv_heads, and head_dim",
        ),
        ("odd_head_dim", "head_dim", 15, r"head_dim \(15\) must be even"),
        (
            "indivisible_heads",
            "num_kv_heads",
            3,
            r"num_heads \(4\) not divisible by num_kv_heads \(3\)",
        ),
        (
            "odd_rotary_dim",
            "rotary_dim",
            3,
            r"rotary_dim \(3\) must be even in 2..16",
        ),
        ("zero_rotary_dim", "rotary_dim", 0, r"rotary_dim \(0\) must be even"),
        ("wide_rotary_dim", "rotary_dim", 18, r"rotary_dim \(18\) must be"),
    )
    for name, key, value, message in cases:
      with self.subTest(case=name):
        document = with_value(tiny_catalog(), (*params, key), value)

        with self.assertRaisesRegex(ValueError, message):
          catalog.validate_catalog(document)

  def test_invalid_model_entries_are_rejected(self):
    model = ("models", "Tiny")
    item = (*model, "block_composition", 0)
    cases = (
        ("model_not_an_object", model, ["Tiny"], "Tiny must be an object"),
        ("zero_total_layers", (*model, "total_layers"), 0, "total_layers"),
        ("string_embed_dim", (*model, "embed_dim"), "64", "invalid embed_dim"),
        ("infinite_eps", (*model, "eps"), float("inf"), "Tiny invalid eps"),
        (
            "empty_composition",
            (*model, "block_composition"),
            [],
            "requires non-empty block_composition",
        ),
        (
            "composition_item_not_an_object",
            item,
            "block.Tiny.Local",
            "block_composition entry must be an object",
        ),
        (
            "unknown_block",
            (*item, "layer_id"),
            "block.Missing",
            "references unknown block 'block.Missing'",
        ),
        ("zero_count", (*item, "count"), 0, "has invalid count"),
        (
            "embed_dim_mismatch",
            (*model, "embed_dim"),
            32,
            r"embed_dim \(32\) mismatches block.Tiny.Local \(64\)",
        ),
        (
            "wrong_layer_sum",
            (*model, "total_layers"),
            4,
            r"block_composition sum \(3\) != total_layers \(4\)",
        ),
    )
    for name, path, value, message in cases:
      with self.subTest(case=name):
        document = with_value(tiny_catalog(), path, value)

        with self.assertRaisesRegex(ValueError, message):
          catalog.validate_catalog(document)

  def test_error_message_starts_with_source(self):
    document = with_value(tiny_catalog(), ("catalog_id",), "")

    with self.assertRaisesRegex(ValueError, "^tiny.json: "):
      catalog.validate_catalog(document, source="tiny.json")

  def test_non_object_catalog_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "catalog must be an object"):
      catalog.validate_catalog(["not", "a", "catalog"])


class ValidateHyperparamsTest(TempDirTestCase):

  def test_minimal_registry_is_valid(self):
    self.assertIsNone(catalog.validate_hyperparams(tiny_hyperparams()))

  def test_wrong_schema_is_rejected(self):
    document = with_value(tiny_hyperparams(), ("schema",), "wrong/1")

    with self.assertRaisesRegex(ValueError, "^hp.json: expected schema"):
      catalog.validate_hyperparams(document, source="hp.json")

  def test_non_object_registry_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "hyperparams must be an object"):
      catalog.validate_hyperparams(["regimes"])

  def test_invalid_regimes_are_rejected(self):
    regime = ("regimes", "short")
    cases = (
        ("empty_table", ("regimes",), {}, "missing non-empty 'regimes'"),
        ("not_an_object", regime, "both", "regime short must be an object"),
        ("invalid_mode", (*regime, "mode"), "sideways", "short invalid mode"),
        (
            "empty_batch_sizes",
            (*regime, "batch_sizes"),
            [],
            r"regimes.short.batch_sizes must be a non-empty list",
        ),
        (
            "zero_seq_length",
            (*regime, "seq_lengths"),
            [0],
            r"regimes.short.seq_lengths contains non-positive integer: 0",
        ),
    )
    for name, path, value, message in cases:
      with self.subTest(case=name):
        document = with_value(tiny_hyperparams(), path, value)

        with self.assertRaisesRegex(ValueError, message):
          catalog.validate_hyperparams(document)

  def test_invalid_quant_schemes_are_rejected(self):
    scheme = ("quant_schemes", "fp32")
    gate = (*scheme, "quality_gate")
    cases = (
        ("empty_table", ("quant_schemes",), {}, "non-empty 'quant_schemes'"),
        ("not_an_object", scheme, "fp32", "fp32 must be an object"),
        ("null_quality_gate", gate, None, "fp32 missing quality_gate"),
        ("negative_rtol", (*gate, "rtol"), -0.1, "invalid rtol/atol"),
        ("string_atol", (*gate, "atol"), "1e-5", "invalid rtol/atol"),
        ("large_min_cosine", (*gate, "min_cosine"), 1.5, "invalid min_cosine"),
        (
            "negative_max_kl",
            (*gate, "max_kl_divergence"),
            -1.0,
            "invalid max_kl_divergence",
        ),
        (
            "nan_max_l2",
            (*gate, "max_l2_rel_err"),
            float("nan"),
            "invalid max_l2_rel_err",
        ),
        ("empty_weight_dtype", (*scheme, "weight_dtype"), "", "weight_dtype"),
    )
    for name, path, value, message in cases:
      with self.subTest(case=name):
        document = with_value(tiny_hyperparams(), path, value)

        with self.assertRaisesRegex(ValueError, message):
          catalog.validate_hyperparams(document)

  def test_invalid_cliff_sweeps_are_rejected(self):
    sweep = ("cliff_sweeps", "none")
    cases = (
        ("empty_table", ("cliff_sweeps",), {}, "non-empty 'cliff_sweeps'"),
        ("not_an_object", sweep, [0], "cliff_sweep none must be an object"),
        ("empty_deltas", (*sweep, "deltas"), [], "requires deltas list"),
        ("boolean_delta", (*sweep, "deltas"), [True], "non-int delta True"),
        (
            "numeric_target_dim",
            (*sweep, "target_dims"),
            [8],
            "target_dims must be a list of strings",
        ),
    )
    for name, path, value, message in cases:
      with self.subTest(case=name):
        document = with_value(tiny_hyperparams(), path, value)

        with self.assertRaisesRegex(ValueError, message):
          catalog.validate_hyperparams(document)

  def test_invalid_machine_profiles_are_rejected(self):
    machine = ("machine_threading", "box")
    cases = (
        ("empty_table", ("machine_threading",), {}, "'machine_threading'"),
        ("not_an_object", machine, [1, 4], "machine box must be an object"),
        (
            "zero_thread_count",
            (*machine, "thread_counts"),
            [0],
            r"machine_threading.box.thread_counts contains non-positive",
        ),
        (
            "zero_physical_cores",
            (*machine, "physical_cores_per_domain"),
            0,
            "invalid physical_cores_per_domain",
        ),
        (
            "numeric_ceilings_file",
            (*machine, "ceilings_file"),
            5,
            "ceilings_file must be a string",
        ),
    )
    for name, path, value, message in cases:
      with self.subTest(case=name):
        document = with_value(tiny_hyperparams(), path, value)

        with self.assertRaisesRegex(ValueError, message):
          catalog.validate_hyperparams(document)

  def test_missing_ceilings_file_is_rejected_when_base_dir_is_given(self):
    document = with_value(
        tiny_hyperparams(),
        ("machine_threading", "box", "ceilings_file"),
        "missing.json",
    )

    with self.assertRaisesRegex(ValueError, "ceilings_file not found"):
      catalog.validate_hyperparams(document, base_dir=self.tmp)

  def test_existing_ceilings_file_is_accepted(self):
    (self.tmp / "ceilings.json").write_text("{}", encoding="utf-8")
    document = with_value(
        tiny_hyperparams(),
        ("machine_threading", "box", "ceilings_file"),
        "ceilings.json",
    )

    self.assertIsNone(catalog.validate_hyperparams(document, base_dir=self.tmp))

  def test_ceilings_file_is_not_checked_without_base_dir(self):
    document = with_value(
        tiny_hyperparams(),
        ("machine_threading", "box", "ceilings_file"),
        "missing.json",
    )

    self.assertIsNone(catalog.validate_hyperparams(document))


class RepoLayerConfigsTest(TempDirTestCase):

  def test_checked_in_catalog_registry_and_suites_are_valid(self):
    counts = catalog.validate_repo_layer_configs(LAYER_ROOT)

    self.assertEqual(counts["catalogs"], 1)
    self.assertEqual(counts["hyperparams"], 1)
    self.assertGreaterEqual(counts["suites"], 4)

  def test_checked_in_catalog_defines_ten_models(self):
    document = catalog.load_json_object(REPO_CATALOG)

    self.assertEqual(len(document["models"]), 10)

  def test_layer_root_without_catalog_is_rejected(self):
    (self.tmp / "suites").mkdir()

    with self.assertRaisesRegex(ValueError, "^No layer catalog JSON found in "):
      catalog.validate_repo_layer_configs(self.tmp)

  def test_catalog_file_with_unrecognized_schema_is_rejected(self):
    write_tiny_registry(self.tmp)
    (self.tmp / "catalog" / "notes.json").write_text(
        json.dumps({"schema": "aeb.notes/1"}), encoding="utf-8"
    )

    with self.assertRaisesRegex(
        ValueError, "Unrecognized schema 'aeb.notes/1' in .*notes.json"
    ):
      catalog.validate_repo_layer_configs(self.tmp)

  def test_suite_resolving_to_no_presets_is_rejected(self):
    suites_dir = write_tiny_registry(self.tmp)
    (suites_dir / "empty.json").write_text(
        json.dumps({"name": "empty"}), encoding="utf-8"
    )

    with self.assertRaisesRegex(
        ValueError, "empty.json missing presets or families after resolve"
    ):
      catalog.validate_repo_layer_configs(self.tmp)

  def test_suite_naming_a_missing_ceilings_file_is_rejected(self):
    write_tiny_suite(
        self.tmp, regime="short", ceilings_file="absent_ceilings.json"
    )

    with self.assertRaisesRegex(
        ValueError, "ceilings_file does not exist: .*absent_ceilings.json"
    ):
      catalog.validate_repo_layer_configs(self.tmp)

  def test_suite_ceilings_file_is_resolved_against_the_suite_directory(self):
    write_tiny_suite(self.tmp, regime="short", ceilings_file="../ceilings.json")
    (self.tmp / "ceilings.json").write_text("{}", encoding="utf-8")

    counts = catalog.validate_repo_layer_configs(self.tmp)

    self.assertEqual(counts, {"catalogs": 1, "hyperparams": 1, "suites": 1})


def small_block_params(**overrides: Any) -> dict[str, Any]:
  """Returns a 2-head attention block small enough to count by hand."""
  return {
      "embed_dim": 4,
      "hidden_dim": 8,
      "num_heads": 2,
      "num_kv_heads": 1,
      "head_dim": 2,
      **overrides,
  }


class BlockRooflineWorkTest(unittest.TestCase):

  def test_prefill_work_of_plain_block(self):
    work = catalog.estimate_block_roofline_work(
        small_block_params(), batch_size=1, seq_len_q=2, seq_len_kv=2
    )

    # Weights: Q 4x4 + K,V 2x(2x4) + O 4x4 + MLP 3x(8x4) = 144 elements.
    # FLOPs: 2 x 2 tokens x 144 = 576 GEMM + 4 x 2 heads x 2 x 2 x 2 = 64 SDPA.
    self.assertEqual(work["theoretical_flops"], 640.0)
    # Bytes: 144 x 4 weights + 8 x 4 norms + 64 I/O + 32 KV state + 16 mask.
    self.assertEqual(work["min_boundary_bytes"], 720.0)
    self.assertAlmostEqual(work["arithmetic_intensity"], 0.8888889, places=6)

  def test_decode_work_has_single_token_and_no_mask(self):
    work = catalog.estimate_block_roofline_work(
        small_block_params(), batch_size=1, seq_len_q=1, seq_len_kv=2
    )

    # FLOPs: 2 x 1 token x 144 = 288 GEMM + 4 x 2 heads x 1 x 2 x 2 = 32 SDPA.
    self.assertEqual(work["theoretical_flops"], 320.0)
    # Bytes: 608 weights and norms + 32 I/O + 32 KV state, no causal mask.
    self.assertEqual(work["min_boundary_bytes"], 672.0)

  def test_quantized_weights_shrink_boundary_bytes_only(self):
    work = catalog.estimate_block_roofline_work(
        small_block_params(),
        batch_size=1,
        seq_len_q=2,
        seq_len_kv=2,
        bits_per_weight=8.0,
    )

    self.assertEqual(work["theoretical_flops"], 640.0)
    # Bytes: 144 x 1 weights + 32 norms + 64 I/O + 32 KV state + 16 mask.
    self.assertEqual(work["min_boundary_bytes"], 288.0)

  def test_optional_block_features_change_weight_and_norm_counts(self):
    params = small_block_params(
        gated_q_proj=True, k_eq_v=True, qk_norm=True, post_norms=True, ple_dim=3
    )

    work = catalog.estimate_block_roofline_work(
        params, batch_size=1, seq_len_q=2, seq_len_kv=2
    )

    # Weights: gated Q 8x4 + shared K=V 2x4 + O 4x4 + MLP 96 + PLE 2x3x4 = 176.
    self.assertEqual(work["theoretical_flops"], 768.0)
    # Norms: 4x4 block + 2x2 QK + 4 PLE = 24 elements.
    # Bytes: 176 x 4 + 24 x 4 + 64 I/O + 32 KV state + 16 mask.
    self.assertEqual(work["min_boundary_bytes"], 912.0)

  def test_shared_kv_block_has_no_key_value_projections(self):
    work = catalog.estimate_block_roofline_work(
        small_block_params(shared_kv=True),
        batch_size=1,
        seq_len_q=2,
        seq_len_kv=2,
    )

    # Weights: Q 16 + O 16 + MLP 96 = 128; FLOPs: 2 x 2 x 128 + 64 SDPA.
    self.assertEqual(work["theoretical_flops"], 576.0)

  def test_sliding_window_does_not_clamp_prefill_attention(self):
    params = {
        "embed_dim": 1536,
        "hidden_dim": 6144,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 256,
        "sliding_window": 512,
        "qk_norm": True,
        "post_norms": True,
    }

    windowed = catalog.estimate_block_roofline_work(
        params, batch_size=1, seq_len_q=2048, seq_len_kv=2048
    )
    full = catalog.estimate_block_roofline_work(
        dict(params, sliding_window=None),
        batch_size=1,
        seq_len_q=2048,
        seq_len_kv=2048,
    )

    self.assertEqual(windowed["theoretical_flops"], full["theoretical_flops"])

  def test_sliding_window_clamps_decode_attention_and_kv_state(self):
    params = {
        "embed_dim": 1536,
        "hidden_dim": 6144,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 256,
        "sliding_window": 512,
        "qk_norm": True,
        "post_norms": True,
    }

    windowed = catalog.estimate_block_roofline_work(
        params, batch_size=1, seq_len_q=1, seq_len_kv=2048
    )
    full = catalog.estimate_block_roofline_work(
        dict(params, sliding_window=None),
        batch_size=1,
        seq_len_q=1,
        seq_len_kv=2048,
    )

    self.assertLess(windowed["theoretical_flops"], full["theoretical_flops"])
    self.assertLess(windowed["min_boundary_bytes"], full["min_boundary_bytes"])


def llama_config(**overrides: Any) -> dict[str, Any]:
  """Returns a two-layer HF config with 4 heads of 64 dims and GQA 2:1."""
  return {
      "model_type": "llama",
      "hidden_size": 256,
      "intermediate_size": 512,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      **overrides,
  }


def gemma4_wrapped_config() -> dict[str, Any]:
  """Returns a 10-layer Gemma 4 style config nested under `text_config`.

  Layers 5 and 10 are global, the last four share KV caches, and optional
  keys are explicit nulls, as in published multimodal configs.
  """
  local_layers = ["sliding_attention"] * 4
  return {
      "model_type": "gemma4",
      "vocab_size": None,
      "text_config": {
          "model_type": "gemma4_text",
          "hidden_size": 1536,
          "intermediate_size": 6144,
          "num_hidden_layers": 10,
          "num_attention_heads": 8,
          "num_key_value_heads": None,
          "num_global_key_value_heads": None,
          "head_dim": None,
          "global_head_dim": None,
          "sliding_window": 512,
          "hidden_activation": "gelu_pytorch_tanh",
          "hidden_size_per_layer_input": 256,
          "num_kv_shared_layers": 4,
          "vocab_size": None,
          "rms_norm_eps": None,
          "partial_rotary_factor": None,
          "rope_parameters": {
              "full_attention": {"partial_rotary_factor": 0.25},
              "sliding_attention": {"partial_rotary_factor": None},
          },
          "layer_types": [
              *local_layers,
              "full_attention",
              *local_layers,
              "full_attention",
          ],
      },
  }


class ScoutHfConfigTest(unittest.TestCase):

  def test_uniform_model_scouts_to_one_block_archetype(self):
    scouted = catalog.scout_hf_config(
        llama_config(), "MiniLlama", hf_repo="org/mini-llama"
    )

    block = scouted["composite_blocks"][0]
    self.assertEqual(len(scouted["composite_blocks"]), 1)
    self.assertEqual(block["id"], "block.MiniLlama.FullAttention")
    self.assertEqual(block["preset_name"], "MiniLlama")
    self.assertEqual(block["params"]["head_dim"], 64)
    self.assertEqual(block["params"]["rotary_dim"], 64)
    self.assertEqual(
        scouted["attention_layers"][0]["id"],
        "attn.M256_Hq4_Hkv2_D64_R64_Full",
    )
    self.assertEqual(scouted["mlp_layers"][0]["id"], "mlp.swiglu_M256_F512")
    self.assertEqual(
        scouted["models"]["MiniLlama"]["block_composition"],
        [{"layer_id": "block.MiniLlama.FullAttention", "count": 2}],
    )
    self.assertEqual(
        scouted["models"]["MiniLlama"]["hf_repo"], "org/mini-llama"
    )

  def test_gemma4_text_config_with_nulls_scouts_four_archetypes(self):
    scouted = catalog.scout_hf_config(
        gemma4_wrapped_config(), "Gemma4-Test10L", hf_repo="google/gemma-4-test"
    )

    self.assertIsNone(catalog.validate_catalog(scouted))
    self.assertEqual(
        [block["id"] for block in scouted["composite_blocks"]],
        [
            "block.Gemma4-Test10L.LocalProducer",
            "block.Gemma4-Test10L.GlobalProducer",
            "block.Gemma4-Test10L.LocalSharedKV",
            "block.Gemma4-Test10L.GlobalSharedKV",
        ],
    )
    self.assertIn("_QKNorm", scouted["attention_layers"][0]["id"])

  def test_gemma4_null_dimensions_fall_back_to_derived_defaults(self):
    scouted = catalog.scout_hf_config(gemma4_wrapped_config(), "Gemma4-Test10L")

    local_params = scouted["composite_blocks"][0]["params"]
    global_params = scouted["composite_blocks"][1]["params"]
    self.assertTrue(local_params["qk_norm"])
    self.assertTrue(local_params["post_norms"])
    self.assertEqual(local_params["num_kv_heads"], 8)
    # head_dim defaults to hidden_size / num_attention_heads = 1536 / 8.
    self.assertEqual(local_params["head_dim"], 192)
    self.assertEqual(local_params["rotary_dim"], 192)
    self.assertEqual(local_params["mlp_activation"], "GeGLU")
    self.assertEqual(local_params["ple_dim"], 256)
    # Only full-attention layers set a 0.25 partial rotary factor.
    self.assertEqual(global_params["rotary_dim"], 48)

  def test_local_and_global_layers_of_one_shape_share_attention_entry(self):
    hf_config = llama_config(
        sliding_window=None,
        layer_types=["sliding_attention", "full_attention"],
    )

    scouted = catalog.scout_hf_config(hf_config, "NoSwTest")

    self.assertIsNone(catalog.validate_catalog(scouted))
    self.assertEqual(len(scouted["composite_blocks"]), 2)
    self.assertEqual(len(scouted["attention_layers"]), 1)
    self.assertEqual(len(scouted["attention_layers"][0]["models"]), 2)

  def test_sliding_window_pattern_derives_local_and_global_layers(self):
    hf_config = llama_config(
        num_hidden_layers=6,
        sliding_window=512,
        sliding_window_pattern=3,
    )

    scouted = catalog.scout_hf_config(hf_config, "Pattern")

    self.assertEqual(
        scouted["models"]["Pattern"]["block_composition"],
        [
            {"layer_id": "block.Pattern.LocalSliding", "count": 4},
            {"layer_id": "block.Pattern.GlobalFull", "count": 2},
        ],
    )
    self.assertEqual(
        scouted["composite_blocks"][0]["params"]["sliding_window"], 512
    )

  def test_linear_attention_blocks_have_no_preset_shape_or_roofline(self):
    hf_config = {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "attn_output_gate": True,
            "hidden_act": "silu",
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        },
    }

    scouted = catalog.scout_hf_config(hf_config, "Qwen35-Test8L")

    self.assertIsNone(catalog.validate_catalog(scouted))
    linear_block, full_block = scouted["composite_blocks"]
    self.assertIsNone(linear_block["params"]["num_heads"])
    self.assertNotIn("roofline_ref_p1024", linear_block)
    self.assertIn("roofline_ref_p1024", full_block)
    self.assertEqual(
        [entry["layer_type"] for entry in scouted["attention_layers"]],
        ["GatedDeltaNet", "SelfAttention"],
    )
    self.assertIn("_GatedQ", scouted["attention_layers"][1]["id"])

  def test_partial_rotary_dim_is_rounded_to_nearest_even_in_range(self):
    cases = (
        ("even_product", 64, 0.25, 16),
        ("odd_product_rounds_down", 6, 0.5, 2),
        ("odd_product_five_rounds_down", 20, 0.25, 4),
        ("fraction_rounds_to_nearest_even", 10, 0.75, 8),
        ("product_below_two_clamps_up", 8, 0.125, 2),
        ("factor_above_one_clamps_to_head_dim", 64, 2.0, 64),
    )
    for name, head_dim, factor, expected_rotary_dim in cases:
      with self.subTest(case=name):
        hf_config = llama_config(
            head_dim=head_dim, partial_rotary_factor=factor
        )

        scouted = catalog.scout_hf_config(hf_config, "Rotary")

        self.assertEqual(
            scouted["composite_blocks"][0]["params"]["rotary_dim"],
            expected_rotary_dim,
        )

  def test_layer_specific_rotary_factor_wins_over_global_factors(self):
    hf_config = llama_config(
        partial_rotary_factor=1.0,
        rope_parameters={
            "partial_rotary_factor": 0.5,
            "full_attention": {"partial_rotary_factor": 0.25},
        },
    )

    scouted = catalog.scout_hf_config(hf_config, "Rotary")

    self.assertEqual(scouted["composite_blocks"][0]["params"]["rotary_dim"], 16)

  def test_rope_parameters_factor_wins_over_top_level_factor(self):
    hf_config = llama_config(
        partial_rotary_factor=1.0,
        rope_parameters={"partial_rotary_factor": 0.5},
    )

    scouted = catalog.scout_hf_config(hf_config, "Rotary")

    self.assertEqual(scouted["composite_blocks"][0]["params"]["rotary_dim"], 32)

  def test_invalid_partial_rotary_factor_is_rejected(self):
    for factor in (0.0, -1.0, float("nan"), "0.5"):
      with self.subTest(factor=factor):
        hf_config = llama_config(partial_rotary_factor=factor)

        with self.assertRaisesRegex(ValueError, "partial_rotary_factor must"):
          catalog.scout_hf_config(hf_config, "Rotary")

  def test_missing_required_dimension_is_rejected(self):
    hf_config = llama_config()
    del hf_config["hidden_size"]

    with self.assertRaisesRegex(
        ValueError, "requires a positive integer 'hidden_size', got None"
    ):
      catalog.scout_hf_config(hf_config, "Broken")

  def test_config_that_cannot_form_a_valid_catalog_is_rejected(self):
    hf_config = llama_config(head_dim=63)

    with self.assertRaisesRegex(ValueError, r"head_dim \(63\) must be even"):
      catalog.scout_hf_config(hf_config, "OddHead")

  def test_partial_rotary_factor_needs_head_dim_of_at_least_two(self):
    hf_config = llama_config(head_dim=1, partial_rotary_factor=0.5)

    with self.assertRaisesRegex(
        ValueError, "partial_rotary_factor requires head_dim >= 2, got 1"
    ):
      catalog.scout_hf_config(hf_config, "Rotary")

  def test_empty_model_name_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "model_name must be a non-empty"):
      catalog.scout_hf_config(llama_config(), "")

  def test_normalization_flags_follow_config_keys_then_model_family(self):
    cases = (
        ("plain_family_has_no_norms", {}, (False, False)),
        ("explicit_qk_norm", {"qk_norm": True}, (True, False)),
        ("legacy_qk_norm_key", {"attn_qk_norm": True}, (True, False)),
        ("explicit_post_norms", {"post_norms": True}, (False, True)),
        ("four_block_norms", {"num_block_norms": 4}, (False, True)),
        ("gemma_family", {"model_type": "gemma4_text"}, (True, True)),
        ("qwen_family", {"model_type": "qwen3_5_text"}, (True, False)),
        (
            "explicit_false_overrides_family",
            {
                "model_type": "gemma4_text",
                "qk_norm": False,
                "post_norms": False,
            },
            (False, False),
        ),
    )
    for name, overrides, expected_flags in cases:
      with self.subTest(case=name):
        scouted = catalog.scout_hf_config(llama_config(**overrides), "Norms")

        params = scouted["composite_blocks"][0]["params"]
        self.assertEqual(
            (params["qk_norm"], params["post_norms"]), expected_flags
        )

  def test_shared_key_value_projection_applies_to_global_layers_only(self):
    hf_config = llama_config(
        attention_k_eq_v=True,
        sliding_window=512,
        layer_types=["sliding_attention", "full_attention"],
    )

    scouted = catalog.scout_hf_config(hf_config, "KeqV")

    local_block, global_block = scouted["composite_blocks"]
    self.assertFalse(local_block["params"]["k_eq_v"])
    self.assertTrue(global_block["params"]["k_eq_v"])
    self.assertEqual(
        [entry["id"] for entry in scouted["attention_layers"]],
        [
            "attn.M256_Hq4_Hkv2_D64_R64_W512",
            "attn.M256_Hq4_Hkv2_D64_R64_Full_KeqV",
        ],
    )

  def test_vocabulary_and_tied_embeddings_prefer_text_config_values(self):
    cases = (
        ("defaults", llama_config(), (32000, True)),
        (
            "flat_config_values",
            llama_config(vocab_size=1000, tie_word_embeddings=False),
            (1000, False),
        ),
        (
            "wrapper_values_fill_in",
            {
                "vocab_size": 2000,
                "tie_word_embeddings": False,
                "text_config": llama_config(),
            },
            (2000, False),
        ),
        (
            "text_config_values_win",
            {
                "vocab_size": 2000,
                "tie_word_embeddings": False,
                "text_config": llama_config(
                    vocab_size=1000, tie_word_embeddings=True
                ),
            },
            (1000, True),
        ),
    )
    for name, hf_config, expected_values in cases:
      with self.subTest(case=name):
        scouted = catalog.scout_hf_config(hf_config, "Vocab")

        model = scouted["models"]["Vocab"]
        self.assertEqual(
            (model["vocab_size"], model["tie_word_embeddings"]),
            expected_values,
        )


class MergeScoutedIntoCatalogTest(unittest.TestCase):

  def test_merge_adds_scouted_model_blocks_and_layers(self):
    scouted = catalog.scout_hf_config(llama_config(), "MiniLlama")

    merged = catalog.merge_scouted_into_catalog(tiny_catalog(), scouted)

    self.assertEqual(sorted(merged["models"]), ["MiniLlama", "Tiny"])
    self.assertEqual(
        [block["id"] for block in merged["composite_blocks"]],
        [
            "block.Tiny.Local",
            "block.Tiny.Global",
            "block.MiniLlama.FullAttention",
        ],
    )
    self.assertEqual(len(merged["attention_layers"]), 1)
    self.assertEqual(len(merged["mlp_layers"]), 1)

  def test_merge_does_not_mutate_base_catalog(self):
    base = tiny_catalog()
    scouted = catalog.scout_hf_config(llama_config(), "MiniLlama")

    catalog.merge_scouted_into_catalog(base, scouted)

    self.assertEqual(base, tiny_catalog())

  def test_merging_same_scouted_model_twice_is_idempotent(self):
    scouted = catalog.scout_hf_config(llama_config(), "MiniLlama")
    merged_once = catalog.merge_scouted_into_catalog(tiny_catalog(), scouted)

    merged_twice = catalog.merge_scouted_into_catalog(merged_once, scouted)

    self.assertEqual(merged_twice, merged_once)

  def test_second_model_with_same_shape_shares_attention_entry(self):
    first = catalog.scout_hf_config(llama_config(), "MiniLlama")
    second = catalog.scout_hf_config(llama_config(), "MiniLlama2")
    merged_first = catalog.merge_scouted_into_catalog(tiny_catalog(), first)

    merged = catalog.merge_scouted_into_catalog(merged_first, second)

    self.assertEqual(len(merged["attention_layers"]), 1)
    self.assertEqual(
        [item["model"] for item in merged["attention_layers"][0]["models"]],
        ["MiniLlama", "MiniLlama2"],
    )

  def test_scouted_model_merges_into_checked_in_catalog(self):
    base = catalog.load_json_object(REPO_CATALOG)
    scouted = catalog.scout_hf_config(llama_config(), "MiniLlama")

    merged = catalog.merge_scouted_into_catalog(base, scouted)

    self.assertIn("MiniLlama", merged["models"])
    self.assertEqual(len(merged["models"]), 11)

  def test_invalid_base_catalog_is_rejected(self):
    scouted = catalog.scout_hf_config(llama_config(), "MiniLlama")
    base = with_value(tiny_catalog(), ("catalog_id",), "")

    with self.assertRaisesRegex(ValueError, "base_catalog: missing non-empty"):
      catalog.merge_scouted_into_catalog(base, scouted)


class ResolveSuiteTest(TempDirTestCase):

  def test_models_expand_into_one_preset_per_block(self):
    suites_dir = write_tiny_registry(self.tmp)

    resolved = catalog.resolve_suite(
        {"name": "tiny-suite", "models": ["Tiny"]}, suites_dir
    )

    self.assertEqual(
        resolved["presets"],
        [
            {
                "name": "Tiny",
                "layer_id": "block.Tiny.Local",
                "num_heads": 4,
                "head_dim": 16,
                "hidden_dim": 128,
                "num_kv_heads": 2,
                "embed_dim": 64,
                "sliding_window": 32,
                "eps": 1e-6,
                "mlp_activation": "swiglu",
                "qk_norm": False,
                "post_norms": False,
            },
            {
                "name": "Tiny-Global",
                "layer_id": "block.Tiny.Global",
                "num_heads": 4,
                "head_dim": 16,
                "hidden_dim": 128,
                "num_kv_heads": 4,
                "embed_dim": 64,
                "sliding_window": 0,
                "eps": 1e-6,
                "mlp_activation": "swiglu",
                "qk_norm": False,
                "post_norms": False,
            },
        ],
    )

  def test_models_record_block_multiplicities_as_layer_weights(self):
    suites_dir = write_tiny_registry(self.tmp)

    resolved = catalog.resolve_suite(
        {"name": "tiny-suite", "models": ["Tiny"]}, suites_dir
    )

    self.assertEqual(
        resolved["model_layer_weights"],
        {
            "Tiny": [
                {"layer_id": "block.Tiny.Local", "preset": "Tiny", "count": 2},
                {
                    "layer_id": "block.Tiny.Global",
                    "preset": "Tiny-Global",
                    "count": 1,
                },
            ],
        },
    )
    self.assertEqual(resolved["catalog_id"], "tiny-catalog")

  def test_catalog_suite_defaults_to_fp32_gate_and_generic_threads(self):
    suites_dir = write_tiny_registry(self.tmp)

    resolved = catalog.resolve_suite(
        {"name": "tiny-suite", "models": ["Tiny"]}, suites_dir
    )

    self.assertEqual(resolved["dtype"], "fp32")
    self.assertEqual(resolved["quality_gate"], {"rtol": 1e-4, "atol": 1e-5})
    self.assertEqual(resolved["thread_counts"], [1, 4, 8, 16])
    self.assertEqual(resolved["cliff_profile"]["name"], "none")

  def test_explicit_layer_ids_select_single_blocks(self):
    suites_dir = write_tiny_registry(self.tmp)

    resolved = catalog.resolve_suite(
        {"name": "one-block", "layer_ids": ["block.Tiny.Global"]}, suites_dir
    )

    self.assertEqual(
        [preset["name"] for preset in resolved["presets"]], ["Tiny-Global"]
    )
    self.assertEqual(resolved["model_layer_weights"], {})

  def test_machine_override_replaces_thread_counts_and_ceilings(self):
    suites_dir = write_tiny_registry(self.tmp)
    suite = {
        "name": "tiny-suite",
        "models": ["Tiny"],
        "thread_counts": [2],
        "ceilings_file": "suite_ceilings.json",
    }

    resolved = catalog.resolve_suite(suite, suites_dir, machine_override="box")

    self.assertEqual(resolved["machine"], "box")
    self.assertEqual(resolved["thread_counts"], [1, 4])
    self.assertEqual(resolved["ceilings_file"], "")
    self.assertEqual(
        resolved["machine_profile"]["physical_cores_per_domain"], 4
    )

  def test_suite_machine_keeps_thread_counts_the_suite_defines(self):
    suites_dir = write_tiny_registry(self.tmp)
    suite = {
        "name": "tiny-suite",
        "models": ["Tiny"],
        "machine": "box",
        "thread_counts": [2],
    }

    resolved = catalog.resolve_suite(suite, suites_dir)

    self.assertEqual(resolved["thread_counts"], [2])

  def test_regime_override_sets_sweep_and_keeps_only_matching_families(self):
    suites_dir = write_tiny_registry(self.tmp)
    suite = {"name": "tiny-suite", "models": ["Tiny"], "seq_lengths": [4]}

    resolved = catalog.resolve_suite(
        suite, suites_dir, regime_override="decode_only"
    )

    self.assertEqual(resolved["batch_sizes"], [1])
    self.assertEqual(resolved["seq_lengths"], [8])
    self.assertEqual(
        resolved["families"],
        [
            "BM_StandardAttention_Decode",
            "BM_DecodeOptimizedAttention",
            "BM_TransformerBlock_Decode",
        ],
    )

  def test_suite_regime_keeps_sweep_points_the_suite_defines(self):
    suites_dir = write_tiny_registry(self.tmp)
    suite = {
        "name": "tiny-suite",
        "models": ["Tiny"],
        "regime": "short",
        "seq_lengths": [4],
    }

    resolved = catalog.resolve_suite(suite, suites_dir)

    self.assertEqual(resolved["seq_lengths"], [4])
    self.assertEqual(resolved["batch_sizes"], [1])

  def test_regime_without_matching_family_is_rejected(self):
    suites_dir = write_tiny_registry(self.tmp)
    suite = {
        "name": "tiny-suite",
        "models": ["Tiny"],
        "families": ["BM_RmsNorm"],
    }

    with self.assertRaisesRegex(
        ValueError, "zero families matching regime 'decode_only'"
    ):
      catalog.resolve_suite(suite, suites_dir, regime_override="decode_only")

  def test_cliff_sweep_adds_perturbed_variants_after_each_preset(self):
    suites_dir = write_tiny_registry(self.tmp)

    resolved = catalog.resolve_suite(
        {"name": "tiny-suite", "models": ["Tiny"]},
        suites_dir,
        cliff_override="plus_minus_8",
    )

    self.assertEqual(
        [preset["name"] for preset in resolved["presets"]],
        [
            "Tiny",
            "Tiny__embed_dim_m8",
            "Tiny__embed_dim_p8",
            "Tiny-Global",
            "Tiny-Global__embed_dim_m8",
            "Tiny-Global__embed_dim_p8",
        ],
    )
    minus_variant = resolved["presets"][1]
    self.assertEqual(minus_variant["embed_dim"], 56)
    self.assertEqual(minus_variant["cliff_parent"], "Tiny")
    self.assertEqual(minus_variant["cliff_dim"], "embed_dim")
    self.assertEqual(minus_variant["cliff_delta"], -8)

  def test_identical_blocks_are_deduplicated_on_request(self):
    local_params = tiny_catalog()["composite_blocks"][0]["params"]
    twin_catalog = with_value(
        tiny_catalog(), ("composite_blocks", 1, "params"), local_params
    )
    suites_dir = write_tiny_registry(self.tmp, twin_catalog)
    suite = {
        "name": "tiny-suite",
        "models": ["Tiny"],
        "deduplicate_identical_presets": True,
    }

    resolved = catalog.resolve_suite(suite, suites_dir)

    self.assertEqual(
        [preset["name"] for preset in resolved["presets"]], ["Tiny"]
    )
    self.assertEqual(
        resolved["model_layer_weights"]["Tiny"][1],
        {
            "layer_id": "block.Tiny.Global",
            "preset": "Tiny",
            "count": 1,
            "deduplicated_from": "Tiny-Global",
        },
    )

  def test_identical_blocks_stay_separate_by_default(self):
    local_params = tiny_catalog()["composite_blocks"][0]["params"]
    twin_catalog = with_value(
        tiny_catalog(), ("composite_blocks", 1, "params"), local_params
    )
    suites_dir = write_tiny_registry(self.tmp, twin_catalog)

    resolved = catalog.resolve_suite(
        {"name": "tiny-suite", "models": ["Tiny"]}, suites_dir
    )

    self.assertEqual(
        [preset["name"] for preset in resolved["presets"]],
        ["Tiny", "Tiny-Global"],
    )

  def test_block_without_attention_heads_is_skipped_as_unsupported(self):
    linear_catalog = with_value(
        tiny_catalog(), ("composite_blocks", 1, "params", "num_heads"), None
    )
    suites_dir = write_tiny_registry(self.tmp, linear_catalog)

    resolved = catalog.resolve_suite(
        {"name": "tiny-suite", "models": ["Tiny"]}, suites_dir
    )

    self.assertEqual(
        [preset["name"] for preset in resolved["presets"]], ["Tiny"]
    )
    self.assertEqual(resolved["skipped_layers"], ["block.Tiny.Global"])
    self.assertEqual(
        resolved["model_layer_weights"]["Tiny"][1],
        {
            "layer_id": "block.Tiny.Global",
            "preset": None,
            "count": 1,
            "unsupported": True,
        },
    )

  def test_partial_rotary_block_is_flagged_as_approximated(self):
    partial_catalog = with_value(
        tiny_catalog(), ("composite_blocks", 0, "params", "rotary_dim"), 8
    )
    suites_dir = write_tiny_registry(self.tmp, partial_catalog)

    resolved = catalog.resolve_suite(
        {"name": "tiny-suite", "models": ["Tiny"]}, suites_dir
    )

    self.assertEqual(
        resolved["approximated_presets"], {"Tiny": ["partial_rotary_dim=8/16"]}
    )
    self.assertEqual(
        resolved["presets"][0]["approximated_features"],
        ["partial_rotary_dim=8/16"],
    )

  def test_cliff_sweep_leaves_presets_without_the_target_dimension_alone(self):
    suites_dir = write_tiny_registry(self.tmp)
    suite = {"name": "inline", "presets": [{"name": "NoEmbed", "num_heads": 4}]}

    resolved = catalog.resolve_suite(
        suite, suites_dir, cliff_override="plus_minus_8"
    )

    self.assertEqual(resolved["presets"], [{"name": "NoEmbed", "num_heads": 4}])

  def test_cliff_sweep_skips_variants_that_would_not_stay_positive(self):
    suites_dir = write_tiny_registry(self.tmp)
    suite = {"name": "inline", "presets": [{"name": "Small", "embed_dim": 8}]}

    resolved = catalog.resolve_suite(
        suite, suites_dir, cliff_override="plus_minus_8"
    )

    self.assertEqual(
        [preset["name"] for preset in resolved["presets"]],
        ["Small", "Small__embed_dim_p8"],
    )

  def test_suite_selecting_only_unsupported_blocks_is_rejected(self):
    linear_catalog = with_value(
        tiny_catalog(), ("composite_blocks", 1, "params", "num_heads"), None
    )
    suites_dir = write_tiny_registry(self.tmp, linear_catalog)
    suite = {"name": "linear-only", "layer_ids": ["block.Tiny.Global"]}

    with self.assertRaisesRegex(
        ValueError, "Suite linear-only resolved to zero runnable presets"
    ):
      catalog.resolve_suite(suite, suites_dir)

  def test_non_object_suite_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "Suite must be a JSON object"):
      catalog.resolve_suite(["Tiny"], self.tmp)

  def test_unknown_model_is_rejected(self):
    suites_dir = write_tiny_registry(self.tmp)

    with self.assertRaisesRegex(ValueError, "references unknown model 'Nope'"):
      catalog.resolve_suite({"name": "bad", "models": ["Nope"]}, suites_dir)

  def test_unknown_layer_id_is_rejected(self):
    suites_dir = write_tiny_registry(self.tmp)

    with self.assertRaisesRegex(
        ValueError, "Unknown composite_block layer_id 'block.Nope'"
    ):
      catalog.resolve_suite(
          {"name": "bad", "layer_ids": ["block.Nope"]}, suites_dir
      )

  def test_unknown_profile_error_lists_known_profiles(self):
    suites_dir = write_tiny_registry(self.tmp)

    with self.assertRaisesRegex(
        catalog.UnknownProfileError,
        r"Unknown machine_threading 'laptop' in .*hyperparams.json; "
        r"known: box$",
    ) as raised:
      catalog.resolve_suite(
          {"name": "tiny-suite", "models": ["Tiny"]},
          suites_dir,
          machine_override="laptop",
      )
    self.assertEqual(raised.exception.kind, "machine_threading")
    self.assertEqual(raised.exception.name, "laptop")
    self.assertEqual(raised.exception.known, ("box",))

  def test_unknown_regime_quant_scheme_and_cliff_sweep_are_rejected(self):
    suites_dir = write_tiny_registry(self.tmp)
    suite = {"name": "tiny-suite", "models": ["Tiny"]}
    cases = (
        ({"regime_override": "nope"}, "Unknown regime 'nope'"),
        ({"quant_override": "nope"}, "Unknown quant_scheme 'nope'"),
        ({"cliff_override": "nope"}, "Unknown cliff_sweep 'nope'"),
    )
    for overrides, message in cases:
      with self.subTest(overrides=overrides):
        with self.assertRaisesRegex(catalog.UnknownProfileError, message):
          catalog.resolve_suite(suite, suites_dir, **overrides)

  def test_malformed_suite_fields_are_rejected(self):
    suites_dir = write_tiny_registry(self.tmp)
    cases = (
        ("numeric_preset", {"presets": ["Tiny", 7]}, "preset must be a name"),
        ("unnamed_preset", {"presets": [{"embed_dim": 8}]}, "with a 'name'"),
        (
            "repeated_preset",
            {"presets": ["Tiny", {"name": "Tiny", "embed_dim": 8}]},
            "Suite bad names preset 'Tiny' more than once",
        ),
        ("presets_not_a_list", {"presets": "Tiny"}, "'presets' must be a list"),
        ("numeric_family", {"families": [3]}, "'families' must be a list"),
        ("numeric_model", {"models": [3]}, "'models' must be a list of names"),
        (
            "layer_ids_not_a_list",
            {"layer_ids": "block.Tiny.Local"},
            "'layer_ids' must be a list of names",
        ),
        ("numeric_catalog", {"catalog": 5}, "'catalog' must be a string: 5"),
        (
            "regime_list",
            {"regime": ["short"]},
            r"'regime' must be a string: \['short'\]",
        ),
        ("zero_batch_size", {"batch_sizes": [0]}, "batch_sizes contains"),
        ("empty_seq_lengths", {"seq_lengths": []}, "seq_lengths must be"),
    )
    for name, fields, message in cases:
      with self.subTest(case=name):
        with self.assertRaisesRegex(ValueError, message):
          catalog.resolve_suite({"name": "bad", **fields}, suites_dir)

  def test_null_or_empty_registry_paths_select_the_default_files(self):
    suites_dir = write_tiny_registry(self.tmp)
    for unset in (None, ""):
      with self.subTest(unset=unset):
        suite = {
            "name": "tiny-suite",
            "models": ["Tiny"],
            "catalog": unset,
            "hyperparams": unset,
        }

        resolved = catalog.resolve_suite(suite, suites_dir)

        self.assertEqual(resolved["catalog_id"], "tiny-catalog")
        self.assertEqual(resolved["quant_scheme"], "fp32")

  def test_inline_suite_without_overrides_is_returned_as_a_copy(self):
    suite = {"name": "inline", "presets": ["TinyCI"], "families": ["BM_RoPE"]}

    resolved = catalog.resolve_suite(suite, self.tmp)

    self.assertEqual(resolved, suite)
    self.assertIsNot(resolved, suite)

  def test_inline_suite_with_override_binds_registry_profile(self):
    suites_dir = write_tiny_registry(self.tmp)
    suite = {"name": "inline", "presets": ["TinyCI"]}

    resolved = catalog.resolve_suite(suite, suites_dir, machine_override="box")

    self.assertEqual(resolved["thread_counts"], [1, 4])
    self.assertEqual(resolved["presets"], ["TinyCI"])
    self.assertNotIn("catalog_id", resolved)

  def test_inline_suite_naming_a_profile_binds_it_from_the_registry(self):
    suites_dir = write_tiny_registry(self.tmp)
    cliff_profile = {
        "name": "plus_minus_8",
        "deltas": [-8, 0, 8],
        "target_dims": ["embed_dim"],
    }
    cases = (
        ("machine", "box", "thread_counts", [1, 4]),
        ("regime", "short", "seq_lengths", [64]),
        ("quant_scheme", "fp32", "dtype", "fp32"),
        ("cliff_sweep", "plus_minus_8", "cliff_profile", cliff_profile),
    )
    for profile_key, profile, resolved_key, expected in cases:
      with self.subTest(profile_key=profile_key):
        suite = {"name": "inline", "presets": ["TinyCI"], profile_key: profile}

        resolved = catalog.resolve_suite(suite, suites_dir)

        self.assertEqual(resolved[resolved_key], expected)

  def test_inline_suite_naming_an_unknown_profile_is_rejected(self):
    suites_dir = write_tiny_registry(self.tmp)
    suite = {"name": "inline", "presets": ["TinyCI"], "regime": "nope"}

    with self.assertRaisesRegex(
        catalog.UnknownProfileError, "Unknown regime 'nope'"
    ):
      catalog.resolve_suite(suite, suites_dir)

  def test_missing_registry_file_raises_os_error(self):
    suites_dir = self.tmp / "suites"
    suites_dir.mkdir()

    with self.assertRaisesRegex(FileNotFoundError, "gemma_qwen_edge.json"):
      catalog.resolve_suite({"name": "bad", "models": ["Tiny"]}, suites_dir)


class ResolveRepoSuitesTest(unittest.TestCase):

  def test_mixed_name_and_inline_presets_expand_only_inline_ones(self):
    suite = {
        "name": "mixed-cliff",
        "presets": [
            "TinyCI",
            {
                "name": "Custom",
                "num_heads": 4,
                "head_dim": 64,
                "num_kv_heads": 2,
                "embed_dim": 256,
                "hidden_dim": 512,
            },
        ],
    }

    resolved = catalog.resolve_suite(
        suite, SUITES_DIR, cliff_override="tile_boundaries"
    )

    names = [catalog.preset_name(preset) for preset in resolved["presets"]]
    self.assertEqual(names[0], "TinyCI")
    self.assertIn("Custom__embed_dim_m16", names)
    self.assertIn("Custom__hidden_dim_p16", names)
    self.assertEqual(len(names), 10)

  def test_gemma4_31b_quant_and_cliff_overrides(self):
    suite = catalog.load_json_object(
        SUITES_DIR / "gemma4-31b-sapphire-rapids.json"
    )

    resolved = catalog.resolve_suite(
        suite,
        SUITES_DIR,
        quant_override="qb4w_b32",
        cliff_override="tile_boundaries",
    )

    self.assertEqual(len(resolved["presets"]), 18)
    self.assertEqual(resolved["dtype"], "int4_block")
    self.assertAlmostEqual(resolved["quality_gate"]["min_cosine"], 0.99)
    self.assertIn(
        "k_eq_v", resolved["approximated_presets"]["Gemma4-31B-Global"]
    )

  def test_gemma4_31b_decode_regime_override_keeps_decode_families(self):
    suite = catalog.load_json_object(
        SUITES_DIR / "gemma4-31b-sapphire-rapids.json"
    )

    resolved = catalog.resolve_suite(
        suite, SUITES_DIR, regime_override="decode_ctx32768"
    )

    self.assertEqual(resolved["seq_lengths"], [32768])
    self.assertEqual(
        resolved["families"],
        [
            "BM_StandardAttention_Decode",
            "BM_DecodeOptimizedAttention",
            "BM_TransformerBlock_Decode",
        ],
    )

  def test_starter_suite_resolves_local_and_global_archetype_presets(self):
    suite = catalog.load_json_object(SUITES_DIR / "edge-starter-models.json")

    resolved = catalog.resolve_suite(suite, SUITES_DIR)

    preset_names = {preset["name"] for preset in resolved["presets"]}
    self.assertLessEqual(
        {
            "Gemma3-270M",
            "Gemma3-270M-Global",
            "Gemma3-1B",
            "Gemma3-1B-Global",
            "Gemma4-E2B",
            "Gemma4-E2B-Global",
        },
        preset_names,
    )
    weights = resolved["model_layer_weights"]
    self.assertLessEqual(
        {weight["preset"] for weight in weights["Gemma4-E2B"]}, preset_names
    )
    self.assertEqual(
        sum(weight["count"] for weight in weights["Qwen3.5-0.8B"]), 24
    )

  def test_starter_suite_block_estimates_cover_supported_layers_only(self):
    suite = catalog.load_json_object(SUITES_DIR / "edge-starter-models.json")
    resolved = catalog.resolve_suite(suite, SUITES_DIR)
    layers = [
        block_record(preset["name"], 1.0) for preset in resolved["presets"]
    ]

    estimates = layer.estimate_model_blocks_latency(
        layers, resolved["model_layer_weights"]
    )

    by_model = {estimate["model"]: estimate for estimate in estimates}
    self.assertTrue(by_model["Gemma4-E2B"]["complete"])
    self.assertEqual(by_model["Gemma4-E2B"]["covered_layers"], 35)
    self.assertFalse(by_model["Qwen3.5-0.8B"]["complete"])
    self.assertEqual(by_model["Qwen3.5-0.8B"]["covered_layers"], 6)
    self.assertEqual(by_model["Qwen3.5-0.8B"]["total_layers"], 24)

  def test_epyc_override_binds_epyc_threads_without_ceilings(self):
    suite = catalog.load_json_object(SUITES_DIR / "gemma4-e2b-cpu.json")

    resolved = catalog.resolve_suite(
        suite, SUITES_DIR, machine_override="epyc-7443p"
    )

    self.assertEqual(resolved["machine"], "epyc-7443p")
    self.assertEqual(resolved["thread_counts"], [1, 4, 8, 12, 24])
    self.assertEqual(resolved["ceilings_file"], "")

  def test_sapphire_rapids_override_binds_existing_ceilings_file(self):
    suite = catalog.load_json_object(SUITES_DIR / "gemma4-e2b-cpu.json")

    resolved = catalog.resolve_suite(
        suite, SUITES_DIR, machine_override="sapphire-rapids"
    )

    self.assertEqual(resolved["machine"], "sapphire-rapids")
    self.assertEqual(
        Path(resolved["ceilings_file"]), SAPPHIRE_RAPIDS_CEILINGS.resolve()
    )

  def test_epyc_override_clears_ceilings_of_sapphire_rapids_suite(self):
    suite = catalog.load_json_object(
        SUITES_DIR / "gemma4-e2b-sapphire-rapids.json"
    )

    resolved = catalog.resolve_suite(
        suite, SUITES_DIR, machine_override="epyc-7443p"
    )

    self.assertEqual(resolved["ceilings_file"], "")


def run_catalog_cli(argv: Sequence[str]) -> tuple[int, str, str]:
  """Runs `catalog.main(argv)`; returns `(exit_code, stdout, stderr)`."""
  stdout = io.StringIO()
  stderr = io.StringIO()
  with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    exit_code = catalog.main(argv)
  return exit_code, stdout.getvalue(), stderr.getvalue()


class CatalogCliTest(TempDirTestCase):

  def test_validate_reports_counts_of_validated_files(self):
    suites_dir = write_tiny_registry(self.tmp)
    (suites_dir / "tiny.json").write_text(
        json.dumps({"name": "tiny", "models": ["Tiny"], "regime": "short"}),
        encoding="utf-8",
    )

    exit_code, stdout, _ = run_catalog_cli(
        ["validate", "--layer-root", str(self.tmp)]
    )

    self.assertEqual(exit_code, 0)
    self.assertIn(
        "VALIDATED 1 catalog(s), 1 hyperparam registry, 1 suite(s)", stdout
    )

  def test_validate_accepts_checked_in_layer_configs(self):
    exit_code, stdout, _ = run_catalog_cli(
        ["validate", "--layer-root", str(LAYER_ROOT)]
    )

    self.assertEqual(exit_code, 0)
    self.assertIn("VALIDATED", stdout)

  def test_validate_fails_for_layer_root_without_catalog(self):
    layer_root = self.tmp / "absent"

    exit_code, stdout, stderr = run_catalog_cli(
        ["validate", "--layer-root", str(layer_root)]
    )

    self.assertEqual(exit_code, 1)
    self.assertEqual(stdout, "")
    self.assertEqual(
        stderr,
        "[aeb.catalog] ERROR: No layer catalog JSON found in "
        f"{layer_root / 'catalog'}\n",
    )

  def test_validate_fails_with_one_line_error_for_broken_suite(self):
    suites_dir = write_tiny_registry(self.tmp)
    (suites_dir / "broken.json").write_text(
        json.dumps({"name": "broken", "models": ["Nope"]}), encoding="utf-8"
    )

    exit_code, _, stderr = run_catalog_cli(
        ["validate", "--layer-root", str(self.tmp)]
    )

    self.assertEqual(exit_code, 1)
    self.assertRegex(
        stderr, r"^\[aeb.catalog\] ERROR: Suite broken references unknown model"
    )

  def test_summary_prints_table_sizes_and_block_composition(self):
    write_tiny_registry(self.tmp)
    catalog_path = self.tmp / "catalog" / "gemma_qwen_edge.json"

    exit_code, stdout, _ = run_catalog_cli(
        ["summary", "--catalog", str(catalog_path)]
    )

    self.assertEqual(exit_code, 0)
    self.assertIn("Catalog: tiny-catalog (1 models)", stdout)
    self.assertIn("composite_blocks    :   2", stdout)
    self.assertIn("( 3L): 2x Tiny.Local, 1x Tiny.Global", stdout)

  def test_summary_lists_models_of_checked_in_catalog(self):
    exit_code, stdout, _ = run_catalog_cli(
        ["summary", "--catalog", str(REPO_CATALOG)]
    )

    self.assertEqual(exit_code, 0)
    self.assertIn("Gemma4-E2B", stdout)

  def test_validate_defaults_to_the_checked_in_layer_directory(self):
    exit_code, stdout, _ = run_catalog_cli(["validate"])

    self.assertEqual(exit_code, 0)
    self.assertIn(f"under {LAYER_ROOT}\n", stdout)

  def test_summary_defaults_to_the_checked_in_catalog(self):
    exit_code, stdout, _ = run_catalog_cli(["summary"])

    self.assertEqual(exit_code, 0)
    self.assertRegex(stdout, r"^Catalog: \S+ \(10 models\)\n")

  def test_scout_writes_scouted_catalog_to_out_file(self):
    hf_path = self.tmp / "config.json"
    hf_path.write_text(json.dumps(llama_config()), encoding="utf-8")
    out_path = self.tmp / "scouted.json"

    exit_code, _, _ = run_catalog_cli([
        "scout",
        "--hf-config",
        str(hf_path),
        "--model-name",
        "MiniLlama",
        "--out",
        str(out_path),
    ])

    self.assertEqual(exit_code, 0)
    self.assertIn("MiniLlama", load_strict_json(out_path)["models"])

  def test_scout_prints_scouted_catalog_without_out_file(self):
    hf_path = self.tmp / "config.json"
    hf_path.write_text(json.dumps(llama_config()), encoding="utf-8")

    exit_code, stdout, _ = run_catalog_cli(
        ["scout", "--hf-config", str(hf_path), "--model-name", "MiniLlama"]
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(json.loads(stdout)["catalog_id"], "scouted-minillama")

  def test_scout_merges_into_existing_catalog_in_place(self):
    hf_path = self.tmp / "config.json"
    hf_path.write_text(json.dumps(llama_config()), encoding="utf-8")
    catalog_path = self.tmp / "catalog.json"
    catalog_path.write_text(json.dumps(tiny_catalog()), encoding="utf-8")

    exit_code, stdout, _ = run_catalog_cli([
        "scout",
        "--hf-config",
        str(hf_path),
        "--model-name",
        "MiniLlama",
        "--merge-into",
        str(catalog_path),
    ])

    self.assertEqual(exit_code, 0)
    self.assertIn(f"Merged MiniLlama into {catalog_path}", stdout)
    self.assertEqual(
        sorted(load_strict_json(catalog_path)["models"]), ["MiniLlama", "Tiny"]
    )

  def test_scout_merge_with_out_file_leaves_base_catalog_untouched(self):
    hf_path = self.tmp / "config.json"
    hf_path.write_text(json.dumps(llama_config()), encoding="utf-8")
    catalog_path = self.tmp / "catalog.json"
    catalog_path.write_text(json.dumps(tiny_catalog()), encoding="utf-8")
    out_path = self.tmp / "merged.json"

    exit_code, _, _ = run_catalog_cli([
        "scout",
        "--hf-config",
        str(hf_path),
        "--model-name",
        "MiniLlama",
        "--merge-into",
        str(catalog_path),
        "--out",
        str(out_path),
    ])

    self.assertEqual(exit_code, 0)
    self.assertEqual(load_strict_json(catalog_path), tiny_catalog())
    self.assertIn("MiniLlama", load_strict_json(out_path)["models"])

  def test_scout_fails_with_one_line_error_for_unusable_config(self):
    hf_path = self.tmp / "config.json"
    hf_path.write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")

    exit_code, _, stderr = run_catalog_cli(
        ["scout", "--hf-config", str(hf_path), "--model-name", "Broken"]
    )

    self.assertEqual(exit_code, 1)
    self.assertRegex(
        stderr, r"^\[aeb.catalog\] ERROR: HF config requires a positive integer"
    )

  def test_resolve_writes_resolved_suite_to_out_file(self):
    out_path = self.tmp / "nested" / "resolved.json"

    exit_code, _, _ = run_catalog_cli([
        "resolve",
        "--suite",
        str(SUITES_DIR / "gemma4-e2b-cpu.json"),
        "--machine",
        "epyc-7443p",
        "--out",
        str(out_path),
    ])

    self.assertEqual(exit_code, 0)
    self.assertEqual(load_strict_json(out_path)["machine"], "epyc-7443p")

  def test_resolve_prints_resolved_suite_without_out_file(self):
    suites_dir = write_tiny_registry(self.tmp)
    suite_path = suites_dir / "tiny.json"
    suite_path.write_text(
        json.dumps({"name": "tiny", "models": ["Tiny"]}), encoding="utf-8"
    )

    exit_code, stdout, _ = run_catalog_cli(
        ["resolve", "--suite", str(suite_path), "--regime", "decode_only"]
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(json.loads(stdout)["seq_lengths"], [8])

  def test_resolve_fails_with_one_line_error_for_unknown_machine(self):
    exit_code, stdout, stderr = run_catalog_cli([
        "resolve",
        "--suite",
        str(SUITES_DIR / "gemma4-e2b-cpu.json"),
        "--machine",
        "laptop",
    ])

    self.assertEqual(exit_code, 1)
    self.assertEqual(stdout, "")
    self.assertRegex(
        stderr,
        r"^\[aeb.catalog\] ERROR: Unknown machine_threading 'laptop' in "
        r".*; known: apple-m5-max, dgx-spark, epyc-7443p, sapphire-rapids, "
        r"strix-halo\n$",
    )

  def test_missing_input_file_fails_with_one_line_error(self):
    exit_code, _, stderr = run_catalog_cli(
        ["resolve", "--suite", str(self.tmp / "absent.json")]
    )

    self.assertEqual(exit_code, 1)
    self.assertRegex(stderr, r"^\[aeb.catalog\] ERROR: .*absent.json")


class LayerCliUsageTest(TempDirTestCase):

  def test_invalid_flags_are_usage_errors_reported_before_running(self):
    suite_file = str(SUITES_DIR / "gemma4-e2b-cpu.json")
    cases = (
        ("non_fp32_dtype", ["--dtype", "int4_block"], "refusing dtype='int4_b"),
        ("unknown_backend", ["--backend", "bogus"], "--backend='bogus'"),
        ("negative_batch", ["--batch-size", "-1"], "--batch-size must be >="),
        ("negative_seq_len", ["--seq-len", "-1"], r"--seq-len must be >= 0"),
        ("negative_threads", ["--threads", "-4"], r"must be >= 0 \(got -4\)"),
        (
            "families_outside_regime",
            [
                "--suite-file",
                suite_file,
                "--regime",
                "decode_ctx32768",
                "--families",
                "BM_RmsNorm",
            ],
            r"--families=\['BM_RmsNorm'\] has no families matching regime "
            "mode='decode'",
        ),
        (
            "quant_scheme_not_fp32",
            ["--suite-file", suite_file, "--quant-scheme", "qb4w_b32"],
            "executes FP32 graphs only; refusing dtype='int4_block'",
        ),
        (
            "quant_scheme_with_quantized_kv_cache",
            ["--suite-file", suite_file, "--quant-scheme", "fp32_kv_q8_0"],
            "refusing quant scheme 'fp32_kv_q8_0' with kv_dtype='q8_0'",
        ),
        ("nan_rtol", ["--parity-rtol", "nan"], "rtol must be a finite number"),
        ("infinite_kl", ["--parity-max-kl", "inf"], "max_kl must be a finite"),
        ("negative_atol", ["--parity-atol", "-1"], "tolerances must be >= 0"),
        ("presets_without_names", ["--presets", " , "], "--presets must name"),
        ("families_without_names", ["--families", ""], "--families must name"),
        (
            "unknown_regime",
            ["--regime", "nope"],
            "Unknown regime 'nope' .* Pass --regime with a known profile",
        ),
    )
    for name, argv, message in cases:
      with self.subTest(case=name):
        bench = FakeLayerBench()

        result = run_layer(self.tmp / name, argv, bench)

        self.assertEqual(result.exit_code, 2)
        self.assertRegex(result.stderr, r"^\[aeb\] ERROR: ")
        self.assertRegex(result.stderr, message)
        self.assertEqual(bench.commands, [])
        self.assertFalse(result.run_dir.exists())

  def test_unknown_machine_profile_names_known_profiles_and_remedy(self):
    suite_path = write_tiny_suite(self.tmp)
    registry_path = (self.tmp / "catalog" / "hyperparams.json").resolve()

    result = run_layer(
        self.tmp,
        ["--suite-file", str(suite_path), "--machine-profile", "my-laptop"],
        FakeLayerBench(),
    )

    self.assertEqual(result.exit_code, 2)
    self.assertEqual(
        result.stderr,
        "[aeb] ERROR: Unknown machine_threading 'my-laptop' in "
        f"{registry_path}; known: box. Pass --machine-profile with a known "
        "profile, or omit it to run without a host threading profile (make "
        "bench: MACHINE_PROFILE=<profile> or MACHINE_PROFILE=).\n",
    )

  def test_explicit_ceilings_json_that_does_not_exist_is_a_usage_error(self):
    missing = self.tmp / "absent_ceilings.json"
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp, [*CASE_FLAGS, "--ceilings-json", str(missing)], bench
    )

    self.assertEqual(result.exit_code, 2)
    self.assertEqual(
        result.stderr, f"[aeb] ERROR: --ceilings-json not found: {missing}\n"
    )
    self.assertEqual(bench.commands, [])

  def test_explicit_kernel_timings_json_that_does_not_exist_is_an_error(self):
    missing = self.tmp / "absent_timings.json"
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp, [*CASE_FLAGS, "--kernel-timings-json", str(missing)], bench
    )

    self.assertEqual(result.exit_code, 2)
    self.assertEqual(
        result.stderr,
        f"[aeb] ERROR: --kernel-timings-json not found: {missing}\n",
    )
    self.assertEqual(bench.commands, [])

  def test_malformed_ceilings_json_is_a_usage_error(self):
    ceilings_path = self.tmp / "ceilings.json"
    ceilings_path.write_text("{not json", encoding="utf-8")

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--ceilings-json", str(ceilings_path)],
        FakeLayerBench(),
    )

    self.assertEqual(result.exit_code, 2)
    self.assertRegex(
        result.stderr,
        r"^\[aeb\] ERROR: cannot load Roofline inputs: .*invalid JSON",
    )

  def test_non_numeric_kernel_timings_are_a_usage_error(self):
    timings_path = self.tmp / "timings.json"
    timings_path.write_text('{"BM_RmsNorm": "fast"}', encoding="utf-8")

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--kernel-timings-json", str(timings_path)],
        FakeLayerBench(),
    )

    self.assertEqual(result.exit_code, 2)
    self.assertRegex(result.stderr, "'BM_RmsNorm' must be a finite number")

  def test_missing_suite_file_is_a_usage_error(self):
    result = run_layer(
        self.tmp,
        ["--suite-file", str(self.tmp / "absent_suite.json")],
        FakeLayerBench(),
    )

    self.assertEqual(result.exit_code, 2)
    self.assertRegex(
        result.stderr,
        r"^\[aeb\] ERROR: cannot resolve layer suite: .*absent_suite.json",
    )

  def test_suite_with_non_numeric_quality_gate_is_a_usage_error(self):
    suite_path = self.tmp / "inline.json"
    suite_path.write_text(
        json.dumps({
            "name": "inline",
            "presets": ["TinyCI"],
            "quality_gate": {"rtol": "loose"},
        }),
        encoding="utf-8",
    )

    result = run_layer(
        self.tmp, ["--suite-file", str(suite_path)], FakeLayerBench()
    )

    self.assertEqual(result.exit_code, 2)
    self.assertRegex(
        result.stderr, r"rtol must be a finite number \(got 'loose'\)"
    )

  def test_suite_with_non_object_quality_gate_is_a_usage_error(self):
    suite_path = write_tiny_suite(self.tmp, quality_gate=[1e-4, 1e-5])

    result = run_layer(
        self.tmp, ["--suite-file", str(suite_path)], FakeLayerBench()
    )

    self.assertEqual(result.exit_code, 2)
    self.assertEqual(
        result.stderr,
        "[aeb] ERROR: suite quality_gate must be a JSON object.\n",
    )

  def test_suite_with_non_object_quant_profile_is_a_usage_error(self):
    suite_path = self.tmp / "inline.json"
    suite_path.write_text(
        json.dumps({
            "name": "inline",
            "presets": ["TinyCI"],
            "quant_profile": "q8_0",
        }),
        encoding="utf-8",
    )

    result = run_layer(
        self.tmp, ["--suite-file", str(suite_path)], FakeLayerBench()
    )

    self.assertEqual(result.exit_code, 2)
    self.assertEqual(
        result.stderr,
        "[aeb] ERROR: suite quant_profile must be a JSON object.\n",
    )

  def test_suite_binding_its_own_quantized_profile_is_a_usage_error(self):
    suite_path = self.tmp / "inline.json"
    suite_path.write_text(
        json.dumps({
            "name": "inline",
            "presets": ["TinyCI"],
            "quant_profile": {"name": "custom", "activation_dtype": "int8"},
        }),
        encoding="utf-8",
    )
    bench = FakeLayerBench()

    result = run_layer(self.tmp, ["--suite-file", str(suite_path)], bench)

    self.assertEqual(result.exit_code, 2)
    self.assertRegex(
        result.stderr,
        "refusing quant scheme 'custom' with activation_dtype='int8'",
    )
    self.assertEqual(bench.commands, [])

  def test_dump_and_compare_directories_are_mutually_exclusive(self):
    stderr = io.StringIO()

    with contextlib.redirect_stderr(stderr):
      with self.assertRaisesRegex(SystemExit, "^2$"):
        layer.main([
            "--label",
            "run",
            "--dump-parity-dir",
            "reference_a",
            "--compare-parity-dir",
            "reference_b",
        ])

    self.assertIn(
        "--compare-parity-dir: not allowed with argument --dump-parity-dir",
        stderr.getvalue(),
    )

  def test_help_prints_flag_descriptions_and_exits_cleanly(self):
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
      with self.assertRaisesRegex(SystemExit, "^0$"):
        layer.main(["--help"])

    self.assertIn("--machine-profile MACHINE_PROFILE", stdout.getvalue())
    self.assertRegex(stdout.getvalue(), r"Roofline\s+%")


class LayerCliRunTest(TempDirTestCase):

  def test_default_run_sweeps_default_presets_and_families_in_one_call(self):
    bench = FakeLayerBench()

    result = run_layer(self.tmp, [], bench)

    summary = result.artifact("summary.json")
    params = result.artifact("meta.json")["params"]
    self.assertEqual(result.exit_code, 0)
    self.assertEqual(summary["status"], "ok")
    # 2 default presets x 9 default families at one operating point.
    self.assertEqual(summary["n_layers"], 18)
    self.assertEqual(len(bench.commands), 1)
    self.assertEqual(params["presets"], ["Gemma4-E2B", "Gemma4-E2B-WideMLP"])
    self.assertEqual(params["families"], list(catalog.DEFAULT_SUITE_FAMILIES))
    self.assertEqual(params["batch_size"], 1)
    self.assertEqual(params["seq_len"], 1024)
    self.assertEqual(params["threads"], 32)

  def test_command_line_carries_filter_suite_and_measurement_flags(self):
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp,
        [
            *CASE_FLAGS,
            "--binary",
            "/opt/bench/layer_bench",
            "--backend",
            "ynnpack",
            "--min-time",
            "0.01s",
            "--seed",
            "7",
            "--no-wipe-caches",
        ],
        bench,
    )

    benchmark_filter = (
        "^(BM_RmsNorm)/(Gemma4\\-E2B)/B:(2)/S:(16)/Threads:4(/real_time)?$"
    )
    self.assertEqual(
        bench.commands,
        [[
            "/opt/bench/layer_bench",
            "--backend=ynnpack",
            f"--suite_file={result.run_dir / 'resolved_suite.json'}",
            f"--benchmark_filter={benchmark_filter}",
            "--benchmark_format=json",
            f"--benchmark_out={result.run_dir / 'bench_000.json'}",
            "--benchmark_min_time=0.01s",
            "--wipe_caches=false",
            "--seed=7",
            "--parity_rtol=0.0001",
            "--parity_atol=1e-05",
        ]],
    )
    self.assertEqual(result.artifact("meta.json")["commands"], bench.commands)

  def test_run_directory_holds_artifacts_and_no_intermediate_output(self):
    result = run_layer(self.tmp, CASE_FLAGS, FakeLayerBench())

    self.assertEqual(
        sorted(path.name for path in result.run_dir.iterdir()),
        [
            "benchmarks.json",
            "driver.log",
            "meta.json",
            "resolved_suite.json",
            "summary.json",
            "sysinfo.json",
        ],
    )

  def test_summary_layer_record_normalizes_benchmark_entry(self):
    result = run_layer(self.tmp, CASE_FLAGS, FakeLayerBench())

    summary = result.artifact("summary.json")
    record = summary["layers"][0]
    self.assertEqual(summary["schema"], "aeb.layer.summary/1")
    self.assertEqual(summary["n_layers"], 1)
    self.assertIsNone(summary["parity_all_passed"])
    self.assertEqual(
        record["name"], "BM_RmsNorm/Gemma4-E2B/B:2/S:16/Threads:4/real_time"
    )
    self.assertEqual(record["preset"], "Gemma4-E2B")
    self.assertEqual(record["layer"], "RmsNorm")
    self.assertEqual(record["batch_size"], 2)
    self.assertEqual(record["seq_len"], 16)
    self.assertEqual(record["threads"], 4)
    self.assertEqual(record["real_time_ms"], 1.25)
    self.assertEqual(record["gflops"], 1024.0)

  def test_benchmarks_artifact_keeps_raw_entries_and_context(self):
    result = run_layer(self.tmp, CASE_FLAGS, FakeLayerBench())

    benchmarks = result.artifact("benchmarks.json")
    self.assertEqual(
        benchmarks["context"], {"host_name": "test-host", "num_cpus": 32}
    )
    self.assertEqual(len(benchmarks["benchmarks"]), 1)
    self.assertEqual(benchmarks["benchmarks"][0]["GFLOP/s"], 1024.0)

  def test_driver_log_captures_command_and_binary_output(self):
    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--binary", "/opt/bench/layer_bench"],
        FakeLayerBench(),
    )

    log_text = (result.run_dir / "driver.log").read_text(encoding="utf-8")
    self.assertRegex(log_text, r"^\$ /opt/bench/layer_bench --backend=xnnpack ")
    self.assertIn("fake stdout\n", log_text)

  def test_progress_is_reported_on_stderr(self):
    result = run_layer(self.tmp, CASE_FLAGS, FakeLayerBench())

    self.assertEqual(
        result.stderr,
        f"[aeb] layer litert-tensor/run (xnnpack) -> {result.run_dir}\n"
        "[aeb] layer litert-tensor/run: 1 benchmarks recorded [ok] -> "
        f"{result.run_dir}\n",
    )

  def test_meta_records_provenance_flags_and_machine_environment(self):
    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--track", "nightly", "--round", "3", "--notes", "warm"],
        FakeLayerBench(),
        environ={"AEB_MACHINE": "ut-box", "AEB_MACHINE_NOTES": "quiet lab"},
    )

    meta = result.artifact("meta.json")
    self.assertEqual(meta["schema"], "aeb.layer/1")
    self.assertEqual(meta["framework"], "litert-tensor")
    self.assertEqual(meta["label"], "run")
    self.assertEqual(meta["dtype"], "fp32")
    self.assertEqual(meta["track"], "nightly")
    self.assertEqual(meta["round"], 3)
    self.assertEqual(meta["notes"], "warm")
    self.assertEqual(meta["machine"], "ut-box")
    self.assertEqual(meta["machine_notes"], "quiet lab")
    self.assertEqual(meta["status"], "ok")
    self.assertEqual(meta["errors"], [])
    self.assertEqual(meta["started_at"], FIXED_UTC_NOW)
    self.assertEqual(
        meta["build"], {"harness_sha256": "test-digest", "repo_rev": "test-rev"}
    )

  def test_started_at_is_read_before_the_first_benchmark_runs(self):
    clock = TickingBench(FakeLayerBench())

    result = run_layer(
        self.tmp, CASE_FLAGS, clock.run_command, utc_now=clock.utc_now
    )

    self.assertEqual(result.artifact("meta.json")["started_at"], "tick-0")

  def test_other_frameworks_accept_any_backend_and_dtype(self):
    bench = FakeLayerBench()
    argv = [
        "--framework",
        "custom-runtime",
        "--backend",
        "vendor",
        "--dtype",
        "int8",
        *CASE_FLAGS,
    ]

    result = run_layer(self.tmp, argv, bench)

    self.assertEqual(result.exit_code, 0)
    self.assertIn("--backend=vendor", bench.commands[0])

  def test_machine_label_defaults_when_environment_has_none(self):
    result = run_layer(self.tmp, CASE_FLAGS, FakeLayerBench())

    self.assertEqual(result.artifact("meta.json")["machine"], "unknown-machine")

  def test_binary_defaults_to_environment_configuration(self):
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp,
        CASE_FLAGS,
        bench,
        environ={"AEB_LAYER_BIN": "/custom/layer_bench"},
    )

    self.assertEqual(bench.commands[0][0], "/custom/layer_bench")
    self.assertEqual(
        result.artifact("meta.json")["binary"], "/custom/layer_bench"
    )

  def test_repeated_presets_and_families_are_benchmarked_once(self):
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp,
        [
            "--presets",
            "Gemma4-E2B, Gemma4-E2B ,TinyCI,,Gemma4-E2B",
            "--families",
            "BM_RmsNorm,BM_RmsNorm",
            "--batch-size",
            "1",
            "--seq-len",
            "16",
            "--threads",
            "4",
        ],
        bench,
    )

    params = result.artifact("meta.json")["params"]
    self.assertEqual(params["presets"], ["Gemma4-E2B", "TinyCI"])
    self.assertEqual(params["families"], ["BM_RmsNorm"])
    self.assertEqual(result.artifact("summary.json")["n_layers"], 2)

  def test_repeated_presets_produce_one_parity_case_each(self):
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp,
        [
            "--presets",
            "Gemma4-E2B,Gemma4-E2B",
            "--families",
            "BM_RmsNorm,BM_RmsNorm",
            "--batch-size",
            "1",
            "--seq-len",
            "16",
            "--threads",
            "4",
            "--dump-parity-dir",
            str(self.tmp / "reference"),
        ],
        bench,
    )

    self.assertEqual(len(bench.commands), 1)
    self.assertEqual(len(result.artifact("parity.json")["records"]), 1)

  def test_family_repeated_in_suite_is_benchmarked_once(self):
    suite_path = self.tmp / "inline.json"
    suite_path.write_text(
        json.dumps({
            "name": "inline",
            "presets": ["TinyCI"],
            "families": ["BM_RmsNorm", "BM_RoPE", "BM_RmsNorm"],
            "batch_sizes": [1],
            "seq_lengths": [16],
        }),
        encoding="utf-8",
    )
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp,
        [
            "--suite-file",
            str(suite_path),
            "--dump-parity-dir",
            str(self.tmp / "reference"),
        ],
        bench,
    )

    params = result.artifact("meta.json")["params"]
    self.assertEqual(params["families"], ["BM_RmsNorm", "BM_RoPE"])
    self.assertEqual(len(bench.commands), 2)

  def test_suite_run_binds_regime_machine_and_model_block_estimates(self):
    suite_path = write_tiny_suite(self.tmp, regime="short", machine="box")

    result = run_layer(
        self.tmp, ["--suite-file", str(suite_path)], FakeLayerBench()
    )

    summary = result.artifact("summary.json")
    params = result.artifact("meta.json")["params"]
    self.assertEqual(result.exit_code, 0)
    self.assertEqual(summary["catalog_id"], "tiny-catalog")
    # 2 block presets x 9 families of the `both`-mode regime.
    self.assertEqual(summary["n_layers"], 18)
    self.assertEqual(params["presets"], ["Tiny", "Tiny-Global"])
    self.assertEqual(params["seq_len"], 64)
    self.assertEqual(params["threads"], 4)
    # Each of the 3 layers (2 local + 1 global) takes the fake's 1.25 ms.
    self.assertEqual(
        [
            (estimate["mode"], estimate["summed_blocks_ms"])
            for estimate in summary["estimated_model_blocks_ms"]
        ],
        [("decode", 3.75), ("prefill", 3.75)],
    )

  def test_requested_presets_filter_suite_and_keep_unknown_names(self):
    suite_path = write_tiny_suite(self.tmp, regime="short")

    result = run_layer(
        self.tmp,
        [
            "--suite-file",
            str(suite_path),
            "--presets",
            "Tiny-Global,BuiltIn",
            "--families",
            "BM_RmsNorm",
        ],
        FakeLayerBench(),
    )

    resolved_suite = result.artifact("resolved_suite.json")
    self.assertEqual(
        [catalog.preset_name(preset) for preset in resolved_suite["presets"]],
        ["Tiny-Global", "BuiltIn"],
    )
    self.assertEqual(resolved_suite["presets"][1], "BuiltIn")
    self.assertEqual(result.artifact("summary.json")["n_layers"], 2)

  def test_requested_preset_brings_its_cliff_sweep_variants(self):
    suite_path = write_tiny_suite(self.tmp, regime="short")

    result = run_layer(
        self.tmp,
        [
            "--suite-file",
            str(suite_path),
            "--cliff-sweep",
            "plus_minus_8",
            "--presets",
            "Tiny",
            "--families",
            "BM_RmsNorm",
        ],
        FakeLayerBench(),
    )

    self.assertEqual(
        result.artifact("meta.json")["params"]["presets"],
        ["Tiny", "Tiny__embed_dim_m8", "Tiny__embed_dim_p8"],
    )

  def test_requested_families_are_filtered_by_suite_regime_mode(self):
    suite_path = write_tiny_suite(self.tmp)

    result = run_layer(
        self.tmp,
        [
            "--suite-file",
            str(suite_path),
            "--regime",
            "decode_only",
            "--families",
            "BM_RmsNorm,BM_StandardAttention_Decode",
        ],
        FakeLayerBench(),
    )

    self.assertEqual(
        result.artifact("meta.json")["params"]["families"],
        ["BM_StandardAttention_Decode"],
    )

  def test_adhoc_cliff_sweep_expands_catalog_preset_without_suite(self):
    result = run_layer(
        self.tmp,
        [
            "--presets",
            "Gemma4-E2B",
            "--families",
            "BM_RmsNorm",
            "--cliff-sweep",
            "tile_boundaries",
            "--batch-size",
            "1",
            "--seq-len",
            "128",
            "--threads",
            "4",
        ],
        FakeLayerBench(),
    )

    summary = result.artifact("summary.json")
    self.assertEqual(result.exit_code, 0)
    self.assertEqual(summary["n_layers"], 9)
    self.assertEqual(
        {record["preset"] for record in summary["layers"]},
        {
            "Gemma4-E2B",
            "Gemma4-E2B__embed_dim_m16",
            "Gemma4-E2B__embed_dim_m8",
            "Gemma4-E2B__embed_dim_p8",
            "Gemma4-E2B__embed_dim_p16",
            "Gemma4-E2B__hidden_dim_m16",
            "Gemma4-E2B__hidden_dim_m8",
            "Gemma4-E2B__hidden_dim_p8",
            "Gemma4-E2B__hidden_dim_p16",
        },
    )

  def test_repo_suite_with_machine_profile_joins_profile_ceilings(self):
    result = run_layer(
        self.tmp,
        [
            "--suite-file",
            str(SUITES_DIR / "gemma4-e2b-cpu.json"),
            "--machine-profile",
            "sapphire-rapids",
            "--families",
            "BM_RmsNorm,BM_TransformerBlock_Prefill",
            "--seq-len",
            "1024",
            "--batch-size",
            "1",
        ],
        FakeLayerBench(),
    )

    summary = result.artifact("summary.json")
    meta = result.artifact("meta.json")
    first_layer = summary["layers"][0]
    self.assertEqual(result.exit_code, 0)
    self.assertEqual(summary["status"], "ok")
    # 4 Gemma4-E2B archetypes + the WideMLP ablation, x 2 families.
    self.assertEqual(summary["n_layers"], 10)
    self.assertEqual(len(meta["commands"]), 1)
    # The profile's physical cores per NUMA domain.
    self.assertEqual(meta["params"]["threads"], 28)
    self.assertEqual(
        Path(meta["ceilings_json"]), SAPPHIRE_RAPIDS_CEILINGS.resolve()
    )
    # 1.6e8 bytes at 244.34 GB/s take 0.655 ms of the measured 1.25 ms.
    self.assertEqual(first_layer["bound_regime"], "memory")
    self.assertAlmostEqual(
        first_layer["whole_subgraph_roofline_pct"], 52.39, places=1
    )

  def test_suite_relative_ceilings_file_is_discovered(self):
    (self.tmp / "probe.json").write_text(
        json.dumps(probe_ceilings()), encoding="utf-8"
    )
    suite_path = self.tmp / "inline.json"
    suite_path.write_text(
        json.dumps({
            "name": "inline",
            "presets": ["TinyCI"],
            "families": ["BM_RmsNorm"],
            "thread_counts": [1],
            "ceilings_file": "probe.json",
        }),
        encoding="utf-8",
    )

    result = run_layer(
        self.tmp, ["--suite-file", str(suite_path)], FakeLayerBench()
    )

    meta = result.artifact("meta.json")
    record = result.artifact("summary.json")["layers"][0]
    self.assertEqual(
        Path(meta["ceilings_json"]), (self.tmp / "probe.json").resolve()
    )
    # 1 thread: 1.28e9 FLOPs at 100 GOPS take 12.8 ms; 1.6e8 bytes at
    # 10 GB/s take 16 ms.
    self.assertEqual(record["bound_regime"], "memory")
    self.assertAlmostEqual(record["t_roofline_ms"], 16.0, places=6)

  def test_absent_suite_ceilings_file_only_skips_roofline(self):
    suite_path = self.tmp / "inline.json"
    suite_path.write_text(
        json.dumps({
            "name": "inline",
            "presets": ["TinyCI"],
            "families": ["BM_RmsNorm"],
            "ceilings_file": "absent_probe.json",
        }),
        encoding="utf-8",
    )

    result = run_layer(
        self.tmp, ["--suite-file", str(suite_path)], FakeLayerBench()
    )

    self.assertEqual(result.exit_code, 0)
    self.assertEqual(result.artifact("meta.json")["ceilings_json"], "")
    self.assertNotIn(
        "whole_subgraph_roofline_pct",
        result.artifact("summary.json")["layers"][0],
    )

  def test_kernel_timings_add_compose_delta_next_to_roofline(self):
    ceilings_path = self.tmp / "ceilings.json"
    ceilings_path.write_text(json.dumps(probe_ceilings()), encoding="utf-8")
    timings_path = self.tmp / "timings.json"
    timings_path.write_text('{"Gemma4-E2B/BM_RmsNorm": 1.0}', encoding="utf-8")

    result = run_layer(
        self.tmp,
        [
            *CASE_FLAGS,
            "--ceilings-json",
            str(ceilings_path),
            "--kernel-timings-json",
            str(timings_path),
        ],
        FakeLayerBench(),
    )

    record = result.artifact("summary.json")["layers"][0]
    self.assertEqual(record["kernel_sum_ms"], 1.0)
    self.assertEqual(record["delta_t_compose_ms"], 0.25)
    # 4 threads: 1.28e9 FLOPs at 400 GOPS take 3.2 ms; 1.6e8 bytes at
    # 25 GB/s take 6.4 ms.
    self.assertEqual(record["bound_regime"], "memory")
    self.assertAlmostEqual(record["t_roofline_ms"], 6.4, places=6)

  def test_kernel_timings_without_ceilings_still_add_compose_delta(self):
    timings_path = self.tmp / "timings.json"
    timings_path.write_text('{"Gemma4-E2B/BM_RmsNorm": 1.0}', encoding="utf-8")

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--kernel-timings-json", str(timings_path)],
        FakeLayerBench(),
    )

    record = result.artifact("summary.json")["layers"][0]
    self.assertEqual(record["kernel_sum_ms"], 1.0)
    self.assertEqual(record["delta_t_compose_ms"], 0.25)
    self.assertNotIn("whole_subgraph_roofline_pct", record)


def dump_reference(root: Path) -> Path:
  """Dumps the reference tensor of `CASE_FLAGS` with an undisturbed bench.

  Args:
    root: Scratch directory of the test.

  Returns:
    The directory holding the reference dump `CASE_DUMP_NAME`.
  """
  reference_dir = root / "reference"
  run_layer(
      root / "reference_run",
      [*CASE_FLAGS, "--dump-parity-dir", str(reference_dir)],
      FakeLayerBench(),
  )
  return reference_dir


class LayerCliParityTest(TempDirTestCase):

  def test_dump_run_writes_reference_tensor_and_record(self):
    reference_dir = self.tmp / "reference"
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp, [*CASE_FLAGS, "--dump-parity-dir", str(reference_dir)], bench
    )

    dump_path = reference_dir / CASE_DUMP_NAME
    parity = result.artifact("parity.json")
    self.assertEqual(result.exit_code, 0)
    self.assertIn(f"--dump_parity_tensors={dump_path}", bench.commands[0])
    self.assertEqual(parity["schema"], "aeb.layer.parity/1")
    self.assertIs(parity["passed"], True)
    self.assertEqual(
        parity["records"],
        [{
            "preset": "Gemma4-E2B",
            "family": "BM_RmsNorm",
            "layer": "RmsNorm",
            "batch_size": 2,
            "seq_len": 16,
            "role": "reference",
            "passed": True,
            "shape": [2, 16, 4],
            "total_elements": 128,
            "file": str(dump_path),
        }],
    )

  def test_matching_candidate_passes_comparison(self):
    reference_dir = dump_reference(self.tmp)
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--compare-parity-dir", str(reference_dir)],
        bench,
    )

    record = result.artifact("parity.json")["records"][0]
    candidate_path = result.run_dir / "parity_cand" / CASE_DUMP_NAME
    self.assertEqual(result.exit_code, 0)
    self.assertIs(result.artifact("summary.json")["parity_all_passed"], True)
    self.assertIs(record["passed"], True)
    self.assertEqual(record["mismatch_count"], 0)
    self.assertEqual(record["preset"], "Gemma4-E2B")
    self.assertEqual(
        bench.commands[0][-2:],
        [
            f"--compare_parity_tensors={reference_dir / CASE_DUMP_NAME}",
            f"--dump_parity_tensors={candidate_path}",
        ],
    )

  def test_drifted_candidate_fails_strict_comparison(self):
    reference_dir = dump_reference(self.tmp)

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--compare-parity-dir", str(reference_dir)],
        FakeLayerBench(drift=0.25),
    )

    meta = result.artifact("meta.json")
    record = result.artifact("parity.json")["records"][0]
    self.assertEqual(result.exit_code, 1)
    self.assertEqual(meta["status"], "error")
    self.assertIs(record["passed"], False)
    self.assertEqual(record["mismatch_count"], 128)
    self.assertAlmostEqual(record["max_abs_error"], 0.25, places=6)
    self.assertRegex(
        meta["errors"][0], r"^Parity failed for Gemma4-E2B/BM_RmsNorm: FAILED"
    )
    self.assertEqual(
        meta["errors"][1],
        "BM_RmsNorm/Gemma4-E2B/B:2/S:16/Threads:4/real_time: "
        "strict allclose failed",
    )

  def test_kl_only_gate_is_evaluated_without_binary_comparison(self):
    reference_dir = dump_reference(self.tmp)
    bench = FakeLayerBench(drift=0.25)

    result = run_layer(
        self.tmp,
        [
            *CASE_FLAGS,
            "--compare-parity-dir",
            str(reference_dir),
            "--parity-max-kl",
            "0.01",
        ],
        bench,
    )

    record = result.artifact("parity.json")["records"][0]
    self.assertEqual(result.exit_code, 0)
    # A uniform offset leaves every row's softmax unchanged.
    self.assertIs(record["passed"], True)
    self.assertAlmostEqual(record["kl_divergence"], 0.0, places=9)
    self.assertFalse(
        [arg for arg in bench.commands[0] if "compare_parity_tensors" in arg]
    )

  def test_kl_only_gate_fails_candidate_with_different_distribution(self):
    reference_dir = dump_reference(self.tmp)
    # One outlier moves nearly all of its row's softmax mass onto itself.
    bench = FakeLayerBench(first_tensor_value=50.0)

    result = run_layer(
        self.tmp,
        [
            *CASE_FLAGS,
            "--compare-parity-dir",
            str(reference_dir),
            "--parity-max-kl",
            "0.01",
        ],
        bench,
    )

    meta = result.artifact("meta.json")
    record = result.artifact("parity.json")["records"][0]
    self.assertEqual(result.exit_code, 1)
    self.assertIs(record["passed"], False)
    self.assertGreater(record["kl_divergence"], 0.01)
    self.assertRegex(
        meta["errors"][0], r"^Parity failed for Gemma4-E2B/BM_RmsNorm: FAILED"
    )

  def test_distribution_gates_are_forwarded_to_the_binary(self):
    reference_dir = dump_reference(self.tmp)
    bench = FakeLayerBench(drift=0.25)

    result = run_layer(
        self.tmp,
        [
            *CASE_FLAGS,
            "--compare-parity-dir",
            str(reference_dir),
            "--parity-min-cosine",
            "0.9",
            "--parity-max-l2-rel-err",
            "0.5",
        ],
        bench,
    )

    self.assertEqual(result.exit_code, 0)
    self.assertIn("--parity_min_cosine=0.9", bench.commands[0])
    self.assertIn("--parity_max_l2_rel_err=0.5", bench.commands[0])
    self.assertIs(result.artifact("parity.json")["passed"], True)

  def test_missing_reference_skips_the_case_and_fails_the_run(self):
    empty_reference_dir = self.tmp / "empty_reference"
    empty_reference_dir.mkdir()
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--compare-parity-dir", str(empty_reference_dir)],
        bench,
    )

    message = (
        "Missing reference parity dump for Gemma4-E2B/BM_RmsNorm "
        f"(B=2, S=16): expected {empty_reference_dir / CASE_DUMP_NAME}"
    )
    self.assertEqual(result.exit_code, 1)
    self.assertEqual(bench.commands, [])
    self.assertIn(f"[aeb] {message}\n", result.stderr)
    self.assertEqual(
        result.artifact("parity.json")["records"],
        [{
            "preset": "Gemma4-E2B",
            "family": "BM_RmsNorm",
            "layer": "RmsNorm",
            "batch_size": 2,
            "seq_len": 16,
            "passed": False,
            "details": message,
        }],
    )

  def test_reference_named_without_operating_point_is_still_accepted(self):
    reference_dir = dump_reference(self.tmp)
    legacy_path = reference_dir / "Gemma4-E2B__BM_RmsNorm.lmlb"
    (reference_dir / CASE_DUMP_NAME).rename(legacy_path)
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--compare-parity-dir", str(reference_dir)],
        bench,
    )

    self.assertEqual(result.exit_code, 0)
    self.assertIn(f"--compare_parity_tensors={legacy_path}", bench.commands[0])

  def test_candidate_dump_the_binary_did_not_write_fails_the_case(self):
    reference_dir = dump_reference(self.tmp)

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--compare-parity-dir", str(reference_dir)],
        FakeLayerBench(write_dumps=False),
    )

    record = result.artifact("parity.json")["records"][0]
    self.assertEqual(result.exit_code, 1)
    self.assertIs(record["passed"], False)
    self.assertRegex(
        record["details"],
        r"^Candidate parity dump was not produced for Gemma4-E2B/BM_RmsNorm",
    )

  def test_reference_dump_the_binary_did_not_write_fails_the_run(self):
    reference_dir = self.tmp / "reference"

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--dump-parity-dir", str(reference_dir)],
        FakeLayerBench(write_dumps=False),
    )

    self.assertEqual(result.exit_code, 1)
    self.assertEqual(
        result.artifact("meta.json")["errors"],
        [
            "Reference parity dump was not produced for "
            "Gemma4-E2B/BM_RmsNorm: expected "
            f"{reference_dir / CASE_DUMP_NAME}"
        ],
    )
    self.assertFalse((result.run_dir / "parity.json").exists())

  def test_unreadable_reference_dump_fails_the_case_but_keeps_results(self):
    reference_dir = self.tmp / "reference"
    reference_dir.mkdir()
    (reference_dir / CASE_DUMP_NAME).write_bytes(b"not an lmlb dump")

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--compare-parity-dir", str(reference_dir)],
        FakeLayerBench(),
    )

    record = result.artifact("parity.json")["records"][0]
    self.assertEqual(result.exit_code, 1)
    self.assertEqual(result.artifact("summary.json")["n_layers"], 1)
    self.assertIs(record["passed"], False)
    self.assertRegex(
        record["details"],
        r"^Parity dumps of Gemma4-E2B/BM_RmsNorm could not be compared: "
        "Invalid LMLB magic",
    )

  def test_invalid_reference_dump_written_by_dump_run_fails_the_run(self):
    reference_dir = self.tmp / "reference"

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--dump-parity-dir", str(reference_dir)],
        corrupt_dump_layer_bench,
    )

    self.assertEqual(result.exit_code, 1)
    self.assertRegex(
        result.artifact("meta.json")["errors"][0],
        r"^Reference parity dump of Gemma4-E2B/BM_RmsNorm is invalid: "
        "Invalid LMLB magic",
    )

  def test_reference_dump_of_earlier_run_is_discarded_when_not_rewritten(self):
    reference_dir = self.tmp / "reference"
    reference_dir.mkdir()
    stale_dump = reference_dir / CASE_DUMP_NAME
    layer.write_parity_dump(stale_dump, [1], [1.0])

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--dump-parity-dir", str(reference_dir)],
        FakeLayerBench(write_dumps=False),
    )

    self.assertEqual(result.exit_code, 1)
    self.assertFalse(stale_dump.exists())
    self.assertEqual(
        result.artifact("meta.json")["errors"],
        [
            "Reference parity dump was not produced for "
            f"Gemma4-E2B/BM_RmsNorm: expected {stale_dump}"
        ],
    )

  def test_dump_path_that_cannot_be_cleared_skips_the_case(self):
    reference_dir = self.tmp / "reference"
    # A directory in place of the dump file cannot be unlinked.
    (reference_dir / CASE_DUMP_NAME).mkdir(parents=True)
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--dump-parity-dir", str(reference_dir)],
        bench,
    )

    self.assertEqual(result.exit_code, 1)
    self.assertEqual(bench.commands, [])
    self.assertRegex(
        result.artifact("meta.json")["errors"][0],
        r"^Stale parity dump of Gemma4-E2B/BM_RmsNorm could not be removed: ",
    )

  def test_candidate_of_different_shape_fails_with_shape_mismatch(self):
    reference_dir = dump_reference(self.tmp)

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--compare-parity-dir", str(reference_dir)],
        FakeLayerBench(tensor_width=8),
    )

    record = result.artifact("parity.json")["records"][0]
    self.assertEqual(result.exit_code, 1)
    self.assertEqual(
        record["details"], "Shape mismatch: [2, 16, 4] vs [2, 16, 8]"
    )

  def test_each_parity_case_runs_in_its_own_invocation(self):
    bench = FakeLayerBench()

    result = run_layer(
        self.tmp,
        [
            "--presets",
            "Gemma4-E2B,TinyCI",
            "--families",
            "BM_RmsNorm,BM_StandardAttention_Decode",
            "--batch-size",
            "1",
            "--seq-len",
            "16",
            "--threads",
            "4",
            "--dump-parity-dir",
            str(self.tmp / "reference"),
        ],
        bench,
    )

    records = result.artifact("parity.json")["records"]
    self.assertEqual(len(bench.commands), 4)
    self.assertEqual(
        [(record["preset"], record["family"]) for record in records],
        [
            ("Gemma4-E2B", "BM_RmsNorm"),
            ("Gemma4-E2B", "BM_StandardAttention_Decode"),
            ("TinyCI", "BM_RmsNorm"),
            ("TinyCI", "BM_StandardAttention_Decode"),
        ],
    )
    # Decode families emit a single query token.
    self.assertEqual(records[1]["shape"], [1, 1, 4])


class StrictJsonArtifactsTest(TempDirTestCase):

  def test_failing_parity_artifacts_are_strict_json_with_a_reason(self):
    reference_dir = dump_reference(self.tmp)

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--compare-parity-dir", str(reference_dir)],
        FakeLayerBench(first_tensor_value=float("nan")),
    )

    self.assertEqual(result.exit_code, 1)
    parity_record = result.artifact("parity.json")["records"][0]
    summary_record = result.artifact("summary.json")["parity"][0]
    meta = result.artifact("meta.json")
    self.assertIs(parity_record["passed"], False)
    self.assertEqual(parity_record["mismatch_count"], 1)
    self.assertIsNone(parity_record["cosine_similarity"])
    self.assertIsNone(parity_record["l2_relative_error"])
    self.assertIsNone(parity_record["kl_divergence"])
    self.assertIsNone(parity_record["max_abs_error"])
    self.assertEqual(
        parity_record["details"],
        "FAILED (mismatches=1, cosine=nan, l2_rel=inf, kl=inf): "
        "1 element pair(s) hold NaN or Inf values",
    )
    self.assertEqual(summary_record, parity_record)
    self.assertIn("1 element pair(s) hold NaN or Inf", meta["errors"][0])

  def test_shape_mismatch_artifacts_are_strict_json_with_a_reason(self):
    reference_dir = dump_reference(self.tmp)

    result = run_layer(
        self.tmp,
        [*CASE_FLAGS, "--compare-parity-dir", str(reference_dir)],
        FakeLayerBench(tensor_width=8),
    )

    parity_record = result.artifact("parity.json")["records"][0]
    self.assertIsNone(parity_record["cosine_similarity"])
    self.assertIsNone(parity_record["max_rel_error"])
    self.assertRegex(parity_record["details"], "^Shape mismatch: ")

  def test_non_finite_benchmark_counters_are_read_and_written_as_null(self):
    bench = FakeLayerBench(
        counters={"GFLOP/s": float("nan"), "GB/s": float("inf")}
    )

    result = run_layer(self.tmp, CASE_FLAGS, bench)

    self.assertEqual(result.exit_code, 0)
    raw_entry = result.artifact("benchmarks.json")["benchmarks"][0]
    record = result.artifact("summary.json")["layers"][0]
    self.assertIsNone(raw_entry["GFLOP/s"])
    self.assertIsNone(raw_entry["GB/s"])
    self.assertIsNone(record["gflops"])
    self.assertIsNone(record["gbps"])
    self.assertEqual(record["real_time_ms"], 1.25)

  def test_no_artifact_contains_non_standard_json_tokens(self):
    bench = FakeLayerBench(
        counters={"GFLOP/s": float("nan"), "AllocDelta_MB": float("-inf")}
    )

    result = run_layer(self.tmp, CASE_FLAGS, bench)

    artifact_text = "".join(
        path.read_text(encoding="utf-8")
        for path in sorted(result.run_dir.glob("*.json"))
    )
    self.assertNotRegex(artifact_text, r"\bNaN\b|\bInfinity\b")


class LayerCliFailureTest(TempDirTestCase):

  def test_binary_that_cannot_be_started_fails_the_run(self):
    missing_binary = self.tmp / "does_not_exist"

    result = run_layer(self.tmp, [*CASE_FLAGS, "--binary", str(missing_binary)])

    meta = result.artifact("meta.json")
    log_text = (result.run_dir / "driver.log").read_text(encoding="utf-8")
    self.assertEqual(result.exit_code, 1)
    self.assertEqual(meta["status"], "error")
    self.assertRegex(
        meta["errors"][0],
        rf"^Failed to execute layer benchmark binary '{missing_binary}': ",
    )
    self.assertIn("[aeb] ERROR: Failed to execute layer", result.stderr)
    self.assertIn("ERROR: Failed to execute layer benchmark binary", log_text)

  def test_binary_that_cannot_be_started_aborts_remaining_cases(self):
    result = run_layer(
        self.tmp,
        [
            "--binary",
            str(self.tmp / "does_not_exist"),
            "--presets",
            "Gemma4-E2B",
            "--families",
            "BM_RmsNorm,BM_RoPE",
            "--batch-size",
            "1",
            "--seq-len",
            "16",
            "--threads",
            "4",
            "--dump-parity-dir",
            str(self.tmp / "reference"),
        ],
    )

    meta = result.artifact("meta.json")
    self.assertEqual(result.exit_code, 1)
    self.assertEqual(len(meta["commands"]), 1)
    self.assertEqual(len(meta["errors"]), 1)

  def test_crashing_binary_is_reported_with_its_exit_status(self):
    crashing_binary = self.tmp / "crashing_layer_bench"
    crashing_binary.write_text(
        "#!/bin/sh\necho 'simulated crash' >&2\nexit 3\n", encoding="utf-8"
    )
    crashing_binary.chmod(0o755)

    result = run_layer(
        self.tmp, [*CASE_FLAGS, "--binary", str(crashing_binary)]
    )

    meta = result.artifact("meta.json")
    log_text = (result.run_dir / "driver.log").read_text(encoding="utf-8")
    self.assertEqual(result.exit_code, 1)
    self.assertRegex(
        meta["errors"][0], rf"^Command failed \(exit 3\): {crashing_binary} "
    )
    self.assertIn("[aeb] Command failed (exit 3): ", result.stderr)
    self.assertIn("simulated crash\n", log_text)
    self.assertEqual(result.artifact("summary.json")["n_layers"], 0)

  def test_failing_case_does_not_stop_the_remaining_cases(self):
    bench = FakeLayerBench(exit_code=3)

    result = run_layer(
        self.tmp,
        [
            "--presets",
            "Gemma4-E2B",
            "--families",
            "BM_RmsNorm,BM_RoPE",
            "--batch-size",
            "1",
            "--seq-len",
            "16",
            "--threads",
            "4",
            "--dump-parity-dir",
            str(self.tmp / "reference"),
        ],
        bench,
    )

    self.assertEqual(result.exit_code, 1)
    self.assertEqual(len(bench.commands), 2)
    self.assertEqual(len(result.artifact("meta.json")["errors"]), 2)

  def test_binary_that_writes_no_benchmark_output_fails_the_run(self):
    result = run_layer(self.tmp, CASE_FLAGS, silent_layer_bench)

    self.assertEqual(result.exit_code, 1)
    self.assertEqual(
        result.artifact("meta.json")["errors"],
        ["Zero benchmark records matched filter."],
    )

  def test_run_without_any_benchmark_record_fails(self):
    bench = FakeLayerBench(report_text='{"context": {}, "benchmarks": []}')

    result = run_layer(self.tmp, CASE_FLAGS, bench)

    self.assertEqual(result.exit_code, 1)
    self.assertEqual(
        result.artifact("meta.json")["errors"],
        ["Zero benchmark records matched filter."],
    )

  def test_empty_benchmark_output_counts_as_no_benchmark_record(self):
    result = run_layer(self.tmp, CASE_FLAGS, FakeLayerBench(report_text=""))

    self.assertEqual(result.exit_code, 1)
    self.assertEqual(
        result.artifact("meta.json")["errors"],
        ["Zero benchmark records matched filter."],
    )

  def test_malformed_benchmark_output_is_reported_and_kept(self):
    bench = FakeLayerBench(report_text="{not json")

    result = run_layer(self.tmp, CASE_FLAGS, bench)

    kept_output = result.run_dir / "bench_000.json"
    self.assertEqual(result.exit_code, 1)
    self.assertRegex(
        result.artifact("meta.json")["errors"][0],
        rf"^Malformed benchmark JSON in {kept_output}: ",
    )
    self.assertEqual(kept_output.read_text(encoding="utf-8"), "{not json")

  def test_non_object_benchmark_output_is_reported(self):
    result = run_layer(self.tmp, CASE_FLAGS, FakeLayerBench(report_text="[]"))

    self.assertEqual(result.exit_code, 1)
    self.assertRegex(
        result.artifact("meta.json")["errors"][0],
        r"^Malformed benchmark JSON in .*: expected an object$",
    )

  def test_benchmark_output_without_a_list_of_entries_is_reported(self):
    reports = {
        "null_benchmarks": '{"benchmarks": null}',
        "object_benchmarks": '{"benchmarks": {"name": "BM_RmsNorm"}}',
        "scalar_entry": '{"benchmarks": [1.5]}',
    }
    for name, report_text in reports.items():
      with self.subTest(case=name):
        bench = FakeLayerBench(report_text=report_text)

        result = run_layer(self.tmp / name, CASE_FLAGS, bench)

        self.assertEqual(result.exit_code, 1)
        self.assertRegex(
            result.artifact("meta.json")["errors"][0],
            r"^Malformed benchmark JSON in .*: expected 'benchmarks' to be "
            r"a list of objects$",
        )

  def test_benchmark_flagged_by_the_binary_fails_the_run(self):
    bench = FakeLayerBench(
        counters={"error_occurred": True, "error_message": "out of memory"}
    )

    result = run_layer(self.tmp, CASE_FLAGS, bench)

    self.assertEqual(result.exit_code, 1)
    self.assertEqual(
        result.artifact("meta.json")["errors"],
        ["BM_RmsNorm/Gemma4-E2B/B:2/S:16/Threads:4/real_time: out of memory"],
    )
    self.assertRegex(result.stderr, r"1 benchmarks recorded \[error\]")


@unittest.skipUnless(shutil.which("make"), "GNU make not installed")
class LayerMakefileTest(unittest.TestCase):
  """Tests `MACHINE=` enforcement and forwarding in `Makefile` targets."""

  def _run_make(
      self,
      args: Sequence[str],
      cwd: Path = LAYER_ROOT,
      extra_env: Mapping[str, str] | None = None,
  ) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"MACHINE", "MACHINE_PROFILE", "MAKEFLAGS", "MFLAGS"}
    }
    if extra_env:
      env.update(extra_env)
    return subprocess.run(
        ["make", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

  def test_layer_bench_requires_machine_argument(self):
    for label, args in (
        ("omitted", ["-n", "bench"]),
        ("empty", ["-n", "MACHINE=", "bench"]),
        ("whitespace", ["-n", "MACHINE=   ", "bench"]),
    ):
      with self.subTest(case=label):
        proc = self._run_make(args)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("MACHINE is required", proc.stderr)

  def test_root_layer_bench_does_not_inject_default_machine(self):
    proc = self._run_make(["-n", "layer-bench"], cwd=REPO_ROOT)

    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("MACHINE is required", proc.stderr)

  def test_layer_bench_forwards_machine_and_default_profile(self):
    proc = self._run_make(["-n", "MACHINE=sapphire-rapids", "bench"])

    self.assertEqual(proc.returncode, 0, msg=proc.stderr)
    self.assertIn("AEB_MACHINE=sapphire-rapids", proc.stdout)
    self.assertIn("--machine-profile sapphire-rapids", proc.stdout)

  def test_layer_bench_accepts_machine_from_environment(self):
    proc = self._run_make(
        ["-n", "bench"], extra_env={"MACHINE": "sapphire-rapids"}
    )

    self.assertEqual(proc.returncode, 0, msg=proc.stderr)
    self.assertIn("AEB_MACHINE=sapphire-rapids", proc.stdout)
    self.assertIn("--machine-profile sapphire-rapids", proc.stdout)

  def test_layer_bench_allows_empty_machine_profile_escape_hatch(self):
    proc = self._run_make(
        ["-n", "MACHINE=custom-host", "MACHINE_PROFILE=", "bench"]
    )

    self.assertEqual(proc.returncode, 0, msg=proc.stderr)
    self.assertIn("AEB_MACHINE=custom-host", proc.stdout)
    self.assertNotIn("--machine-profile", proc.stdout)

  def test_root_layer_bench_forwards_explicit_machine_and_empty_profile(self):
    proc = self._run_make(
        ["-n", "MACHINE=custom-host", "MACHINE_PROFILE=", "layer-bench"],
        cwd=REPO_ROOT,
    )

    self.assertEqual(proc.returncode, 0, msg=proc.stderr)
    self.assertIn("AEB_MACHINE=custom-host", proc.stdout)
    self.assertNotIn("--machine-profile", proc.stdout)

  def test_layer_help_succeeds_without_machine(self):
    proc = self._run_make(["help"])

    self.assertEqual(proc.returncode, 0, msg=proc.stderr)
    self.assertIn("make MACHINE=<m> bench", proc.stdout)
    self.assertNotIn("- MACHINE=\n", proc.stdout)


if __name__ == "__main__":
  unittest.main()
