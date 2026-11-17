r"""Framework-neutral layer-level microbenchmark, Roofline, and parity runner.

  python3 -m aeb.layer --framework litert-tensor --label xnnpack \
      --backend xnnpack --presets Gemma4-E2B --batch-size 1 --seq-len 1024 \
      --threads 32 --min-time 0.2s --ceilings-json /path/to/ceilings.json \
      --dump-parity-dir /results/.../parity_ref

  python3 -m aeb.layer --framework litert-tensor --label ynnpack \
      --backend ynnpack --presets Gemma4-E2B --batch-size 1 --seq-len 1024 \
      --threads 32 --min-time 0.2s --ceilings-json /path/to/ceilings.json \
      --compare-parity-dir /results/.../parity_ref

Invokes a compiled `layer_ml_bench` binary (`layer_bench`) implementing the
`LayerBackend` / `LayerBenchmarkRunner` C++ contract, collects Google Benchmark
JSON telemetry (latency, GFLOP/s, GB/s, arithmetic intensity, workspace memory),
joins against Level 1 hardware ceiling probes (`ceilings_<machine>.json`) to
compute Whole-Subgraph Practical Roofline Utilization (`Efficiency_L`) and
optional composition deltas (`DeltaT_compose`), and dumps or compares `LMLB`
(`0x4C4D4C42`) binary output tensors (allclose, cosine similarity, relative L2
error, and softmax KL-divergence).
"""

from __future__ import annotations

import argparse
import array
from collections.abc import Callable, Mapping, MutableMapping, Sequence
import dataclasses
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from typing import Any

from . import catalog
from . import common

LMLB_MAGIC = 0x4C4D4C42  # "LMLB" in little-endian

DEFAULT_FAMILIES = catalog.DEFAULT_SUITE_FAMILIES
DEFAULT_PRESETS = ("Gemma4-E2B", "Gemma4-E2B-WideMLP")

_LMLB_MAX_RANK = 16
_KL_EPSILON = 1e-12
_INSTALL_BIN_DIR = Path("/opt/litert-tensor/bin")
_FP32_DTYPES = ("fp32", "float32")
_QUANT_PROFILE_DTYPE_FIELDS = (
    "weight_dtype",
    "activation_dtype",
    "accumulator_dtype",
    "kv_dtype",
)
_BLOCK_FAMILIES = ("BM_TransformerBlock_Prefill", "BM_TransformerBlock_Decode")
_CORE_COUNT_SUFFIX = re.compile(r"_(\d+)c$")

_EXIT_OK = 0
_EXIT_FAILED = 1
_EXIT_USAGE = 2


class _UsageError(Exception):
  """An invalid flag value or unusable input file (exit code 2)."""


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
  """Runs a `layer_bench` command line and captures its text output.

  Args:
    command: Executable path followed by its arguments.

  Returns:
    The completed process; a non-zero exit status is reported, not raised.

  Raises:
    OSError: If the executable cannot be started.
  """
  return subprocess.run(
      list(command), capture_output=True, text=True, check=False
  )


def _process_environment() -> Mapping[str, str]:
  """Returns the environment variables of the current process."""
  return os.environ


@dataclasses.dataclass(frozen=True)
class RunnerHooks:
  """Host-side collaborators of the layer runner, replaceable in tests.

  Attributes:
    new_run_dir: Creates and returns a fresh run directory, given the run kind,
      framework, and label.
    write_sysinfo: Writes host system information as JSON to the given path.
    run_command: Runs one `layer_bench` command line and returns the completed
      process with captured text output; raises `OSError` when the executable
      cannot be started.
    environ: Environment variables consulted for the machine label and for
      `layer_bench` binary discovery.
    install_bin_dir: Directory searched for an installed `layer_bench` before
      `PATH`.
    utc_now: Returns the current UTC time as an ISO 8601 string.
    build_info: Returns the build provenance recorded with the results.
  """

  new_run_dir: Callable[[str, str, str], Path] = common.new_run_dir
  write_sysinfo: Callable[[Path], None] = common.write_sysinfo
  run_command: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = (
      _run_command
  )
  environ: Mapping[str, str] = dataclasses.field(
      default_factory=_process_environment
  )
  install_bin_dir: Path = _INSTALL_BIN_DIR
  utc_now: Callable[[], str] = common.utc_now
  build_info: Callable[[], dict[str, Any]] = common.build_info


@dataclasses.dataclass(frozen=True)
class _ParityGate:
  """Numerical parity thresholds for candidate tensor dumps.

  Attributes:
    rtol: Relative tolerance of the strict elementwise allclose gate.
    atol: Absolute tolerance of the strict elementwise allclose gate.
    min_cosine: Minimum cosine similarity; `<= 0` disables the gate.
    max_l2_rel_err: Maximum relative L2 error; `<= 0` disables the gate.
    max_kl: Maximum mean row-wise softmax KL-divergence; `<= 0` disables the
      gate.
  """

  rtol: float
  atol: float
  min_cosine: float
  max_l2_rel_err: float
  max_kl: float

  @property
  def kl_only(self) -> bool:
    """Whether the KL-divergence gate is the only distribution gate enabled."""
    return (
        self.max_kl > 0.0
        and self.min_cosine <= 0.0
        and self.max_l2_rel_err <= 0.0
    )


@dataclasses.dataclass(frozen=True)
class _ElementStats:
  """Elementwise error accumulators of a reference/candidate tensor pair.

  Attributes:
    mismatches: Pairs outside `atol + rtol * |ref|`, plus non-finite pairs.
    non_finite: Pairs in which either value is NaN or infinite.
    max_abs: Largest absolute error; infinite when any pair is non-finite.
    max_rel: Largest relative error; infinite when any pair is non-finite.
    dot: Sum of `ref * cand` over the finite pairs.
    ref_sq: Sum of `ref**2` over the finite pairs.
    cand_sq: Sum of `cand**2` over the finite pairs.
    diff_sq: Sum of `(ref - cand)**2` over the finite pairs.
  """

  mismatches: int
  non_finite: int
  max_abs: float
  max_rel: float
  dot: float
  ref_sq: float
  cand_sq: float
  diff_sq: float

  @property
  def cosine_similarity(self) -> float:
    """Cosine similarity of the tensors; NaN when any pair is non-finite.

    Two all-zero tensors count as identical (1.0); an all-zero tensor against
    a non-zero one counts as orthogonal (0.0).
    """
    if self.non_finite:
      return math.nan
    ref_norm = math.sqrt(self.ref_sq)
    cand_norm = math.sqrt(self.cand_sq)
    if ref_norm == 0.0 and cand_norm == 0.0:
      return 1.0
    if ref_norm == 0.0 or cand_norm == 0.0:
      return 0.0
    return max(-1.0, min(1.0, self.dot / (ref_norm * cand_norm)))

  @property
  def l2_relative_error(self) -> float:
    """Relative L2 error of the candidate.

    Infinite when any pair is non-finite, or when the candidate differs from
    an all-zero reference.
    """
    if self.non_finite:
      return math.inf
    ref_norm = math.sqrt(self.ref_sq)
    diff_norm = math.sqrt(self.diff_sq)
    if ref_norm == 0.0:
      return 0.0 if diff_norm == 0.0 else math.inf
    return diff_norm / ref_norm


@dataclasses.dataclass(frozen=True)
class _BenchCase:
  """One `layer_bench` invocation.

  Attributes:
    preset: Preset of a single parity case; empty for a whole-sweep case.
    family: Benchmark family of a single parity case; empty for a whole-sweep
      case.
    batch_size: Batch size of a single parity case; 0 for a whole-sweep case.
    seq_len: Sequence length of a single parity case; 0 for a whole-sweep case.
    filter_regex: Google Benchmark filter selecting the benchmarks to run.
  """

  preset: str
  family: str
  batch_size: int
  seq_len: int
  filter_regex: str

  @property
  def dump_stem(self) -> str:
    """File name stem of the case's tensor dump."""
    return f"{self.preset}__{self.family}__B{self.batch_size}_S{self.seq_len}"

  def identity(self) -> dict[str, Any]:
    """Returns the fields identifying the case in a parity record."""
    return {
        "preset": self.preset,
        "family": self.family,
        "layer": self.family.removeprefix("BM_"),
        "batch_size": self.batch_size,
        "seq_len": self.seq_len,
    }


@dataclasses.dataclass(frozen=True)
class _ParityFiles:
  """Tensor dump files exchanged with `layer_bench` for one case.

  Attributes:
    dump: File the binary writes its output tensor to; `None` when the run
      neither dumps nor compares tensors.
    reference: Existing reference dump that `dump` is compared against; `None`
      unless the run compares tensors.
  """

  dump: Path | None = None
  reference: Path | None = None


@dataclasses.dataclass(frozen=True)
class _Sweep:
  """Benchmarks selected for a run and their operating points.

  Attributes:
    presets: Preset names to benchmark, in execution order.
    families: Benchmark family names to benchmark, in execution order.
    batch_sizes: Batch sizes to sweep.
    seq_lengths: Sequence (KV context) lengths to sweep.
    threads: Worker thread count of every benchmark.
  """

  presets: Sequence[str]
  families: Sequence[str]
  batch_sizes: Sequence[int]
  seq_lengths: Sequence[int]
  threads: int


@dataclasses.dataclass(frozen=True)
class _RunPlan:
  """Everything resolved from the flags before any benchmark runs.

  Attributes:
    suite_data: Resolved suite the binary registers its benchmarks from.
    dtype: Weight/compute dtype label of the run.
    backend: Kernel backend passed to the binary.
    binary: Path of the `layer_bench` executable.
    sweep: Benchmarks to run and their operating points.
    min_time: Google Benchmark minimum measurement time per benchmark.
    seed: Seed of the binary's synthetic tensors.
    wipe_caches: Whether the binary evicts CPU caches between iterations.
    gate: Numerical parity thresholds.
    dump_dir: Directory receiving reference tensor dumps; `None` unless the run
      dumps tensors.
    compare_dir: Directory holding reference tensor dumps; `None` unless the run
      compares tensors.
    ceilings_path: Hardware ceilings file in effect; empty when there is none.
    ceilings: Parsed hardware ceilings; `None` when there is no ceilings file.
    kernel_timings: Summed kernel milliseconds by benchmark key; `None` when no
      kernel timings file was passed.
  """

  suite_data: dict[str, Any]
  dtype: str
  backend: str
  binary: str
  sweep: _Sweep
  min_time: str
  seed: int
  wipe_caches: bool
  gate: _ParityGate
  dump_dir: Path | None
  compare_dir: Path | None
  ceilings_path: str
  ceilings: dict[str, Any] | None
  kernel_timings: dict[str, float] | None


@dataclasses.dataclass
class _RunLog:
  """Outputs accumulated over the `layer_bench` invocations of a run.

  Attributes:
    raw_benchmarks: Raw Google Benchmark entries, in execution order.
    context: Google Benchmark `context` object of the first parsed output.
    parity_records: Per-case reference or comparison records.
    commands: Command lines handed to the binary, in execution order.
    errors: Human-readable failures; the run succeeded when this is empty.
  """

  raw_benchmarks: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  context: dict[str, Any] = dataclasses.field(default_factory=dict)
  parity_records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  commands: list[list[str]] = dataclasses.field(default_factory=list)
  errors: list[str] = dataclasses.field(default_factory=list)


def read_parity_dump_header(path: Path) -> tuple[list[int], int]:
  """Validates an LMLB dump header and file size without unpacking floats.

  Args:
    path: Path to an `.lmlb` binary tensor dump.

  Returns:
    Tuple `(shape, n_elems)` where `shape` is the list of tensor dimensions
    and `n_elems` is the total element count.

  Raises:
    ValueError: If magic, rank, dimensions, or file size are invalid.
    OSError: If `path` cannot be opened or stat'd.
  """
  file_size = path.stat().st_size
  if file_size < 8:
    raise ValueError(f"Parity dump {path} is too short ({file_size} bytes)")
  with open(path, "rb") as dump_file:
    magic, rank = struct.unpack("<II", dump_file.read(8))
    if magic != LMLB_MAGIC:
      raise ValueError(
          f"Invalid LMLB magic in {path}: 0x{magic:08X} != 0x{LMLB_MAGIC:08X}"
      )
    if rank == 0 or rank > _LMLB_MAX_RANK:
      raise ValueError(f"Invalid tensor rank in {path}: {rank}")
    header_bytes = 8 + 8 * rank
    if file_size < header_bytes:
      raise ValueError(f"Truncated shape header in {path}")
    shape = list(struct.unpack(f"<{rank}q", dump_file.read(8 * rank)))
  n_elems = 1
  for dim in shape:
    if dim <= 0:
      raise ValueError(f"Non-positive dimension {dim} in {path}")
    n_elems *= dim
  expected_bytes = header_bytes + 4 * n_elems
  if file_size != expected_bytes:
    raise ValueError(
        f"Payload size mismatch in {path}: got {file_size} bytes, "
        f"expected {expected_bytes} for shape {shape}"
    )
  return shape, n_elems


def _read_parity_dump_array(
    path: Path,
) -> tuple[list[int], array.array[float]]:
  """Reads an LMLB dump into `(shape, array('f'))` without boxing floats."""
  shape, n_elems = read_parity_dump_header(path)
  header_bytes = 8 + 8 * len(shape)
  values = array.array("f")
  with open(path, "rb") as dump_file:
    dump_file.seek(header_bytes)
    values.fromfile(dump_file, n_elems)
  if sys.byteorder != "little":
    values.byteswap()
  return shape, values


def read_parity_dump(path: Path) -> tuple[list[int], list[float]]:
  """Reads an LMLB (0x4C4D4C42) binary tensor dump into `(shape, values)`.

  Args:
    path: Path to an `.lmlb` binary tensor dump produced by `layer_bench` or
      `write_parity_dump`.

  Returns:
    Tuple of `(shape, values)` where `shape` is a list of positive integer
    dimensions and `values` is a flat list of FP32 floats.

  Raises:
    ValueError: If magic, rank, dimensions, or payload length are invalid.
    OSError: If `path` cannot be read.
  """
  shape, values = _read_parity_dump_array(path)
  return shape, values.tolist()


def write_parity_dump(
    path: Path, shape: Sequence[int], values: Sequence[float]
) -> None:
  """Writes an LMLB (0x4C4D4C42) binary tensor dump file.

  Args:
    path: Destination `.lmlb` file path (parent directories are created).
    shape: Positive integer tensor dimensions with rank in `1..16`.
    values: Flat sequence of FP32 values matching the product of `shape`.

  Raises:
    ValueError: If `shape` rank/dimensions or `len(values)` are invalid.
  """
  rank = len(shape)
  if rank == 0 or rank > _LMLB_MAX_RANK:
    raise ValueError(f"Invalid rank {rank}")
  n_elems = 1
  for dim in shape:
    if dim <= 0:
      raise ValueError(f"Non-positive dimension {dim}")
    n_elems *= dim
  if len(values) != n_elems:
    raise ValueError(
        f"len(values)={len(values)} does not match shape product {n_elems}"
    )
  path.parent.mkdir(parents=True, exist_ok=True)
  header = struct.pack(f"<II{rank}q", LMLB_MAGIC, rank, *shape)
  payload = struct.pack(f"<{n_elems}f", *values)
  path.write_bytes(header + payload)


def _softmax(row: Sequence[float]) -> list[float]:
  """Returns the softmax of `row`, shifted by its maximum for stability."""
  row_max = max(row)
  exponentials = [math.exp(value - row_max) for value in row]
  total = sum(exponentials)
  return [value / total for value in exponentials]


def _mean_row_softmax_kl(
    ref_vals: Sequence[float], cand_vals: Sequence[float], row_width: int
) -> float:
  """Computes mean row-wise KL(softmax(ref) || softmax(cand)) across rows.

  Args:
    ref_vals: Flat reference tensor.
    cand_vals: Flat candidate tensor of the same length.
    row_width: Size of the trailing tensor dimension that forms one row.

  Returns:
    The mean KL-divergence over rows, or 0.0 when the tensor is empty or its
    length is not a multiple of `row_width`.
  """
  if row_width <= 0 or not ref_vals or len(ref_vals) % row_width != 0:
    return 0.0
  n_rows = len(ref_vals) // row_width
  total_kl = 0.0
  for row_index in range(n_rows):
    start = row_index * row_width
    ref_probs = _softmax(ref_vals[start : start + row_width])
    cand_probs = _softmax(cand_vals[start : start + row_width])
    row_kl = 0.0
    for ref_prob, cand_prob in zip(ref_probs, cand_probs, strict=True):
      if ref_prob > _KL_EPSILON:
        row_kl += ref_prob * math.log(ref_prob / max(cand_prob, _KL_EPSILON))
    total_kl += max(0.0, row_kl)
  return total_kl / n_rows


def _accumulate_element_stats(
    ref_vals: Sequence[float],
    cand_vals: Sequence[float],
    rtol: float,
    atol: float,
) -> _ElementStats:
  """Accumulates elementwise errors of a candidate against a reference.

  Args:
    ref_vals: Flat reference tensor.
    cand_vals: Flat candidate tensor of the same length.
    rtol: Relative tolerance of the allclose check.
    atol: Absolute tolerance of the allclose check.

  Returns:
    Error accumulators over all element pairs.
  """
  mismatches = 0
  non_finite = 0
  max_abs = 0.0
  max_rel = 0.0
  dot = 0.0
  ref_sq = 0.0
  cand_sq = 0.0
  diff_sq = 0.0
  for ref, cand in zip(ref_vals, cand_vals, strict=True):
    if not (math.isfinite(ref) and math.isfinite(cand)):
      non_finite += 1
      continue
    diff = ref - cand
    dot += ref * cand
    ref_sq += ref * ref
    cand_sq += cand * cand
    diff_sq += diff * diff
    abs_err = abs(diff)
    rel_err = (abs_err / abs(ref)) if abs(ref) > 1e-12 else 0.0
    if abs_err > max_abs:
      max_abs = abs_err
    if rel_err > max_rel:
      max_rel = rel_err
    if abs_err > atol + rtol * abs(ref):
      mismatches += 1
  if non_finite:
    max_abs = math.inf
    max_rel = math.inf
  return _ElementStats(
      mismatches=mismatches + non_finite,
      non_finite=non_finite,
      max_abs=max_abs,
      max_rel=max_rel,
      dot=dot,
      ref_sq=ref_sq,
      cand_sq=cand_sq,
      diff_sq=diff_sq,
  )


def _passes_parity_gates(
    stats: _ElementStats,
    kl_div: float,
    min_cosine: float,
    max_l2_rel_err: float,
    max_kl_divergence: float,
) -> bool:
  """Decides whether a candidate tensor passes the enabled parity gates.

  Args:
    stats: Elementwise accumulators of the compared tensor pair.
    kl_div: Mean row-wise softmax KL-divergence.
    min_cosine: Minimum cosine similarity gate (`<= 0` disables).
    max_l2_rel_err: Maximum relative L2 error gate (`<= 0` disables).
    max_kl_divergence: Maximum KL-divergence gate (`<= 0` disables).

  Returns:
    `False` when any metric is NaN or infinite. Otherwise whether every
    enabled distribution gate holds, or, with all of them disabled, whether
    no element is outside the allclose tolerance.
  """
  cosine_sim = stats.cosine_similarity
  l2_rel_err = stats.l2_relative_error
  if not (
      math.isfinite(cosine_sim)
      and math.isfinite(l2_rel_err)
      and math.isfinite(kl_div)
  ):
    return False
  if min_cosine > 0.0 or max_l2_rel_err > 0.0 or max_kl_divergence > 0.0:
    cosine_ok = min_cosine <= 0.0 or cosine_sim >= min_cosine
    l2_ok = max_l2_rel_err <= 0.0 or l2_rel_err <= max_l2_rel_err
    kl_ok = max_kl_divergence <= 0.0 or kl_div <= max_kl_divergence
    return cosine_ok and l2_ok and kl_ok
  return stats.mismatches == 0


def _failure_details(stats: _ElementStats, kl_div: float) -> str:
  """Describes a failed comparison, including why metrics are undefined.

  Args:
    stats: Elementwise accumulators of the compared tensor pair.
    kl_div: Mean row-wise softmax KL-divergence.

  Returns:
    The metrics summary, followed by the reason when a metric is NaN or
    infinite (non-finite tensor values, or an all-zero reference).
  """
  l2_rel_err = stats.l2_relative_error
  details = (
      f"FAILED (mismatches={stats.mismatches}, "
      f"cosine={stats.cosine_similarity:.8f}, l2_rel={l2_rel_err:.6g}, "
      f"kl={kl_div:.6g})"
  )
  if stats.non_finite:
    pairs = f"{stats.non_finite} element pair(s)"
    return f"{details}: {pairs} hold NaN or Inf values"
  if math.isinf(l2_rel_err):
    return (
        f"{details}: reference tensor is all zeros, so relative L2 error is "
        "unbounded"
    )
  return details


def _shape_mismatch_result(
    ref_shape: Sequence[int],
    cand_shape: Sequence[int],
    ref_elems: int,
    cand_elems: int,
) -> dict[str, Any]:
  """Builds the failed comparison result of two differently shaped dumps.

  Args:
    ref_shape: Shape of the reference tensor.
    cand_shape: Shape of the candidate tensor.
    ref_elems: Element count of the reference tensor.
    cand_elems: Element count of the candidate tensor.

  Returns:
    A failed result whose metrics are NaN or infinite and whose `details`
    name both shapes.
  """
  return {
      "passed": False,
      "ref_shape": ref_shape,
      "cand_shape": cand_shape,
      "total_elements": ref_elems,
      "mismatch_count": max(1, ref_elems, cand_elems),
      "max_abs_error": math.inf,
      "max_rel_error": math.inf,
      "cosine_similarity": math.nan,
      "l2_relative_error": math.inf,
      "kl_divergence": math.inf,
      "details": f"Shape mismatch: {ref_shape} vs {cand_shape}",
  }


def compare_parity_dumps(
    ref_path: Path,
    cand_path: Path,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    min_cosine: float = 0.0,
    max_l2_rel_err: float = 0.0,
    max_kl_divergence: float = 0.0,
) -> dict[str, Any]:
  """Compares two LMLB dumps (allclose, cosine similarity, L2 error, KL).

  Args:
    ref_path: Reference `.lmlb` dump path.
    cand_path: Candidate `.lmlb` dump path.
    rtol: Relative tolerance for strict elementwise allclose.
    atol: Absolute tolerance for strict elementwise allclose.
    min_cosine: Optional minimum cosine similarity gate (`<= 0` disables).
    max_l2_rel_err: Optional maximum relative L2 error gate (`<= 0` disables).
    max_kl_divergence: Optional maximum row-wise softmax KL gate (`<= 0`
      disables).

  Returns:
    Dictionary containing `passed`, error/similarity metrics, and `details`.
    Metrics are NaN or infinite when shapes differ, a tensor holds non-finite
    values, or the reference is all zeros; `details` then states the reason.

  Raises:
    ValueError: If either dump has an invalid header or payload.
    OSError: If either dump cannot be read.
  """
  ref_shape, ref_vals = _read_parity_dump_array(ref_path)
  cand_shape, cand_vals = _read_parity_dump_array(cand_path)
  if ref_shape != cand_shape:
    return _shape_mismatch_result(
        ref_shape, cand_shape, len(ref_vals), len(cand_vals)
    )
  stats = _accumulate_element_stats(ref_vals, cand_vals, rtol, atol)
  kl_div = math.inf
  if not stats.non_finite:
    kl_div = _mean_row_softmax_kl(ref_vals, cand_vals, ref_shape[-1])
  passed = _passes_parity_gates(
      stats, kl_div, min_cosine, max_l2_rel_err, max_kl_divergence
  )
  return {
      "passed": passed,
      "shape": ref_shape,
      "total_elements": len(ref_vals),
      "mismatch_count": stats.mismatches,
      "max_abs_error": stats.max_abs,
      "max_rel_error": stats.max_rel,
      "cosine_similarity": stats.cosine_similarity,
      "l2_relative_error": stats.l2_relative_error,
      "kl_divergence": kl_div,
      "rtol": rtol,
      "atol": atol,
      "min_cosine": min_cosine,
      "max_l2_rel_err": max_l2_rel_err,
      "max_kl_divergence": max_kl_divergence,
      "details": "PASSED" if passed else _failure_details(stats, kl_div),
  }


def _interpolate_ceiling(
    entry: Mapping[str, Any], threads: int, numa_threads: int
) -> float:
  """Piecewise-interpolates between 1-core and multi-core (`_<N>c`) ceilings.

  Args:
    entry: Probe entry with a `single_core` value and `numa_node*` values whose
      `_<N>c` suffix gives their thread count.
    threads: Thread count to interpolate the ceiling at.
    numa_threads: Thread count assumed for `numa_node*` keys without a suffix.

  Returns:
    The interpolated ceiling, clamped to the lowest and highest measured
    thread counts; 0.0 when the entry holds no measurements.
  """
  knots_by_threads: dict[int, float] = {}
  single_core = float(entry.get("single_core", 0.0))
  if single_core > 0.0:
    knots_by_threads[1] = single_core
  for key, value in entry.items():
    if key.startswith("numa_node") and not key.endswith("_stats"):
      match = _CORE_COUNT_SUFFIX.search(key)
      knot_threads = int(match.group(1)) if match else max(numa_threads, 2)
      knots_by_threads[knot_threads] = float(value)
  if not knots_by_threads:
    return 0.0
  knots = sorted(knots_by_threads.items())
  if threads <= knots[0][0]:
    return knots[0][1]
  if threads >= knots[-1][0]:
    return knots[-1][1]
  for (low_threads, low), (high_threads, high) in itertools.pairwise(knots):
    if threads <= high_threads:
      fraction = (threads - low_threads) / float(high_threads - low_threads)
      return low + fraction * (high - low)
  return knots[-1][1]


def load_hardware_ceilings(path: Path) -> dict[str, Any]:
  """Loads a Level 1 `ceilings_<machine>.json` probe file.

  Args:
    path: Path to the JSON ceilings file.

  Returns:
    Parsed ceilings dictionary.

  Raises:
    OSError: If `path` cannot be read.
    ValueError: If the file is not a JSON object.
  """
  return catalog.load_json_object(path)


def load_kernel_timings(path: Path) -> dict[str, float]:
  """Loads a Level 1 summed kernel timings JSON file.

  Args:
    path: JSON object mapping `<preset>/<family>`, `<preset>/<layer>`,
      `<family>`, or `<layer>` keys to summed kernel milliseconds.

  Returns:
    Summed kernel milliseconds by key.

  Raises:
    OSError: If `path` cannot be read.
    ValueError: If the file is not a JSON object of finite numbers.
  """
  timings: dict[str, float] = {}
  for key, value in catalog.load_json_object(path).items():
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number or not math.isfinite(value):
      raise ValueError(
          f"{path}: kernel timing {key!r} must be a finite number, got "
          f"{value!r}"
      )
    timings[key] = float(value)
  return timings


def _compute_ceiling_keys(dtype: str) -> tuple[str, ...]:
  """Returns compute-ceiling probe keys for a dtype, by decreasing priority."""
  dtype_norm = dtype.strip().lower()
  if dtype_norm.startswith(("bf16", "fp16")):
    return ("amx_bf16_tdpbf16ps", "avx512_fp32_fma")
  if dtype_norm.startswith(
      ("int8", "qd8", "qc8", "int4", "qb4", "int2", "qc2")
  ):
    return (
        "amx_int8_tdpbssd",
        "avx512_vnni_int8_vpdpbusd",
        "avx_vnni_256_int8_vpdpbusd",
        "avx512_fp32_fma",
    )
  return ("avx512_fp32_fma",)


def select_practical_ceilings(
    ceilings: Mapping[str, Any], threads: int, dtype: str = "fp32"
) -> tuple[float, float]:
  """Returns `(c_prac_gops, bw_prac_gbps)` from Level 1 probe ceilings JSON.

  Args:
    ceilings: Parsed Level 1 hardware ceilings dictionary.
    threads: Active worker thread count for piecewise ceiling interpolation.
    dtype: Weight/compute dtype string (e.g., `fp32`, `bf16`, `int4_block`).

  Returns:
    Tuple `(c_prac_gops, bw_prac_gbps)` in GOPS and GB/s.
  """
  numa_threads = int(ceilings.get("threads_numa_node0", 28))
  compute = ceilings.get("compute_ceilings_gops", {})
  c_prac_gops = 0.0
  for key in _compute_ceiling_keys(dtype):
    entry = compute.get(key)
    if isinstance(entry, dict):
      c_prac_gops = _interpolate_ceiling(entry, threads, numa_threads)
      if c_prac_gops > 0.0:
        break

  bandwidth = ceilings.get("bandwidth_ceilings_gbps", {})
  dram = bandwidth.get("ddr5_dram") or bandwidth.get("dram") or {}
  dram_entry = dram.get("read_only") or dram.get("stream_triad") or {}
  bw_prac_gbps = (
      _interpolate_ceiling(dram_entry, threads, numa_threads)
      if isinstance(dram_entry, dict)
      else 0.0
  )
  return c_prac_gops, bw_prac_gbps


def _compose_delta(
    layer_record: Mapping[str, Any], kernel_timings: Mapping[str, float]
) -> dict[str, float]:
  """Computes the composition overhead of a layer over its kernels.

  Args:
    layer_record: Normalized layer record from `parse_benchmark_entry`.
    kernel_timings: Summed kernel milliseconds keyed by `<preset>/<family>`,
      `<preset>/<layer>`, `<family>`, or `<layer>` (first match wins).

  Returns:
    `kernel_sum_ms` and `delta_t_compose_ms` (measured minus summed kernel
    time), or an empty dictionary when no key matches the record.
  """
  family = layer_record.get("family", "")
  layer_name = layer_record.get("layer", "")
  preset = layer_record.get("preset", "")
  real_ms = float(layer_record.get("real_time_ms", 0.0))
  keys = (f"{preset}/{family}", f"{preset}/{layer_name}", family, layer_name)
  for key in keys:
    kernel_sum = kernel_timings.get(key)
    if kernel_sum is not None:
      kernel_sum_ms = float(kernel_sum)
      return {
          "kernel_sum_ms": kernel_sum_ms,
          "delta_t_compose_ms": real_ms - kernel_sum_ms,
      }
  return {}


def compute_subgraph_roofline(
    layer_record: Mapping[str, Any],
    ceilings: Mapping[str, Any],
    dtype: str = "fp32",
    kernel_timings: Mapping[str, float] | None = None,
) -> dict[str, Any]:
  """Computes Whole-Subgraph Practical Roofline Utilization and DeltaT_compose.

  Args:
    layer_record: Normalized layer record from `parse_benchmark_entry`.
    ceilings: Parsed Level 1 hardware ceilings dictionary.
    dtype: Weight/compute dtype used to pick practical compute ceilings.
    kernel_timings: Optional mapping from layer/preset key to summed kernel ms;
      `None` or empty leaves `delta_t_compose_ms` unset.

  Returns:
    Dictionary with practical ceilings, `t_roofline_ms`,
    `whole_subgraph_roofline_pct`, `bound_regime`, and `delta_t_compose_ms`.
    The Roofline fields are `None` when a ceiling or the measured time is not
    positive; `delta_t_compose_ms` is `None` without a matching kernel timing.
  """
  threads = int(layer_record.get("threads", 1))
  real_ms = float(layer_record.get("real_time_ms", 0.0))
  flops = float(layer_record.get("theoretical_flops", 0.0))
  mem_bytes = float(layer_record.get("theoretical_bytes", 0.0))

  c_prac_gops, bw_prac_gbps = select_practical_ceilings(
      ceilings, threads=threads, dtype=dtype
  )
  roofline: dict[str, Any] = {
      "c_prac_gops": c_prac_gops,
      "bw_prac_gbps": bw_prac_gbps,
      "t_roofline_ms": None,
      "whole_subgraph_roofline_pct": None,
      "bound_regime": None,
      "delta_t_compose_ms": None,
  }
  if c_prac_gops > 0.0 and bw_prac_gbps > 0.0 and real_ms > 0.0:
    t_comp_ms = (flops / (c_prac_gops * 1e9)) * 1e3
    t_mem_ms = (mem_bytes / (bw_prac_gbps * 1e9)) * 1e3
    t_min_ms = max(t_comp_ms, t_mem_ms)
    roofline["t_roofline_ms"] = t_min_ms
    roofline["whole_subgraph_roofline_pct"] = (t_min_ms / real_ms) * 100.0
    roofline["bound_regime"] = "compute" if t_comp_ms >= t_mem_ms else "memory"
  if kernel_timings:
    roofline.update(_compose_delta(layer_record, kernel_timings))
  return roofline


def _block_latency_by_point(
    layers: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int, int], dict[str, float]]:
  """Groups successful TransformerBlock latencies by operating point.

  Args:
    layers: Normalized layer benchmark records.

  Returns:
    Mapping from `(mode, batch_size, seq_len, threads)` to the measured
    milliseconds of each preset at that point.
  """
  by_point: dict[tuple[str, int, int, int], dict[str, float]] = {}
  for record in layers:
    family = record.get("family", "")
    if family not in _BLOCK_FAMILIES or record.get("error_occurred"):
      continue
    point = (
        "prefill" if family.endswith("_Prefill") else "decode",
        int(record.get("batch_size", 1)),
        int(record.get("seq_len", 0)),
        int(record.get("threads", 1)),
    )
    by_point.setdefault(point, {})[record["preset"]] = float(
        record["real_time_ms"]
    )
  return by_point


def _estimate_model_at_point(
    model_name: str,
    weights: Sequence[Mapping[str, Any]],
    point: tuple[str, int, int, int],
    preset_ms: Mapping[str, float],
) -> dict[str, Any] | None:
  """Sums a model's weighted block latencies at one operating point.

  Args:
    model_name: Model the estimate is attributed to.
    weights: Block-multiplicity records of the model.
    point: Operating point `(mode, batch_size, seq_len, threads)`.
    preset_ms: Measured milliseconds per preset at `point`.

  Returns:
    The latency attribution record, or `None` when none of the model's
    blocks was measured at `point`.
  """
  covered = 0
  summed_ms = 0.0
  missing: list[str] = []
  approximations: dict[str, list[str]] = {}
  for weight in weights:
    preset = weight.get("preset")
    count = int(weight["count"])
    block_ms = preset_ms.get(preset) if preset else None
    if block_ms is None:
      missing.append(preset or weight.get("layer_id", "<unsupported>"))
      continue
    summed_ms += count * block_ms
    covered += count
    if weight.get("approximated_features"):
      approximations[preset] = list(weight["approximated_features"])
  if covered == 0:
    return None
  total_layers = sum(int(weight["count"]) for weight in weights)
  mode, batch_size, seq_len, threads = point
  estimate: dict[str, Any] = {
      "model": model_name,
      "mode": mode,
      "batch_size": batch_size,
      "seq_len": seq_len,
      "threads": threads,
      "summed_blocks_ms": summed_ms,
      "covered_layers": covered,
      "total_layers": total_layers,
      "complete": covered == total_layers and not missing,
      "missing_presets": missing,
  }
  if approximations:
    estimate["approximated_presets"] = approximations
  return estimate


def estimate_model_blocks_latency(
    layers: Sequence[Mapping[str, Any]],
    model_layer_weights: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
  """Sums weighted TransformerBlock latencies for DeltaT_runtime attribution.

  Args:
    layers: Normalized layer benchmark records from `parse_benchmark_entry`.
    model_layer_weights: Mapping of `model_name` to block-multiplicity records
      emitted by `catalog.resolve_suite`.

  Returns:
    List of per-model, per-operating-point latency attribution dictionaries.
  """
  if not layers or not model_layer_weights:
    return []
  by_point = _block_latency_by_point(layers)
  estimates: list[dict[str, Any]] = []
  for model_name, weights in sorted(model_layer_weights.items()):
    for point, preset_ms in sorted(by_point.items()):
      estimate = _estimate_model_at_point(model_name, weights, point, preset_ms)
      if estimate is not None:
        estimates.append(estimate)
  return estimates


def _split_benchmark_name(name: str) -> tuple[str, str, dict[str, str]]:
  """Splits `<family>/<preset>/<Key>:<Value>/...` into its components.

  Args:
    name: Google Benchmark name.

  Returns:
    Tuple `(family, preset, fields)`; `preset` is empty when the name has no
    second component, and `fields` holds the `Key:Value` components.
  """
  parts = name.split("/")
  preset = parts[1] if len(parts) > 1 else ""
  fields: dict[str, str] = {}
  for part in parts[2:]:
    key, separator, value = part.partition(":")
    if separator:
      fields[key] = value
  return parts[0], preset, fields


def _giga_rate(
    entry: Mapping[str, Any], giga_key: str, unit_keys: Sequence[str]
) -> float:
  """Reads a throughput counter in billions of units per second.

  Args:
    entry: Raw Google Benchmark entry.
    giga_key: Counter already expressed in 1e9 units per second.
    unit_keys: Fallback counters in units per second, by decreasing priority.

  Returns:
    The rate in 1e9 units per second, or 0.0 when no counter is present.
  """
  giga_value = entry.get(giga_key)
  if giga_value is not None:
    return float(giga_value)
  for key in unit_keys:
    unit_value = entry.get(key)
    if unit_value is not None:
      return float(unit_value) / 1e9
  return 0.0


def _theoretical_work(
    entry: Mapping[str, Any], key: str, giga_rate: float, real_time_ms: float
) -> float:
  """Reads a theoretical work counter, back-computing it from a rate.

  Args:
    entry: Raw Google Benchmark entry.
    key: Counter holding the theoretical work per iteration.
    giga_rate: Measured throughput in 1e9 work units per second.
    real_time_ms: Measured wall time per iteration.

  Returns:
    The counter when it is positive, else `rate * time` when both are
    positive, else the (non-positive) counter value.
  """
  work = float(entry.get(key, 0.0))
  if work <= 0.0 and giga_rate > 0.0 and real_time_ms > 0.0:
    work = giga_rate * 1e9 * (real_time_ms * 1e-3)
  return work


def _parity_counters(entry: Mapping[str, Any]) -> dict[str, Any]:
  """Extracts the C++ `Parity_*` counters of a benchmark entry.

  Args:
    entry: Raw Google Benchmark entry.

  Returns:
    Parity fields of the layer record; each is `None` when the binary did not
    report the counter (the benchmark ran without tensor comparison).
  """
  passed = entry.get("Parity_Passed")
  mismatches = entry.get("Parity_Mismatches")
  return {
      "parity_passed": None if passed is None else bool(passed >= 0.5),
      "parity_max_abs_err": entry.get("Parity_MaxAbsErr"),
      "parity_max_rel_err": entry.get("Parity_MaxRelErr"),
      "parity_cosine_sim": entry.get("Parity_CosineSim"),
      "parity_l2_rel_err": entry.get("Parity_L2RelErr"),
      "parity_mismatches": None if mismatches is None else int(mismatches),
  }


def _timing_fields(entry: Mapping[str, Any]) -> dict[str, float]:
  """Extracts timings in milliseconds and throughput rates of an entry.

  Args:
    entry: Raw Google Benchmark entry.

  Returns:
    The `real_time_ms`, `cpu_time_ms`, `gflops`, `gbps`, and
    `arithmetic_intensity` fields of the layer record. An unknown `time_unit`
    is read as milliseconds.
  """
  time_scales = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
  scale = time_scales.get(entry.get("time_unit", "ms"), 1.0)
  return {
      "real_time_ms": float(entry.get("real_time", 0.0)) * scale,
      "cpu_time_ms": float(entry.get("cpu_time", 0.0)) * scale,
      "gflops": _giga_rate(entry, "GFLOP/s", ("FLOPS", "FLOP/s")),
      "gbps": _giga_rate(entry, "GB/s", ("Bytes/s", "B/s")),
      "arithmetic_intensity": float(
          entry.get(
              "ArithmeticIntensity",
              entry.get("ArithmeticIntensity_FLOPs/Byte", 0.0),
          )
      ),
  }


def _work_and_memory_fields(
    entry: Mapping[str, Any], timing: Mapping[str, float]
) -> dict[str, Any]:
  """Extracts the theoretical work and memory counters of an entry.

  Args:
    entry: Raw Google Benchmark entry.
    timing: Result of `_timing_fields` for the same entry, used to back-compute
      theoretical work the binary did not report.

  Returns:
    The peak memory, allocation, theoretical FLOPs/bytes, workspace, and heap
    fields of the layer record. The workspace peak falls back to the peak
    memory counter.
  """
  peak_mem_mb = float(entry.get("PeakMem_MB", 0.0))
  real_time_ms = timing["real_time_ms"]
  return {
      "peak_mem_mb": peak_mem_mb,
      "alloc_delta_mb": float(entry.get("AllocDelta_MB", 0.0)),
      "theoretical_flops": _theoretical_work(
          entry, "Theoretical_FLOPs", timing["gflops"], real_time_ms
      ),
      "theoretical_bytes": _theoretical_work(
          entry, "Theoretical_Bytes", timing["gbps"], real_time_ms
      ),
      "workspace_peak_bytes": float(
          entry.get(
              "Workspace_Peak_Bytes",
              entry.get("WorkspacePeakBytes", peak_mem_mb * 1024.0 * 1024.0),
          )
      ),
      "workspace_allocated_bytes": float(
          entry.get("Workspace_TotalAllocated_Bytes", 0.0)
      ),
      "workspace_allocations": int(entry.get("Workspace_Allocations", 0)),
      "heap_allocated_bytes": float(entry.get("Heap_Allocated_Bytes", 0.0)),
  }


def parse_benchmark_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
  """Normalizes a single Google Benchmark JSON entry into a layer record.

  Args:
    entry: Raw dictionary from Google Benchmark's `benchmarks` JSON array.

  Returns:
    Normalized layer dictionary with timing, throughput, memory, and parity
    fields.
  """
  name = entry.get("name", "")
  family, preset, fields = _split_benchmark_name(name)
  timing = _timing_fields(entry)
  seq_len_kv = int(fields.get("S", entry.get("SeqLenKV", 0)))
  default_seq_q = 1 if family in catalog.DECODE_ONLY_FAMILIES else seq_len_kv
  return {
      "name": name,
      "family": family,
      "layer": family.removeprefix("BM_"),
      "preset": preset,
      "batch_size": int(fields.get("B", entry.get("BatchSize", 1))),
      "seq_len": seq_len_kv,
      "seq_len_q": int(round(float(entry.get("SeqLenQ", default_seq_q)))),
      "threads": int(fields.get("Threads", entry.get("Threads", 1))),
      "iterations": int(entry.get("iterations", 0)),
      **timing,
      **_work_and_memory_fields(entry, timing),
      "error_occurred": bool(entry.get("error_occurred", False)),
      "error_message": entry.get("error_message", ""),
      **_parity_counters(entry),
  }


def _sweep_axis(
    suite_data: MutableMapping[str, Any],
    key: str,
    override: int,
    fallback: int,
) -> list[int]:
  """Resolves one sweep axis and registers its values in the suite.

  Args:
    suite_data: Resolved suite; `suite_data[key]` is extended in place.
    key: Suite key of the axis, e.g. `batch_sizes`.
    override: CLI value replacing the suite's values; 0 keeps them.
    fallback: Value used when neither an override nor suite values exist.

  Returns:
    The values to sweep, without repeats, in first-occurrence order.
  """
  registered = list(
      dict.fromkeys(int(value) for value in suite_data.get(key, []))
  )
  if override > 0:
    selected = [int(override)]
  elif registered:
    selected = list(registered)
  else:
    selected = [fallback]
  suite_data[key] = sorted(set(registered) | set(selected))
  return selected


def resolve_sweep_points(
    suite_data: MutableMapping[str, Any],
    batch_size_arg: int = 0,
    seq_len_arg: int = 0,
    threads_arg: int = 0,
) -> tuple[list[int], list[int], int]:
  """Resolves `(batch_sizes, seq_lengths, threads)` from suite or CLI overrides.

  Mutates `suite_data` in-place so `layer_bench --suite_file=...` registers
  every requested operating point, including when `suite_data` starts empty.

  Args:
    suite_data: Resolved suite dictionary (updated in place).
    batch_size_arg: CLI `--batch-size` override (`0` uses suite/default).
    seq_len_arg: CLI `--seq-len` override (`0` uses suite/default).
    threads_arg: CLI `--threads` override (`0` uses machine profile/default).

  Returns:
    Tuple `(batch_sizes, seq_lengths, effective_threads)`.
  """
  batch_sizes = _sweep_axis(suite_data, "batch_sizes", batch_size_arg, 1)
  seq_lengths = _sweep_axis(suite_data, "seq_lengths", seq_len_arg, 1024)
  registered_threads = [
      int(value) for value in suite_data.get("thread_counts", [])
  ]
  machine_profile = suite_data.get("machine_profile") or {}
  if threads_arg > 0:
    threads = int(threads_arg)
  elif machine_profile.get("physical_cores_per_domain"):
    threads = int(machine_profile["physical_cores_per_domain"])
  elif registered_threads:
    threads = registered_threads[-1]
  else:
    threads = 32
  suite_data["thread_counts"] = sorted(set(registered_threads) | {threads})
  return batch_sizes, seq_lengths, threads


def default_binary(
    backend: str,
    env: Mapping[str, str] | None = None,
    install_dir: Path | None = None,
) -> str:
  """Resolves the `layer_bench` binary path from env vars or default locations.

  Args:
    backend: Backend identifier such as `"xnnpack"` or `"ynnpack"`.
    env: Environment variables to consult (`AEB_LAYER_BIN_<BACKEND>`,
      `AEB_LAYER_BIN`, and `PATH`); `None` uses the process environment.
    install_dir: Directory the benchmark image installs `layer_bench` into;
      `None` uses `/opt/litert-tensor/bin`.

  Returns:
    Resolved executable path string: the configured binary, else an
    installed one, else one found on `PATH`. When none of them exists, the
    backend-specific install path, which may not exist.
  """
  if env is None:
    env = os.environ
  if install_dir is None:
    install_dir = _INSTALL_BIN_DIR
  configured = env.get(f"AEB_LAYER_BIN_{backend.upper()}") or env.get(
      "AEB_LAYER_BIN"
  )
  if configured:
    return configured
  search_path = env.get("PATH", os.defpath)
  fallback = install_dir / f"layer_bench_{backend}"
  for candidate in (
      fallback,
      install_dir / "layer_bench",
      shutil.which(f"layer_bench_{backend}", path=search_path),
      shutil.which("layer_bench", path=search_path),
  ):
    if candidate and Path(candidate).is_file():
      return str(candidate)
  return str(fallback)


def _add_identity_flags(parser: argparse.ArgumentParser) -> None:
  """Registers the flags naming the framework, backend, and binary."""
  parser.add_argument(
      "--framework",
      default="litert-tensor",
      help="framework identifier recorded with the results",
  )
  parser.add_argument(
      "--label", required=True, help="configuration label, e.g. xnnpack"
  )
  parser.add_argument(
      "--backend",
      default="xnnpack",
      help="delegate/kernel backend identifier (xnnpack | ynnpack)",
  )
  parser.add_argument(
      "--dtype",
      default="",
      help="weight/compute dtype label (default: the suite's dtype, else fp32)",
  )
  parser.add_argument(
      "--binary",
      default="",
      help=(
          "layer_bench executable (default: $AEB_LAYER_BIN_<BACKEND>, "
          "$AEB_LAYER_BIN, or an installed layer_bench)"
      ),
  )


def _add_selection_flags(parser: argparse.ArgumentParser) -> None:
  """Registers the flags selecting the suite, inputs, and benchmarks."""
  parser.add_argument(
      "--suite-file",
      default="",
      help="optional declarative JSON layer suite file",
  )
  parser.add_argument(
      "--ceilings-json",
      default="",
      help="optional Level 1 hardware ceiling probes JSON for Roofline %%",
  )
  parser.add_argument(
      "--kernel-timings-json",
      default="",
      help="optional Level 1 summed kernel timings JSON for DeltaT_compose",
  )
  parser.add_argument(
      "--presets",
      default=None,
      help="comma-separated model presets to benchmark",
  )
  parser.add_argument(
      "--families",
      default=None,
      help="comma-separated BM_* layer families to benchmark",
  )


def _add_sweep_flags(parser: argparse.ArgumentParser) -> None:
  """Registers the operating-point and measurement flags."""
  parser.add_argument(
      "--batch-size",
      type=int,
      default=0,
      help="batch size filter (0 = sweep suite batch_sizes, fallback 1)",
  )
  parser.add_argument(
      "--seq-len",
      type=int,
      default=0,
      help=(
          "sequence length filter (0 = sweep suite seq_lengths, fallback 1024)"
      ),
  )
  parser.add_argument(
      "--threads",
      type=int,
      default=0,
      help=(
          "worker threads (0 = auto from machine_profile "
          "physical_cores_per_domain, fallback 32)"
      ),
  )
  parser.add_argument(
      "--min-time",
      default="0.2s",
      help="Google Benchmark minimum measurement time per benchmark",
  )
  parser.add_argument(
      "--seed", type=int, default=1234, help="seed of the synthetic tensors"
  )
  parser.add_argument(
      "--wipe-caches",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="evict CPU caches between benchmark iterations",
  )


def _add_parity_flags(parser: argparse.ArgumentParser) -> None:
  """Registers the tensor dump/compare flags and their thresholds."""
  parity_group = parser.add_mutually_exclusive_group()
  parity_group.add_argument(
      "--dump-parity-dir",
      default="",
      help="directory receiving one reference tensor dump per case",
  )
  parity_group.add_argument(
      "--compare-parity-dir",
      default="",
      help="directory of reference tensor dumps to compare each case against",
  )
  thresholds = (
      ("--parity-rtol", "relative tolerance of the strict allclose gate"),
      ("--parity-atol", "absolute tolerance of the strict allclose gate"),
      ("--parity-min-cosine", "minimum cosine similarity (<= 0 disables)"),
      ("--parity-max-l2-rel-err", "maximum relative L2 error (<= 0 disables)"),
      ("--parity-max-kl", "maximum softmax KL-divergence (<= 0 disables)"),
  )
  for flag, description in thresholds:
    parser.add_argument(
        flag,
        type=float,
        default=None,
        help=f"{description}; default: the suite's quality gate",
    )


def _add_profile_flags(parser: argparse.ArgumentParser) -> None:
  """Registers the hyperparameter profile and result metadata flags."""
  parser.add_argument(
      "--regime", default="", help="optional hyperparams regime"
  )
  parser.add_argument(
      "--quant-scheme", default="", help="optional hyperparams quant_scheme"
  )
  parser.add_argument(
      "--cliff-sweep", default="", help="optional hyperparams cliff_sweep"
  )
  parser.add_argument(
      "--machine-profile",
      default="",
      help=(
          "optional hyperparams machine_threading profile (thread counts and "
          "Roofline ceilings)"
      ),
  )
  parser.add_argument(
      "--round", type=int, default=0, help="optimization round recorded"
  )
  parser.add_argument("--track", default="layer", help="results track recorded")
  parser.add_argument("--notes", default="", help="free-form notes recorded")


def _build_arg_parser() -> argparse.ArgumentParser:
  """Builds the command-line parser of the layer benchmark runner."""
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  _add_identity_flags(parser)
  _add_selection_flags(parser)
  _add_sweep_flags(parser)
  _add_parity_flags(parser)
  _add_profile_flags(parser)
  return parser


def _strict_json(value: Any) -> Any:
  """Copies a JSON-compatible value, replacing non-finite floats by `None`.

  RFC 8259 has no encoding for NaN or infinities, so they are written as
  `null`.

  Args:
    value: JSON-compatible value (mappings, sequences, and scalars).

  Returns:
    A structure equal to `value` except that NaN and infinite floats are
    `None` and tuples are lists.
  """
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, Mapping):
    return {key: _strict_json(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_strict_json(item) for item in value]
  return value


def _dump_strict_json(path: Path, payload: Any) -> None:
  """Writes `payload` to `path` as RFC 8259 JSON.

  Args:
    path: Destination file; parent directories are created.
    payload: JSON-compatible value; non-finite floats are written as `null`.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(_strict_json(payload), indent=1, allow_nan=False)
  path.write_text(text + "\n", encoding="utf-8")


def _split_csv(flag: str, value: str | None) -> list[str] | None:
  """Splits a comma-separated flag into unique, stripped, non-empty names.

  Args:
    flag: Name of the flag, quoted in the error message.
    value: Raw flag value; `None` when the flag was not passed.

  Returns:
    The names in first-occurrence order without duplicates, or `None` when
    `value` is `None`.

  Raises:
    _UsageError: If the flag was passed without naming anything.
  """
  if value is None:
    return None
  stripped = (name.strip() for name in value.split(","))
  names = list(dict.fromkeys(name for name in stripped if name))
  if not names:
    raise _UsageError(f"{flag} must name at least one entry (got {value!r}).")
  return names


def _validate_flags(args: argparse.Namespace) -> None:
  """Rejects flag values that can never produce a valid run.

  Args:
    args: Parsed command-line flags.

  Raises:
    _UsageError: If a sweep override is negative, or an explicitly passed
      ceilings or kernel timings file does not exist.
  """
  for flag, value in (
      ("--batch-size", args.batch_size),
      ("--seq-len", args.seq_len),
      ("--threads", args.threads),
  ):
    if value < 0:
      raise _UsageError(f"{flag} must be >= 0 (got {value}).")
  for flag, path in (
      ("--ceilings-json", args.ceilings_json),
      ("--kernel-timings-json", args.kernel_timings_json),
  ):
    if path and not Path(path).is_file():
      raise _UsageError(f"{flag} not found: {path}")


def _default_suites_dir() -> Path:
  """Returns the repository `layer/suites` directory, or the container's."""
  suites_dir = Path(__file__).resolve().parents[4] / "layer" / "suites"
  return suites_dir if suites_dir.is_dir() else Path("/layer/suites")


def _load_default_catalog_presets(
    suites_dir: Path, preset_names: Sequence[str]
) -> list[dict[str, Any] | str]:
  """Resolves preset names into catalog preset dicts for ad-hoc cliff sweeps.

  Args:
    suites_dir: Suites directory the default catalog is located relative to.
    preset_names: Preset names to resolve.

  Returns:
    For each name, the preset dictionary of the catalog block with that
    preset name, or the name itself when the default catalog is missing or
    has no such preset (cliff sweeps cannot perturb plain names).

  Raises:
    OSError: If the default catalog cannot be read.
    ValueError: If the default catalog is invalid.
  """
  catalog_path = (suites_dir / catalog.DEFAULT_CATALOG_FILE).resolve()
  if not catalog_path.is_file():
    return list(preset_names)
  catalog_data = catalog.load_json_object(catalog_path)
  catalog.validate_catalog(catalog_data, source=str(catalog_path))
  presets_by_name: dict[str, dict[str, Any]] = {}
  for block in catalog_data.get("composite_blocks", []):
    preset = catalog.block_to_preset(block)
    if preset is not None:
      presets_by_name[preset["name"]] = preset
  return [presets_by_name.get(name, name) for name in preset_names]


def _load_suite(
    args: argparse.Namespace,
    requested_presets: Sequence[str] | None,
    requested_families: Sequence[str] | None,
) -> tuple[dict[str, Any], Path | None]:
  """Resolves the suite file, or an ad-hoc suite for profile-only flags.

  Args:
    args: Parsed command-line flags.
    requested_presets: Names from `--presets`; `None` when not passed.
    requested_families: Names from `--families`; `None` when not passed.

  Returns:
    Tuple `(suite_data, suite_dir)`. Without a suite file or any
    hyperparameter profile flag, `suite_data` is empty and `suite_dir` is
    `None`; otherwise `suite_dir` is the directory that relative suite paths
    resolve against.

  Raises:
    OSError: If the suite or a file it references cannot be read.
    ValueError: If the suite or a file it references is invalid.
  """
  overrides = {
      "machine_override": args.machine_profile,
      "regime_override": args.regime,
      "quant_override": args.quant_scheme,
      "cliff_override": args.cliff_sweep,
  }
  if args.suite_file:
    suite_path = Path(args.suite_file)
    suite = catalog.load_json_object(suite_path)
    suite_dir = suite_path.parent
    return catalog.resolve_suite(suite, suite_dir, **overrides), suite_dir
  if not any(overrides.values()):
    return {}, None
  suites_dir = _default_suites_dir()
  seed_suite: dict[str, Any] = {}
  if args.cliff_sweep:
    seed_suite["presets"] = _load_default_catalog_presets(
        suites_dir, requested_presets or DEFAULT_PRESETS
    )
  if requested_families is not None:
    seed_suite["families"] = list(requested_families)
  return catalog.resolve_suite(seed_suite, suites_dir, **overrides), suites_dir


def _unknown_profile_hint(kind: str) -> str:
  """Returns the remedy appended to an unknown-profile error.

  Args:
    kind: Profile kind reported by `catalog.UnknownProfileError`.

  Returns:
    A sentence naming the flag that selects a profile of that kind.
  """
  if kind == "machine_threading":
    return (
        "Pass --machine-profile with a known profile, or omit it to run "
        "without a host threading profile (make bench: "
        "MACHINE_PROFILE=<profile> or MACHINE_PROFILE=)."
    )
  # The remaining profile flags are named after their profile kind.
  flag = "--" + kind.replace("_", "-")
  return f"Pass {flag} with a known profile."


def _resolve_suite(
    args: argparse.Namespace,
    requested_presets: Sequence[str] | None,
    requested_families: Sequence[str] | None,
) -> tuple[dict[str, Any], Path | None]:
  """Resolves the run's suite, reporting bad inputs as usage errors.

  Args:
    args: Parsed command-line flags.
    requested_presets: Names from `--presets`; `None` when not passed.
    requested_families: Names from `--families`; `None` when not passed.

  Returns:
    Tuple `(suite_data, suite_dir)` as documented by `_load_suite`.

  Raises:
    _UsageError: If the suite, catalog, or hyperparameter registry is
      unreadable or invalid, or names an undefined profile.
  """
  try:
    return _load_suite(args, requested_presets, requested_families)
  except catalog.UnknownProfileError as error:
    hint = _unknown_profile_hint(error.kind)
    raise _UsageError(f"{error}. {hint}") from error
  except (OSError, ValueError) as error:
    raise _UsageError(f"cannot resolve layer suite: {error}") from error


def _discover_ceilings(
    suite_data: Mapping[str, Any], suite_dir: Path | None
) -> str:
  """Locates the hardware ceilings file a resolved suite points at.

  Args:
    suite_data: Resolved suite, optionally naming a `ceilings_file`.
    suite_dir: Directory that a relative `ceilings_file` resolves against;
      `None` when the run has no suite.

  Returns:
    The path of the suite's ceilings file, or an empty string when the suite
    names none or the file does not exist.
  """
  ceilings_file = suite_data.get("ceilings_file")
  if not ceilings_file or suite_dir is None:
    return ""
  path = Path(ceilings_file)
  if not path.is_absolute():
    path = (suite_dir / path).resolve()
  return str(path) if path.is_file() else ""


def _validate_framework(
    framework: str, backend: str, dtype: str, quant_profile: Any
) -> None:
  """Rejects backend and dtype combinations a framework cannot execute.

  Args:
    framework: Framework identifier from `--framework`.
    backend: Kernel backend from `--backend`.
    dtype: Weight/compute dtype label of the run.
    quant_profile: The resolved suite's `quant_profile` value: the quantization
      scheme bound to the run, naming its weight, activation, accumulator, and
      KV-cache dtypes. `None` when the suite binds no scheme; a dtype the scheme
      does not name counts as FP32.

  Raises:
    _UsageError: If `framework` is `litert-tensor` and the backend is
      unsupported, `quant_profile` is not an object, or `dtype` or any dtype
      of the scheme is not FP32.
  """
  if framework != "litert-tensor":
    return
  if backend not in ("xnnpack", "ynnpack"):
    raise _UsageError(
        f"unsupported --backend={backend!r} for "
        "framework='litert-tensor'; expected 'xnnpack' or 'ynnpack'."
    )
  if dtype.lower() not in _FP32_DTYPES:
    raise _UsageError(
        "framework='litert-tensor' layer_bench currently "
        f"executes FP32 graphs only; refusing dtype={dtype!r} to prevent "
        "mislabeling FP32 measurements."
    )
  if quant_profile is None:
    return
  if not isinstance(quant_profile, Mapping):
    raise _UsageError("suite quant_profile must be a JSON object.")
  for field in _QUANT_PROFILE_DTYPE_FIELDS:
    scheme_dtype = str(quant_profile.get(field) or "fp32")
    if scheme_dtype.lower() not in _FP32_DTYPES:
      raise _UsageError(
          "framework='litert-tensor' layer_bench currently executes FP32 "
          f"graphs only; refusing quant scheme {quant_profile.get('name')!r} "
          f"with {field}={scheme_dtype!r} to prevent mislabeling FP32 "
          "measurements."
      )


def _resolve_parity_gate(
    args: argparse.Namespace, quality_gate: Any
) -> _ParityGate:
  """Combines CLI parity thresholds with the suite's quality gate.

  Args:
    args: Parsed command-line flags; a `--parity-*` flag overrides the suite.
    quality_gate: The resolved suite's `quality_gate` value.

  Returns:
    The thresholds in effect for the run.

  Raises:
    _UsageError: If a threshold is not a finite number, or a tolerance is
      negative.
  """
  if not isinstance(quality_gate, Mapping):
    raise _UsageError("suite quality_gate must be a JSON object.")
  thresholds: dict[str, float] = {}
  for name, override, suite_key, default in (
      ("rtol", args.parity_rtol, "rtol", 1e-4),
      ("atol", args.parity_atol, "atol", 1e-5),
      ("min_cosine", args.parity_min_cosine, "min_cosine", 0.0),
      ("max_l2_rel_err", args.parity_max_l2_rel_err, "max_l2_rel_err", 0.0),
      ("max_kl", args.parity_max_kl, "max_kl_divergence", 0.0),
  ):
    raw = quality_gate.get(suite_key, default) if override is None else override
    is_number = isinstance(raw, (int, float)) and not isinstance(raw, bool)
    if not is_number or not math.isfinite(raw):
      raise _UsageError(
          f"parity threshold {name} must be a finite number (got {raw!r})."
      )
    thresholds[name] = float(raw)
  gate = _ParityGate(**thresholds)
  if gate.rtol < 0.0 or gate.atol < 0.0:
    raise _UsageError(
        f"parity tolerances must be >= 0 (got rtol={gate.rtol}, "
        f"atol={gate.atol})."
    )
  return gate


def _load_roofline_inputs(
    ceilings_path: str, kernel_timings_path: str
) -> tuple[dict[str, Any] | None, dict[str, float] | None]:
  """Loads the hardware ceilings and kernel timings of a run.

  Args:
    ceilings_path: Hardware ceilings JSON file; empty when there is none.
    kernel_timings_path: Summed kernel timings JSON file; empty when there is
      none.

  Returns:
    Tuple `(ceilings, kernel_timings)`; each is `None` when its path is
    empty.

  Raises:
    _UsageError: If a file is unreadable or malformed.
  """
  ceilings = None
  kernel_timings = None
  try:
    if ceilings_path:
      ceilings = load_hardware_ceilings(Path(ceilings_path))
    if kernel_timings_path:
      kernel_timings = load_kernel_timings(Path(kernel_timings_path))
  except (OSError, ValueError) as error:
    raise _UsageError(f"cannot load Roofline inputs: {error}") from error
  return ceilings, kernel_timings


def _filter_suite_presets(
    suite_presets: Sequence[dict[str, Any] | str], requested: Sequence[str]
) -> list[dict[str, Any] | str]:
  """Keeps the requested suite presets and their cliff-sweep variants.

  Args:
    suite_presets: Presets of the resolved suite.
    requested: Preset names from `--presets`.

  Returns:
    The suite presets whose name or `cliff_parent` was requested, in suite
    order, followed by requested names the suite does not define (so presets
    built into the binary remain selectable).
  """
  wanted = set(requested)
  selected: list[dict[str, Any] | str] = []
  matched: set[str] = set()
  for preset in suite_presets:
    name = catalog.preset_name(preset)
    parent = preset.get("cliff_parent") if isinstance(preset, dict) else None
    if name in wanted or parent in wanted:
      selected.append(preset)
      matched.add(name)
  selected.extend(name for name in requested if name not in matched)
  return selected


def _select_presets(
    suite_data: MutableMapping[str, Any], requested: Sequence[str] | None
) -> list[str]:
  """Selects the presets to benchmark and registers them in the suite.

  Args:
    suite_data: Resolved suite; `suite_data["presets"]` is replaced by the
      selection so the binary registers exactly these presets.
    requested: Preset names from `--presets`; `None` keeps the suite's presets,
      or the built-in defaults when the suite defines none.

  Returns:
    Names of the selected presets, in execution order.
  """
  suite_presets = suite_data.get("presets")
  if requested is None:
    if not suite_presets:
      suite_data["presets"] = list(DEFAULT_PRESETS)
  elif suite_presets:
    suite_data["presets"] = _filter_suite_presets(suite_presets, requested)
  else:
    suite_data["presets"] = list(requested)
  return [catalog.preset_name(preset) for preset in suite_data["presets"]]


def _select_families(
    suite_data: MutableMapping[str, Any], requested: Sequence[str] | None
) -> list[str]:
  """Selects the benchmark families and registers them in the suite.

  Args:
    suite_data: Resolved suite; `suite_data["families"]` is replaced by the
      selection.
    requested: Family names from `--families`, filtered by the suite's regime
      mode when it has one; `None` keeps the suite's families, or the built-in
      defaults when the suite defines none.

  Returns:
    Names of the selected families, without repeats, in execution order.

  Raises:
    _UsageError: If no requested family matches the regime's mode.
  """
  if requested is None:
    families = list(
        dict.fromkeys(suite_data.get("families") or DEFAULT_FAMILIES)
    )
  else:
    families = list(requested)
    regime = suite_data.get("regime_profile")
    if regime:
      mode = str(regime.get("mode", "both"))
      families = catalog.filter_families_for_mode(requested, mode)
      if not families:
        raise _UsageError(
            f"--families={requested!r} has no "
            f"families matching regime mode={mode!r}."
        )
  suite_data["families"] = list(families)
  return families


def _select_sweep(
    args: argparse.Namespace,
    suite_data: MutableMapping[str, Any],
    requested_presets: Sequence[str] | None,
    requested_families: Sequence[str] | None,
) -> _Sweep:
  """Selects the benchmarks and operating points of a run.

  Every selected value is also registered in `suite_data`, which is what the
  binary reads its benchmarks from.

  Args:
    args: Parsed command-line flags providing the sweep overrides.
    suite_data: Resolved suite (updated in place).
    requested_presets: Names from `--presets`; `None` when not passed.
    requested_families: Names from `--families`; `None` when not passed.

  Returns:
    The selected presets, families, and operating points.

  Raises:
    _UsageError: If no requested family matches the suite's regime mode.
  """
  presets = _select_presets(suite_data, requested_presets)
  families = _select_families(suite_data, requested_families)
  batch_sizes, seq_lengths, threads = resolve_sweep_points(
      suite_data,
      batch_size_arg=args.batch_size,
      seq_len_arg=args.seq_len,
      threads_arg=args.threads,
  )
  return _Sweep(presets, families, batch_sizes, seq_lengths, threads)


def _plan_run(args: argparse.Namespace, hooks: RunnerHooks) -> _RunPlan:
  """Resolves flags, suite, and input files into a run plan.

  Args:
    args: Parsed command-line flags.
    hooks: Host collaborators providing the environment variables and install
      directory consulted for binary discovery.

  Returns:
    The resolved plan; nothing has been written or executed yet.

  Raises:
    _UsageError: If a flag value, the suite, or an input file is invalid.
  """
  _validate_flags(args)
  requested_presets = _split_csv("--presets", args.presets)
  requested_families = _split_csv("--families", args.families)
  suite_data, suite_dir = _resolve_suite(
      args, requested_presets, requested_families
  )
  dtype = str(args.dtype or suite_data.get("dtype") or "fp32")
  _validate_framework(
      args.framework, args.backend, dtype, suite_data.get("quant_profile")
  )
  gate = _resolve_parity_gate(args, suite_data.get("quality_gate", {}))
  ceilings_path = args.ceilings_json or _discover_ceilings(
      suite_data, suite_dir
  )
  ceilings, kernel_timings = _load_roofline_inputs(
      ceilings_path, args.kernel_timings_json
  )
  dump_dir = args.dump_parity_dir
  compare_dir = args.compare_parity_dir
  return _RunPlan(
      suite_data=suite_data,
      dtype=dtype,
      backend=args.backend,
      binary=args.binary
      or default_binary(args.backend, hooks.environ, hooks.install_bin_dir),
      sweep=_select_sweep(
          args, suite_data, requested_presets, requested_families
      ),
      min_time=args.min_time,
      seed=args.seed,
      wipe_caches=args.wipe_caches,
      gate=gate,
      dump_dir=Path(dump_dir) if dump_dir else None,
      compare_dir=Path(compare_dir) if compare_dir else None,
      ceilings_path=ceilings_path,
      ceilings=ceilings,
      kernel_timings=kernel_timings,
  )


def _alternation(values: Sequence[Any]) -> str:
  """Returns a regex group matching any of `values` literally."""
  return "(" + "|".join(re.escape(str(value)) for value in values) + ")"


def _plan_cases(plan: _RunPlan) -> list[_BenchCase]:
  """Expands a run plan into `layer_bench` invocations.

  Tensor dumps hold a single output, so parity runs invoke the binary once
  per `(preset, family, batch size, sequence length)`; other runs cover the
  whole sweep with one alternation filter.

  Args:
    plan: Resolved run plan.

  Returns:
    The invocations, in execution order.
  """
  sweep = plan.sweep
  suffix = f"/Threads:{sweep.threads}(/real_time)?$"
  if plan.dump_dir is None and plan.compare_dir is None:
    sweep_filter = (
        f"^{_alternation(sweep.families)}/{_alternation(sweep.presets)}"
        f"/B:{_alternation(sweep.batch_sizes)}"
        f"/S:{_alternation(sweep.seq_lengths)}{suffix}"
    )
    return [_BenchCase("", "", 0, 0, sweep_filter)]
  cases: list[_BenchCase] = []
  for preset, family, batch_size, seq_len in itertools.product(
      sweep.presets, sweep.families, sweep.batch_sizes, sweep.seq_lengths
  ):
    case_filter = (
        f"^{re.escape(family)}/{re.escape(preset)}"
        f"/B:{batch_size}/S:{seq_len}{suffix}"
    )
    cases.append(_BenchCase(preset, family, batch_size, seq_len, case_filter))
  return cases


def _record_parity_failure(
    case: _BenchCase, message: str, log: _RunLog
) -> None:
  """Records a parity case that failed before its tensors were compared.

  Args:
    case: Invocation whose parity check failed.
    message: Human-readable failure, stored as the record's `details`.
    log: Run log; receives the failure and a failed parity record.
  """
  log.errors.append(message)
  log.parity_records.append(
      {**case.identity(), "passed": False, "details": message}
  )


def _locate_parity_files(
    plan: _RunPlan, case: _BenchCase, run_dir: Path, log: _RunLog
) -> _ParityFiles | None:
  """Chooses the tensor dump files of one case.

  Args:
    plan: Resolved run plan.
    case: Invocation being prepared.
    run_dir: Run directory; candidate dumps go to its `parity_cand` folder.
    log: Run log; receives the failure when a reference dump is missing.

  Returns:
    The files to exchange with the binary (all unset when the run has no
    parity directory), or `None` when the run compares tensors but the case
    has no reference dump and must be skipped.
  """
  if plan.dump_dir is not None:
    return _ParityFiles(dump=plan.dump_dir / f"{case.dump_stem}.lmlb")
  if plan.compare_dir is None:
    return _ParityFiles()
  reference = plan.compare_dir / f"{case.dump_stem}.lmlb"
  if not reference.is_file():
    # Dumps written before batch size and sequence length were part of the
    # file name remain usable as references.
    legacy = plan.compare_dir / f"{case.preset}__{case.family}.lmlb"
    if not legacy.is_file():
      message = (
          f"Missing reference parity dump for {case.preset}/{case.family} "
          f"(B={case.batch_size}, S={case.seq_len}): expected {reference}"
      )
      _record_parity_failure(case, message, log)
      print(f"[aeb] {message}", file=sys.stderr)
      return None
    reference = legacy
  candidate_dir = run_dir / "parity_cand"
  candidate_dir.mkdir(parents=True, exist_ok=True)
  return _ParityFiles(
      dump=candidate_dir / f"{case.dump_stem}.lmlb", reference=reference
  )


def _build_command(
    plan: _RunPlan,
    suite_file: Path,
    case: _BenchCase,
    out_json: Path,
    parity: _ParityFiles,
) -> list[str]:
  """Builds the `layer_bench` command line of one invocation.

  Args:
    plan: Resolved run plan.
    suite_file: Resolved suite JSON the binary registers benchmarks from.
    case: Invocation to build the command for.
    out_json: File the binary writes its Google Benchmark JSON to.
    parity: Tensor dump files of the case.

  Returns:
    The executable path followed by its arguments.
  """
  gate = plan.gate
  command = [
      plan.binary,
      f"--backend={plan.backend}",
      f"--suite_file={suite_file}",
      f"--benchmark_filter={case.filter_regex}",
      "--benchmark_format=json",
      f"--benchmark_out={out_json}",
      f"--benchmark_min_time={plan.min_time}",
      f"--wipe_caches={'true' if plan.wipe_caches else 'false'}",
      f"--seed={plan.seed}",
      f"--parity_rtol={gate.rtol}",
      f"--parity_atol={gate.atol}",
  ]
  if gate.min_cosine > 0.0:
    command.append(f"--parity_min_cosine={gate.min_cosine}")
  if gate.max_l2_rel_err > 0.0:
    command.append(f"--parity_max_l2_rel_err={gate.max_l2_rel_err}")
  # The binary has no KL flag and would enforce strict elementwise allclose;
  # with only the KL gate enabled it just dumps the candidate tensor and
  # `compare_parity_dumps` evaluates the gate.
  if parity.reference is not None and not gate.kl_only:
    command.append(f"--compare_parity_tensors={parity.reference}")
  if parity.dump is not None:
    command.append(f"--dump_parity_tensors={parity.dump}")
  return command


def _invoke(
    command: Sequence[str], driver_log: Path, hooks: RunnerHooks, log: _RunLog
) -> subprocess.CompletedProcess[str] | None:
  """Runs one command and appends it and its output to the driver log.

  Args:
    command: Command line to run; its first element is the binary.
    driver_log: Log file the command and its output are appended to.
    hooks: Host collaborators providing the command runner.
    log: Run log; receives the failure when the binary cannot be started.

  Returns:
    The completed process, or `None` when the binary could not be started.
  """
  command_line = " ".join(command)
  try:
    process = hooks.run_command(command)
  except OSError as error:
    message = (
        f"Failed to execute layer benchmark binary {command[0]!r}: {error}"
    )
    log.errors.append(message)
    with open(driver_log, "a", encoding="utf-8") as log_file:
      log_file.write(f"$ {command_line}\nERROR: {message}\n")
    print(f"[aeb] ERROR: {message}", file=sys.stderr)
    return None
  with open(driver_log, "a", encoding="utf-8") as log_file:
    log_file.write(f"$ {command_line}\n")
    if process.stdout:
      log_file.write(process.stdout)
    if process.stderr:
      log_file.write(process.stderr)
  return process


def _parse_benchmark_report(raw_text: str) -> dict[str, Any]:
  """Parses the Google Benchmark JSON report of one invocation.

  Args:
    raw_text: Report text. Bare `NaN` and `Infinity` tokens, which Google
      Benchmark emits, are accepted.

  Returns:
    The report object; its `benchmarks` member, when present, is a list of
    objects.

  Raises:
    ValueError: If the text is not JSON or does not have that structure.
  """
  report = json.loads(raw_text)
  if not isinstance(report, dict):
    raise ValueError("expected an object")
  benchmarks = report.get("benchmarks", [])
  if not isinstance(benchmarks, list) or not all(
      isinstance(entry, dict) for entry in benchmarks
  ):
    raise ValueError("expected 'benchmarks' to be a list of objects")
  return report


def _collect_benchmarks(out_json: Path, log: _RunLog) -> None:
  """Moves the Google Benchmark output of one invocation into the run log.

  A missing or empty output file contributes nothing. A parsed file is
  deleted; a malformed one is kept for inspection and recorded as a failure.

  Args:
    out_json: Google Benchmark JSON written by the binary.
    log: Run log; receives the benchmark entries and the first `context`.
  """
  if not out_json.is_file():
    return
  raw_text = out_json.read_text(encoding="utf-8").strip()
  if raw_text:
    try:
      report = _parse_benchmark_report(raw_text)
    except ValueError as error:
      log.errors.append(f"Malformed benchmark JSON in {out_json}: {error}")
      return
    if not log.context and "context" in report:
      log.context = report["context"]
    log.raw_benchmarks.extend(report.get("benchmarks", []))
  out_json.unlink(missing_ok=True)


def _record_comparison(
    gate: _ParityGate,
    case: _BenchCase,
    reference: Path,
    candidate: Path,
    log: _RunLog,
) -> None:
  """Compares a case's candidate dump against its reference.

  Args:
    gate: Numerical parity thresholds.
    case: Invocation that should have written the candidate dump.
    reference: Reference dump of the case.
    candidate: Candidate dump the binary was asked to write.
    log: Run log; receives the comparison record and any failure.
  """
  case_name = f"{case.preset}/{case.family}"
  if not (reference.is_file() and candidate.is_file()):
    message = (
        f"Candidate parity dump was not produced for {case_name}: "
        f"expected {candidate}"
    )
    _record_parity_failure(case, message, log)
    return
  try:
    comparison = compare_parity_dumps(
        reference,
        candidate,
        rtol=gate.rtol,
        atol=gate.atol,
        min_cosine=gate.min_cosine,
        max_l2_rel_err=gate.max_l2_rel_err,
        max_kl_divergence=gate.max_kl,
    )
  except (OSError, ValueError) as error:
    message = f"Parity dumps of {case_name} could not be compared: {error}"
    _record_parity_failure(case, message, log)
    return
  comparison.update(case.identity())
  log.parity_records.append(comparison)
  if not comparison["passed"]:
    log.errors.append(f"Parity failed for {case_name}: {comparison['details']}")


def _record_reference(case: _BenchCase, dump: Path, log: _RunLog) -> None:
  """Records the reference dump a case wrote.

  Args:
    case: Invocation that should have written the dump.
    dump: Reference dump file of the case.
    log: Run log; receives the reference record, or the failure when the dump is
      missing or invalid.
  """
  case_name = f"{case.preset}/{case.family}"
  if not dump.is_file():
    log.errors.append(
        f"Reference parity dump was not produced for {case_name}: "
        f"expected {dump}"
    )
    return
  try:
    shape, n_elems = read_parity_dump_header(dump)
  except (OSError, ValueError) as error:
    log.errors.append(
        f"Reference parity dump of {case_name} is invalid: {error}"
    )
    return
  log.parity_records.append({
      **case.identity(),
      "role": "reference",
      "passed": True,
      "shape": shape,
      "total_elements": n_elems,
      "file": str(dump),
  })


def _remove_stale_dump(case: _BenchCase, dump: Path, log: _RunLog) -> bool:
  """Removes a tensor dump left at a case's dump path by an earlier run.

  A dump found after the invocation is then known to come from it, so a
  binary that fails to write one cannot pass off an outdated tensor.

  Args:
    case: Invocation about to write the dump.
    dump: Tensor dump file of the case.
    log: Run log; receives the failure when the file cannot be removed.

  Returns:
    Whether `dump` is free for the binary to write.
  """
  try:
    dump.unlink(missing_ok=True)
  except OSError as error:
    log.errors.append(
        f"Stale parity dump of {case.preset}/{case.family} could not be "
        f"removed: {error}"
    )
    return False
  return True


def _run_cases(plan: _RunPlan, run_dir: Path, hooks: RunnerHooks) -> _RunLog:
  """Runs every planned `layer_bench` invocation and gathers its outputs.

  A case without a reference dump or with an unremovable stale dump is
  skipped, a failing command skips only its own outputs, and a binary that
  cannot be started aborts the remaining cases.

  Args:
    plan: Resolved run plan.
    run_dir: Run directory holding the resolved suite, the driver log, the
      intermediate benchmark outputs, and candidate tensor dumps.
    hooks: Host collaborators providing the command runner.

  Returns:
    The benchmark entries, parity records, commands, and failures of the run.
  """
  log = _RunLog()
  suite_file = run_dir / "resolved_suite.json"
  for index, case in enumerate(_plan_cases(plan)):
    out_json = run_dir / f"bench_{index:03d}.json"
    parity = _locate_parity_files(plan, case, run_dir, log)
    if parity is None:
      continue
    if parity.dump is not None and not _remove_stale_dump(
        case, parity.dump, log
    ):
      continue
    command = _build_command(plan, suite_file, case, out_json, parity)
    log.commands.append(command)
    process = _invoke(command, run_dir / "driver.log", hooks, log)
    if process is None:
      break
    if process.returncode != 0:
      message = (
          f"Command failed (exit {process.returncode}): {' '.join(command)}"
      )
      log.errors.append(message)
      print(f"[aeb] {message}", file=sys.stderr)
      continue
    _collect_benchmarks(out_json, log)
    if parity.dump is None:
      continue
    if parity.reference is None:
      _record_reference(case, parity.dump, log)
    else:
      _record_comparison(plan.gate, case, parity.reference, parity.dump, log)
  return log


def _annotate_layers(plan: _RunPlan, log: _RunLog) -> list[dict[str, Any]]:
  """Normalizes the benchmark entries and joins Roofline attribution.

  Args:
    plan: Resolved run plan providing ceilings, dtype, and kernel timings.
    log: Run log; receives a failure for every benchmark the binary flagged.

  Returns:
    One layer record per benchmark entry. Records carry the Roofline fields
    when hardware ceilings are available, and the composition delta when a
    kernel timing matches.
  """
  layers = [parse_benchmark_entry(entry) for entry in log.raw_benchmarks]
  for layer_record in layers:
    if plan.ceilings is not None:
      layer_record.update(
          compute_subgraph_roofline(
              layer_record,
              plan.ceilings,
              dtype=plan.dtype,
              kernel_timings=plan.kernel_timings,
          )
      )
    elif plan.kernel_timings:
      layer_record.update(_compose_delta(layer_record, plan.kernel_timings))
    if layer_record["error_occurred"]:
      log.errors.append(
          f"{layer_record['name']}: {layer_record['error_message']}"
      )
  return layers


def _meta_params(plan: _RunPlan) -> dict[str, Any]:
  """Builds the `params` section of `meta.json` from a run plan.

  Args:
    plan: Resolved run plan.

  Returns:
    The benchmark selection, operating points, and thresholds of the run.
    `batch_size` and `seq_len` are scalars for single-point sweeps and lists
    otherwise.
  """
  gate = plan.gate
  sweep = plan.sweep
  batch_sizes = list(sweep.batch_sizes)
  seq_lengths = list(sweep.seq_lengths)
  return {
      "presets": list(sweep.presets),
      "families": list(sweep.families),
      "batch_sizes": batch_sizes,
      "seq_lengths": seq_lengths,
      "batch_size": batch_sizes[0] if len(batch_sizes) == 1 else batch_sizes,
      "seq_len": seq_lengths[0] if len(seq_lengths) == 1 else seq_lengths,
      "threads": sweep.threads,
      "min_time": plan.min_time,
      "seed": plan.seed,
      "wipe_caches": plan.wipe_caches,
      "parity_rtol": gate.rtol,
      "parity_atol": gate.atol,
      "parity_min_cosine": gate.min_cosine,
      "parity_max_l2_rel_err": gate.max_l2_rel_err,
      "parity_max_kl": gate.max_kl,
  }


def _build_meta(
    args: argparse.Namespace,
    plan: _RunPlan,
    run_dir: Path,
    started_at: str,
    hooks: RunnerHooks,
    log: _RunLog,
) -> dict[str, Any]:
  """Builds the `meta.json` provenance record of a run.

  Args:
    args: Parsed command-line flags.
    plan: Resolved run plan.
    run_dir: Run directory holding the resolved suite.
    started_at: UTC timestamp taken before the first benchmark ran.
    hooks: Host collaborators providing the machine environment and build
      provenance.
    log: Run log providing the executed commands and failures.

  Returns:
    The `aeb.layer/1` metadata dictionary.
  """
  environ = hooks.environ
  return {
      "schema": "aeb.layer/1",
      "framework": args.framework,
      "label": args.label,
      "backend": plan.backend,
      "dtype": plan.dtype,
      "track": args.track,
      "round": args.round,
      "notes": args.notes,
      "machine": environ.get("AEB_MACHINE", "unknown-machine"),
      "machine_notes": environ.get("AEB_MACHINE_NOTES", ""),
      "started_at": started_at,
      "binary": plan.binary,
      "suite_file": args.suite_file,
      "resolved_suite_file": str(run_dir / "resolved_suite.json"),
      "catalog_id": plan.suite_data.get("catalog_id", ""),
      "ceilings_json": plan.ceilings_path,
      "commands": log.commands,
      "params": _meta_params(plan),
      "build": hooks.build_info(),
      "status": "error" if log.errors else "ok",
      "errors": log.errors,
  }


def _write_artifacts(
    args: argparse.Namespace,
    plan: _RunPlan,
    run_dir: Path,
    meta: Mapping[str, Any],
    log: _RunLog,
    layers: Sequence[Mapping[str, Any]],
) -> None:
  """Writes the benchmark, parity, metadata, and summary JSON artifacts.

  Args:
    args: Parsed command-line flags.
    plan: Resolved run plan.
    run_dir: Run directory receiving the artifacts.
    meta: Provenance record from `_build_meta`.
    log: Run log providing raw benchmarks and parity records.
    layers: Annotated layer records from `_annotate_layers`.
  """
  identity = {
      "framework": args.framework,
      "label": args.label,
      "backend": plan.backend,
  }
  _dump_strict_json(
      run_dir / "benchmarks.json",
      {"context": log.context, "benchmarks": log.raw_benchmarks},
  )
  parity_passed = None
  if log.parity_records:
    parity_passed = all(
        record.get("passed", False) for record in log.parity_records
    )
    _dump_strict_json(
        run_dir / "parity.json",
        {
            "schema": "aeb.layer.parity/1",
            **identity,
            "passed": parity_passed,
            "records": log.parity_records,
        },
    )
  summary = {
      "schema": "aeb.layer.summary/1",
      **identity,
      "dtype": plan.dtype,
      "track": args.track,
      "catalog_id": plan.suite_data.get("catalog_id", ""),
      "status": meta["status"],
      "n_layers": len(layers),
      "parity_all_passed": parity_passed,
      "estimated_model_blocks_ms": estimate_model_blocks_latency(
          layers, plan.suite_data.get("model_layer_weights", {})
      ),
      "layers": layers,
      "parity": log.parity_records,
  }
  _dump_strict_json(run_dir / "meta.json", meta)
  _dump_strict_json(run_dir / "summary.json", summary)


def _run_layer_benchmark(args: argparse.Namespace, hooks: RunnerHooks) -> int:
  """Executes the resolved layer suite, Roofline join, and parity validation.

  Args:
    args: Parsed command-line flags.
    hooks: Host collaborators (run directory, sysinfo, command runner,
      environment, clock, and build provenance).

  Returns:
    0 when every benchmark and parity check succeeded, else 1.

  Raises:
    _UsageError: If a flag value, the suite, or an input file is invalid; no
      run directory has been created in that case.
  """
  plan = _plan_run(args, hooks)
  run_dir = hooks.new_run_dir("layer", args.framework, args.label)
  started_at = hooks.utc_now()
  hooks.write_sysinfo(run_dir / "sysinfo.json")
  _dump_strict_json(run_dir / "resolved_suite.json", plan.suite_data)
  run_name = f"{args.framework}/{args.label}"
  print(
      f"[aeb] layer {run_name} ({plan.backend}) -> {run_dir}",
      file=sys.stderr,
      flush=True,
  )
  if plan.dump_dir is not None:
    plan.dump_dir.mkdir(parents=True, exist_ok=True)

  log = _run_cases(plan, run_dir, hooks)
  if not log.raw_benchmarks and not log.errors:
    log.errors.append("Zero benchmark records matched filter.")
  layers = _annotate_layers(plan, log)
  meta = _build_meta(args, plan, run_dir, started_at, hooks, log)
  _write_artifacts(args, plan, run_dir, meta, log, layers)
  print(
      f"[aeb] layer {run_name}: {len(layers)} benchmarks "
      f"recorded [{meta['status']}] -> {run_dir}",
      file=sys.stderr,
  )
  return _EXIT_FAILED if log.errors else _EXIT_OK


def main(
    argv: Sequence[str] | None = None, hooks: RunnerHooks | None = None
) -> int:
  """Parses CLI flags and runs the Level 2 layer benchmark pipeline.

  Args:
    argv: Command-line arguments without the program name; `None` reads
      `sys.argv[1:]`.
    hooks: Host collaborators; `None` uses the real run directory, sysinfo
      writer, subprocess runner, process environment, clock, and build
      provenance.

  Returns:
    Process exit code: 0 when all benchmarks and parity checks succeeded, 1
    when any failed, and 2 for invalid flags or unusable input files.

  Raises:
    SystemExit: If `argv` asks for `--help` or cannot be parsed; argparse
      prints the usage and exits with status 0 or 2.
  """
  args = _build_arg_parser().parse_args(argv)
  try:
    return _run_layer_benchmark(args, hooks or RunnerHooks())
  except _UsageError as error:
    print(f"[aeb] ERROR: {error}", file=sys.stderr)
    return _EXIT_USAGE


if __name__ == "__main__":
  sys.exit(main())
