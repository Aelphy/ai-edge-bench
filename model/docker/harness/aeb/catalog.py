"""Declarative layer catalog, hyperparameter registry, and HF config scout.

Separates immutable model subgraph definitions (`layer/catalog/*.json`,
schema `aeb.layer.catalog/1`) from execution regimes, quantization fidelity
gates, `(+-Delta)` tile-cliff perturbations, and host threading profiles
(`layer/catalog/hyperparams.json`, schema `aeb.layer.hyperparams/1`).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, MutableMapping, MutableSet, Sequence
import copy
import dataclasses
import json
import math
from pathlib import Path
import sys
from typing import Any

CATALOG_SCHEMA = "aeb.layer.catalog/1"
HYPERPARAMS_SCHEMA = "aeb.layer.hyperparams/1"

LAYER_TABLES = (
    "composite_blocks",
    "attention_layers",
    "mlp_layers",
    "ple_layers",
    "norm_layers",
    "rope_layers",
    "linear_fc_layers",
)

VALID_LAYER_TYPES = frozenset({
    "TransformerBlock",
    "SelfAttention",
    "GatedDeltaNet",
    "GatedMlp",
    "PerLayerEmbedding",
    "RmsNorm",
    "RoPE",
    "FullyConnected",
})

DEFAULT_SUITE_FAMILIES = (
    "BM_RmsNorm",
    "BM_RoPE",
    "BM_GatedMlp_SwiGLU",
    "BM_GatedMlp_GeGLU",
    "BM_StandardAttention_Prefill",
    "BM_StandardAttention_Decode",
    "BM_DecodeOptimizedAttention",
    "BM_TransformerBlock_Prefill",
    "BM_TransformerBlock_Decode",
)

DECODE_ONLY_FAMILIES = frozenset({
    "BM_StandardAttention_Decode",
    "BM_DecodeOptimizedAttention",
    "BM_TransformerBlock_Decode",
})

PREFILL_FAMILIES = frozenset(DEFAULT_SUITE_FAMILIES) - DECODE_ONLY_FAMILIES

# Default registry locations, relative to the directory of a suite file.
DEFAULT_CATALOG_FILE = "../catalog/gemma_qwen_edge.json"
DEFAULT_HYPERPARAMS_FILE = "../catalog/hyperparams.json"

_ATTENTION_LAYER_TYPES = ("TransformerBlock", "SelfAttention")
_REGIME_MODES = ("prefill", "decode", "both")
# Suite keys naming a hyperparameter profile of the registry.
_SUITE_PROFILE_KEYS = ("regime", "quant_scheme", "machine", "cliff_sweep")
# Suite keys holding a file path or a profile name.
_SUITE_STRING_KEYS = (
    "catalog",
    "hyperparams",
    "ceilings_file",
    *_SUITE_PROFILE_KEYS,
)
# Suite keys holding a list of names.
_SUITE_NAME_LIST_KEYS = ("families", "models", "layer_ids")
_DEFAULT_THREAD_COUNTS = (1, 4, 8, 16)
_DEFAULT_EPS = 1e-6
_DEFAULT_VOCAB_SIZE = 32000
# Distinguishes a key that is absent from one that is explicitly null.
_UNSET = object()


class UnknownProfileError(ValueError):
  """A suite or override names a profile that `hyperparams.json` lacks.

  Attributes:
    kind: Profile kind that was looked up: `regime`, `quant_scheme`,
      `machine_threading`, or `cliff_sweep`.
    name: Profile name that was requested.
    known: Sorted names of the profiles the registry defines for `kind`.
  """

  def __init__(self, kind: str, name: Any, known: Sequence[str], source: Path):
    """Initializes the instance.

    Args:
      kind: Profile kind that was looked up.
      name: Profile name that was requested.
      known: Names of the profiles the registry defines for `kind`.
      source: Path of the registry file, for the error message.
    """
    self.kind = kind
    self.name = name
    self.known = tuple(sorted(known))
    super().__init__(
        f"Unknown {kind} {name!r} in {source}; known: {', '.join(self.known)}"
    )


@dataclasses.dataclass(frozen=True)
class _Registry:
  """A validated hyperparameter registry and the file it was loaded from.

  Attributes:
    tables: Parsed `aeb.layer.hyperparams/1` document.
    path: File the registry was read from; relative `ceilings_file` entries
      resolve against its directory.
  """

  tables: Mapping[str, Any]
  path: Path

  def profile(self, table: str, kind: str, name: Any) -> dict[str, Any]:
    """Returns the profile called `name` from a registry table.

    Args:
      table: Registry table to search, e.g. `machine_threading`.
      kind: Singular label of `table` for error messages, e.g. `regime`.
      name: Profile name to look up.

    Returns:
      The profile dictionary.

    Raises:
      UnknownProfileError: If `table` defines no profile called `name`.
    """
    profiles = self.tables[table]
    profile = profiles.get(name) if isinstance(name, str) else None
    if profile is None:
      raise UnknownProfileError(kind, name, list(profiles), self.path)
    return profile


@dataclasses.dataclass(frozen=True)
class _TextDims:
  """Decoder dimensions shared by all blocks of a scouted HF text model.

  Attributes:
    embed_dim: Residual stream width (`hidden_size`).
    hidden_dim: Feed-forward inner width (`intermediate_size`).
    num_layers: Number of decoder blocks.
    mlp_activation: Gated MLP flavor, `GeGLU` or `SwiGLU`.
    num_heads: Query heads per attention layer.
    num_kv_heads: Key/value heads of local (sliding-window) layers.
    global_kv_heads: Key/value heads of global (full-attention) layers.
    head_dim: Per-head width of local layers.
    global_head_dim: Per-head width of global layers.
    sliding_window: Raw `sliding_window` config value (may be `None`).
    ple_dim: Per-layer embedding width, 0 when the model has none.
    kv_shared_layers: Trailing layers that reuse an earlier layer's KV cache.
    k_eq_v: Whether global layers share one projection for keys and values.
    gated_q: Whether the query projection carries an output gate.
    qk_norm: Whether queries and keys are RMS-normalized.
    post_norms: Whether blocks apply post-attention and post-MLP norms.
    eps: RMS normalization epsilon.
  """

  embed_dim: int
  hidden_dim: int
  num_layers: int
  mlp_activation: str
  num_heads: int
  num_kv_heads: int
  global_kv_heads: int
  head_dim: int
  global_head_dim: int
  sliding_window: Any
  ple_dim: int
  kv_shared_layers: int
  k_eq_v: bool
  gated_q: bool
  qk_norm: bool
  post_norms: bool
  eps: float


@dataclasses.dataclass
class _CatalogPresets:
  """Presets resolved from catalog blocks for one suite.

  Attributes:
    presets: Suite presets followed by one preset per runnable catalog block.
    skipped_layers: Block IDs without a `layer_bench` preset (e.g. linear
      attention blocks).
    approximated: Preset name to the block features `layer_bench` approximates.
    remap: Deduplicated preset name to the canonical preset that replaces it.
  """

  presets: list[dict[str, Any] | str]
  skipped_layers: list[str] = dataclasses.field(default_factory=list)
  approximated: dict[str, list[str]] = dataclasses.field(default_factory=dict)
  remap: dict[str, str] = dataclasses.field(default_factory=dict)


def _is_pos_int(value: Any) -> bool:
  """Returns whether `value` is a positive integer (booleans excluded)."""
  return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_real(value: Any) -> bool:
  """Returns whether `value` is an `int` or `float` (booleans excluded)."""
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_pos_finite_num(value: Any) -> bool:
  """Returns whether `value` is a positive, finite real number."""
  return _is_real(value) and math.isfinite(value) and value > 0


def _as_float(value: Any) -> float:
  """Returns `value` as a float, or NaN when it is not a real number."""
  return float(value) if _is_real(value) else math.nan


def _check_pos_int_list(values: Any, label: str) -> None:
  """Checks that `values` is a non-empty list of positive integers.

  Args:
    values: Candidate value parsed from JSON.
    label: Human-readable location of `values` for the error message.

  Raises:
    ValueError: If `values` is not a non-empty list of positive integers.
  """
  if not isinstance(values, list) or not values:
    raise ValueError(f"{label} must be a non-empty list of positive integers")
  for item in values:
    if not _is_pos_int(item):
      raise ValueError(f"{label} contains non-positive integer: {item!r}")


def load_json_object(path: Path) -> dict[str, Any]:
  """Reads a UTF-8 JSON file whose top-level value is an object.

  Args:
    path: File to read.

  Returns:
    The parsed JSON object.

  Raises:
    OSError: If `path` cannot be read.
    ValueError: If the content is not valid JSON or its top-level value is
      not an object.
  """
  text = path.read_text(encoding="utf-8")
  try:
    document = json.loads(text)
  except json.JSONDecodeError as error:
    raise ValueError(f"{path}: invalid JSON: {error}") from error
  if not isinstance(document, dict):
    raise ValueError(
        f"{path}: expected a JSON object, got {type(document).__name__}"
    )
  return document


def _dump_json(document: Mapping[str, Any]) -> str:
  """Serializes `document` as indented, strict (NaN-free) JSON text.

  Args:
    document: JSON-compatible mapping.

  Returns:
    JSON text terminated by a newline.

  Raises:
    ValueError: If `document` contains a non-finite float.
  """
  return json.dumps(document, indent=2, allow_nan=False) + "\n"


def filter_families_for_mode(families: Sequence[str], mode: str) -> list[str]:
  """Filters benchmark family names to match a regime's execution mode.

  Args:
    families: Candidate benchmark family names (e.g. `BM_RmsNorm`).
    mode: Execution mode (`"prefill"`, `"decode"`, or `"both"`).

  Returns:
    Filtered list of family names matching `mode`.
  """
  if mode == "decode":
    return [name for name in families if name in DECODE_ONLY_FAMILIES]
  if mode == "prefill":
    return [name for name in families if name in PREFILL_FAMILIES]
  return list(families)


def _validate_layer_entry(
    entry: Any, table: str, seen_ids: MutableSet[str], source: str
) -> str:
  """Validates the fields every catalog layer entry must carry.

  Args:
    entry: Candidate entry of a layer table.
    table: Name of the layer table holding `entry`.
    seen_ids: Layer IDs encountered so far; updated with the entry's ID.
    source: Human-readable path or identifier for error messages.

  Returns:
    The entry's layer ID.

  Raises:
    ValueError: If `entry` is not an object, or its `id`, `layer_type`, or
      `params` is missing, duplicated, or unsupported.
  """
  if not isinstance(entry, dict):
    raise ValueError(f"{source}: '{table}' entry must be an object")
  layer_id = entry.get("id", "")
  if not isinstance(layer_id, str) or not layer_id:
    raise ValueError(f"{source}: '{table}' entry missing non-empty 'id'")
  if layer_id in seen_ids:
    raise ValueError(f"{source}: duplicate layer id {layer_id!r}")
  seen_ids.add(layer_id)

  layer_type = entry.get("layer_type", "")
  if not isinstance(layer_type, str) or layer_type not in VALID_LAYER_TYPES:
    raise ValueError(
        f"{source}: {layer_id} has unsupported layer_type {layer_type!r}; "
        f"expected one of {sorted(VALID_LAYER_TYPES)}"
    )
  if not isinstance(entry.get("params"), dict):
    raise ValueError(f"{source}: {layer_id} missing 'params' object")
  return layer_id


def _validate_composite_block(
    entry: Mapping[str, Any],
    layer_id: str,
    seen_presets: MutableSet[str],
    source: str,
) -> None:
  """Validates the preset name and dimensions of a `composite_blocks` entry.

  Args:
    entry: Entry that already passed `_validate_layer_entry`.
    layer_id: ID of `entry`.
    seen_presets: Preset names encountered so far; updated with the entry's.
    source: Human-readable path or identifier for error messages.

  Raises:
    ValueError: If the preset name is malformed or duplicated, or
      `embed_dim`, `hidden_dim`, or `eps` is invalid.
  """
  name = entry.get("preset_name", "")
  if not isinstance(name, str):
    raise ValueError(f"{source}: {layer_id} preset_name must be a string")
  if name:
    if "/" in name:
      raise ValueError(
          f"{source}: {layer_id} preset_name {name!r} must not contain '/'"
      )
    if name in seen_presets:
      raise ValueError(f"{source}: duplicate preset_name {name!r}")
    seen_presets.add(name)
  params = entry["params"]
  if not _is_pos_int(params.get("embed_dim")):
    raise ValueError(f"{source}: {layer_id} requires positive embed_dim")
  if not _is_pos_int(params.get("hidden_dim")):
    raise ValueError(f"{source}: {layer_id} requires positive hidden_dim")
  eps = params.get("eps", _DEFAULT_EPS)
  if not _is_pos_finite_num(eps):
    raise ValueError(
        f"{source}: {layer_id} requires positive finite eps, got {eps!r}"
    )


def _validate_attention_params(
    params: Mapping[str, Any], layer_id: str, source: str
) -> None:
  """Validates multi-head attention shapes; no-op when `num_heads` is null.

  Args:
    params: `params` object of a `TransformerBlock` or `SelfAttention` entry.
    layer_id: ID of the entry owning `params`.
    source: Human-readable path or identifier for error messages.

  Raises:
    ValueError: If head counts, `head_dim`, or `rotary_dim` violate the
      grouped-query attention and RoPE shape invariants.
  """
  num_heads = params.get("num_heads")
  if num_heads is None:
    return
  num_kv_heads = params.get("num_kv_heads")
  head_dim = params.get("head_dim")
  if not (
      _is_pos_int(num_heads)
      and _is_pos_int(num_kv_heads)
      and _is_pos_int(head_dim)
  ):
    raise ValueError(
        f"{source}: {layer_id} requires positive num_heads, "
        "num_kv_heads, and head_dim"
    )
  if head_dim % 2 != 0:
    raise ValueError(
        f"{source}: {layer_id} head_dim ({head_dim}) must be even (for RoPE)"
    )
  if num_heads % num_kv_heads != 0:
    raise ValueError(
        f"{source}: {layer_id} num_heads ({num_heads}) not divisible by "
        f"num_kv_heads ({num_kv_heads})"
    )
  rotary_dim = params.get("rotary_dim")
  if rotary_dim is None:
    rotary_dim = head_dim
  if not _is_pos_int(rotary_dim) or rotary_dim > head_dim or rotary_dim % 2:
    raise ValueError(
        f"{source}: {layer_id} rotary_dim ({rotary_dim}) must be even in "
        f"2..{head_dim}"
    )


def _validate_layer_tables(
    catalog: Mapping[str, Any], source: str
) -> dict[str, dict[str, Any]]:
  """Validates every layer table of a catalog.

  Args:
    catalog: Parsed catalog dictionary.
    source: Human-readable path or identifier for error messages.

  Returns:
    The `composite_blocks` entries keyed by layer ID.

  Raises:
    ValueError: If a table is not a list or any entry is invalid.
  """
  seen_ids: set[str] = set()
  seen_presets: set[str] = set()
  blocks_by_id: dict[str, dict[str, Any]] = {}
  for table in LAYER_TABLES:
    entries = catalog.get(table, [])
    if not isinstance(entries, list):
      raise ValueError(f"{source}: '{table}' must be a list")
    for entry in entries:
      layer_id = _validate_layer_entry(entry, table, seen_ids, source)
      if table == "composite_blocks":
        _validate_composite_block(entry, layer_id, seen_presets, source)
        blocks_by_id[layer_id] = entry
      if entry["layer_type"] in _ATTENTION_LAYER_TYPES:
        _validate_attention_params(entry["params"], layer_id, source)
  return blocks_by_id


def _validate_model(
    model_name: str,
    model_info: Any,
    blocks_by_id: Mapping[str, Mapping[str, Any]],
    source: str,
) -> None:
  """Validates one `models` entry and its block multiplicities.

  Args:
    model_name: Key of the entry in the catalog's `models` object.
    model_info: Candidate model entry.
    blocks_by_id: Validated `composite_blocks` entries keyed by layer ID.
    source: Human-readable path or identifier for error messages.

  Raises:
    ValueError: If the entry is not an object, a dimension is invalid, or
      `block_composition` references unknown blocks, mismatches `embed_dim`,
      or does not sum to `total_layers`.
  """
  prefix = f"{source}: model {model_name}"
  if not isinstance(model_info, dict):
    raise ValueError(f"{prefix} must be an object")
  total_layers = model_info.get("total_layers")
  embed_dim = model_info.get("embed_dim")
  if not _is_pos_int(total_layers):
    raise ValueError(f"{prefix} invalid total_layers")
  if not _is_pos_int(embed_dim):
    raise ValueError(f"{prefix} invalid embed_dim")
  if not _is_pos_finite_num(model_info.get("eps", _DEFAULT_EPS)):
    raise ValueError(f"{prefix} invalid eps")
  composition = model_info.get("block_composition", [])
  if not isinstance(composition, list) or not composition:
    raise ValueError(f"{prefix} requires non-empty block_composition")

  layer_count = 0
  for item in composition:
    if not isinstance(item, dict):
      raise ValueError(f"{prefix} block_composition entry must be an object")
    block_id = item.get("layer_id", "")
    count = item.get("count", 0)
    block = blocks_by_id.get(block_id) if isinstance(block_id, str) else None
    if block is None:
      raise ValueError(f"{prefix} references unknown block {block_id!r}")
    if not _is_pos_int(count):
      raise ValueError(f"{prefix} block {block_id!r} has invalid count")
    block_embed_dim = block["params"]["embed_dim"]
    if block_embed_dim != embed_dim:
      raise ValueError(
          f"{prefix} embed_dim ({embed_dim}) mismatches {block_id} "
          f"({block_embed_dim})"
      )
    layer_count += count
  if layer_count != total_layers:
    raise ValueError(
        f"{prefix} block_composition sum ({layer_count}) != total_layers "
        f"({total_layers})"
    )


def validate_catalog(
    catalog: Mapping[str, Any], source: str = "<catalog>"
) -> None:
  """Validates `aeb.layer.catalog/1` invariants and model layer counts.

  Args:
    catalog: Parsed JSON dictionary for a layer catalog.
    source: Human-readable path or identifier for error messages.

  Raises:
    ValueError: If any schema, layer_type enum, shape divisibility, ID
      uniqueness, or model block-multiplicity invariant is violated.
  """
  if not isinstance(catalog, Mapping):
    raise ValueError(f"{source}: catalog must be an object")
  if catalog.get("schema") != CATALOG_SCHEMA:
    raise ValueError(
        f"{source}: expected schema {CATALOG_SCHEMA!r}, "
        f"got {catalog.get('schema')!r}"
    )
  catalog_id = catalog.get("catalog_id")
  if not isinstance(catalog_id, str) or not catalog_id:
    raise ValueError(f"{source}: missing non-empty 'catalog_id'")

  blocks_by_id = _validate_layer_tables(catalog, source)
  models = catalog.get("models")
  if not isinstance(models, dict) or not models:
    raise ValueError(f"{source}: 'models' must be a non-empty object")
  for model_name, model_info in models.items():
    _validate_model(model_name, model_info, blocks_by_id, source)


def _validate_regimes(regimes: Any, source: str) -> None:
  """Validates the `regimes` table of a hyperparameter registry.

  Args:
    regimes: Candidate `regimes` value.
    source: Human-readable path or identifier for error messages.

  Raises:
    ValueError: If the table is empty or a regime has an invalid mode or
      operating-point list.
  """
  if not isinstance(regimes, dict) or not regimes:
    raise ValueError(f"{source}: missing non-empty 'regimes'")
  for name, regime in regimes.items():
    if not isinstance(regime, dict):
      raise ValueError(f"{source}: regime {name} must be an object")
    if regime.get("mode") not in _REGIME_MODES:
      raise ValueError(f"{source}: regime {name} invalid mode")
    _check_pos_int_list(
        regime.get("batch_sizes"), f"{source}:regimes.{name}.batch_sizes"
    )
    _check_pos_int_list(
        regime.get("seq_lengths"), f"{source}:regimes.{name}.seq_lengths"
    )


def _validate_quality_gate(gate: Any, scheme_name: str, source: str) -> None:
  """Validates the numerical fidelity thresholds of a quantization scheme.

  Args:
    gate: Candidate `quality_gate` value.
    scheme_name: Name of the scheme owning `gate`.
    source: Human-readable path or identifier for error messages.

  Raises:
    ValueError: If `gate` is not an object or a threshold is missing,
      non-finite, or out of range.
  """
  prefix = f"{source}: quant_scheme {scheme_name}"
  if not isinstance(gate, dict):
    raise ValueError(f"{prefix} missing quality_gate")
  rtol = _as_float(gate.get("rtol", -1.0))
  atol = _as_float(gate.get("atol", -1.0))
  if not (math.isfinite(rtol) and math.isfinite(atol)) or rtol < 0 or atol < 0:
    raise ValueError(f"{prefix} invalid rtol/atol ({rtol}, {atol})")
  min_cosine = _as_float(gate.get("min_cosine", 0.0))
  if not math.isfinite(min_cosine) or not 0.0 <= min_cosine <= 1.0:
    raise ValueError(f"{prefix} invalid min_cosine")
  for key in ("max_l2_rel_err", "max_kl_divergence"):
    threshold = _as_float(gate.get(key, 0.0))
    if not math.isfinite(threshold) or threshold < 0:
      raise ValueError(f"{prefix} invalid {key}")


def _validate_quant_schemes(schemes: Any, source: str) -> None:
  """Validates the `quant_schemes` table of a hyperparameter registry.

  Args:
    schemes: Candidate `quant_schemes` value.
    source: Human-readable path or identifier for error messages.

  Raises:
    ValueError: If the table is empty or a scheme lacks a weight dtype or a
      valid quality gate.
  """
  if not isinstance(schemes, dict) or not schemes:
    raise ValueError(f"{source}: missing non-empty 'quant_schemes'")
  for name, scheme in schemes.items():
    if not isinstance(scheme, dict):
      raise ValueError(f"{source}: quant_scheme {name} must be an object")
    _validate_quality_gate(scheme.get("quality_gate"), name, source)
    weight_dtype = scheme.get("weight_dtype")
    if not isinstance(weight_dtype, str) or not weight_dtype:
      raise ValueError(f"{source}: quant_scheme {name} missing weight_dtype")


def _validate_cliff_sweeps(sweeps: Any, source: str) -> None:
  """Validates the `cliff_sweeps` table of a hyperparameter registry.

  Args:
    sweeps: Candidate `cliff_sweeps` value.
    source: Human-readable path or identifier for error messages.

  Raises:
    ValueError: If the table is empty or a sweep has invalid `deltas` or
      `target_dims`.
  """
  if not isinstance(sweeps, dict) or not sweeps:
    raise ValueError(f"{source}: missing non-empty 'cliff_sweeps'")
  for name, sweep in sweeps.items():
    if not isinstance(sweep, dict):
      raise ValueError(f"{source}: cliff_sweep {name} must be an object")
    deltas = sweep.get("deltas")
    if not isinstance(deltas, list) or not deltas:
      raise ValueError(f"{source}: cliff_sweep {name} requires deltas list")
    for delta in deltas:
      if not isinstance(delta, int) or isinstance(delta, bool):
        raise ValueError(
            f"{source}: cliff_sweep {name} non-int delta {delta!r}"
        )
    target_dims = sweep.get("target_dims", [])
    if not isinstance(target_dims, list) or not all(
        isinstance(dim, str) for dim in target_dims
    ):
      raise ValueError(
          f"{source}: cliff_sweep {name} target_dims must be a list of strings"
      )


def _validate_machine_threading(
    machines: Any, source: str, base_dir: Path | None
) -> None:
  """Validates the `machine_threading` table of a hyperparameter registry.

  Args:
    machines: Candidate `machine_threading` value.
    source: Human-readable path or identifier for error messages.
    base_dir: Directory that relative `ceilings_file` paths resolve against;
      `None` skips the file existence check.

  Raises:
    ValueError: If the table is empty or a machine has invalid thread
      counts, core count, or `ceilings_file`.
  """
  if not isinstance(machines, dict) or not machines:
    raise ValueError(f"{source}: missing non-empty 'machine_threading'")
  for name, machine in machines.items():
    if not isinstance(machine, dict):
      raise ValueError(f"{source}: machine {name} must be an object")
    _check_pos_int_list(
        machine.get("thread_counts"),
        f"{source}:machine_threading.{name}.thread_counts",
    )
    cores = machine.get("physical_cores_per_domain")
    if cores is not None and not _is_pos_int(cores):
      raise ValueError(
          f"{source}: machine {name} invalid physical_cores_per_domain"
      )
    ceilings_file = machine.get("ceilings_file", "")
    if not isinstance(ceilings_file, str):
      raise ValueError(
          f"{source}: machine {name} ceilings_file must be a string"
      )
    if ceilings_file and base_dir is not None:
      resolved = (base_dir / ceilings_file).resolve()
      if not resolved.is_file():
        raise ValueError(
            f"{source}: machine {name} ceilings_file not found: {resolved}"
        )


def validate_hyperparams(
    hyperparams: Mapping[str, Any],
    source: str = "<hyperparams>",
    base_dir: Path | None = None,
) -> None:
  """Validates `aeb.layer.hyperparams/1` regimes, quant gates, and machines.

  Args:
    hyperparams: Parsed JSON dictionary for a hyperparameter registry.
    source: Human-readable path or identifier for error messages.
    base_dir: Directory that relative `ceilings_file` paths resolve against;
      `None` skips checking that those files exist.

  Raises:
    ValueError: If schema, regimes, quantization gates, cliff sweeps, or
      machine threading profiles are invalid.
  """
  if not isinstance(hyperparams, Mapping):
    raise ValueError(f"{source}: hyperparams must be an object")
  if hyperparams.get("schema") != HYPERPARAMS_SCHEMA:
    raise ValueError(
        f"{source}: expected schema {HYPERPARAMS_SCHEMA!r}, "
        f"got {hyperparams.get('schema')!r}"
    )
  _validate_regimes(hyperparams.get("regimes"), source)
  _validate_quant_schemes(hyperparams.get("quant_schemes"), source)
  _validate_cliff_sweeps(hyperparams.get("cliff_sweeps"), source)
  _validate_machine_threading(
      hyperparams.get("machine_threading"), source, base_dir
  )


def _has_post_norms(params: Mapping[str, Any]) -> bool:
  """Returns whether a block applies post-attention and post-MLP norms."""
  four_norms = int(params.get("num_block_norms") or 2) >= 4
  return bool(params.get("post_norms", four_norms))


def _block_weight_and_norm_elements(
    params: Mapping[str, Any],
) -> tuple[int, int]:
  """Counts the trainable elements of a transformer block.

  Args:
    params: Block parameter dictionary from a `composite_blocks` entry.

  Returns:
    Tuple `(weight_elems, norm_elems)`: elements of the projection, MLP, and
    per-layer-embedding matrices, and of the normalization scale vectors.
  """
  embed_dim = int(params["embed_dim"])
  hidden_dim = int(params["hidden_dim"])
  num_heads = int(params.get("num_heads") or 0)
  num_kv_heads = int(params.get("num_kv_heads") or 0)
  head_dim = int(params.get("head_dim") or 0)

  query_out = num_heads * head_dim * (2 if params.get("gated_q_proj") else 1)
  if params.get("shared_kv", False):
    kv_projections = 0
  else:
    kv_projections = 1 if params.get("k_eq_v", False) else 2
  kv_out = kv_projections * num_kv_heads * head_dim
  output_in = num_heads * head_dim
  weight_elems = (
      query_out * embed_dim
      + kv_out * embed_dim
      + embed_dim * output_in
      + 3 * hidden_dim * embed_dim
  )
  norm_elems = 4 * embed_dim if _has_post_norms(params) else 2 * embed_dim
  if params.get("qk_norm", False):
    norm_elems += 2 * head_dim
  ple_dim = int(params.get("ple_dim") or 0)
  if ple_dim > 0:
    weight_elems += 2 * ple_dim * embed_dim
    norm_elems += embed_dim
  return weight_elems, norm_elems


def estimate_block_roofline_work(
    params: Mapping[str, Any],
    batch_size: int = 1,
    seq_len_q: int = 1024,
    seq_len_kv: int = 1024,
    bits_per_weight: float = 32.0,
    act_bytes: int = 4,
    kv_bytes: int = 4,
) -> dict[str, float]:
  """Computes analytical FLOPs and minimum boundary DRAM bytes for a block.

  Args:
    params: Block parameter dictionary from a `composite_blocks` entry.
    batch_size: Batch size `B`.
    seq_len_q: Query sequence length `T_q` (1 for single-token decode).
    seq_len_kv: KV cache length `S_kv`.
    bits_per_weight: Effective bits per weight parameter.
    act_bytes: Bytes per activation element.
    kv_bytes: Bytes per KV cache element.

  Returns:
    Dictionary with `theoretical_flops`, `min_boundary_bytes`, and
    `arithmetic_intensity`.
  """
  embed_dim = int(params["embed_dim"])
  num_heads = int(params.get("num_heads") or 0)
  num_kv_heads = int(params.get("num_kv_heads") or 0)
  head_dim = int(params.get("head_dim") or 0)
  window = params.get("sliding_window")
  # Sliding-window KV clamping applies in single-token decode (T_q == 1); full
  # sequence prefill executes dense T_q x S_kv masked attention.
  if seq_len_q == 1 and _is_pos_int(window):
    kv_len = min(seq_len_kv, int(window))
  else:
    kv_len = seq_len_kv
  tokens = batch_size * seq_len_q
  weight_elems, norm_elems = _block_weight_and_norm_elements(params)

  # Projections + MLP GEMMs: 2 * M_tokens * N * K FLOPs each.
  gemm_flops = 2.0 * tokens * weight_elems
  sdpa_flops = 4.0 * batch_size * num_heads * seq_len_q * kv_len * head_dim
  total_flops = gemm_flops + sdpa_flops

  weight_bytes = weight_elems * (bits_per_weight / 8.0) + 4.0 * norm_elems
  io_bytes = 2.0 * tokens * embed_dim * act_bytes
  kv_state_bytes = (
      2.0 * batch_size * num_kv_heads * kv_len * head_dim * kv_bytes
  )
  has_mask = seq_len_q > 1 and num_heads > 0
  mask_bytes = 4.0 * seq_len_q * kv_len if has_mask else 0.0
  min_bytes = weight_bytes + io_bytes + kv_state_bytes + mask_bytes
  return {
      "theoretical_flops": total_flops,
      "min_boundary_bytes": min_bytes,
      "arithmetic_intensity": total_flops / max(min_bytes, 1.0),
  }


def _partial_rotary_factor(
    text_config: Mapping[str, Any], layer_kind: str
) -> Any:
  """Finds the most specific `partial_rotary_factor` for a layer kind.

  Args:
    text_config: HF text decoder configuration.
    layer_kind: HF `layer_types` entry, e.g. `full_attention`.

  Returns:
    The raw factor from `rope_parameters[layer_kind]`, else from
    `rope_parameters`, else from the top level of `text_config`; `None` when
    none of them sets one.
  """
  rope_parameters = text_config.get("rope_parameters")
  if isinstance(rope_parameters, dict):
    per_kind = rope_parameters.get(layer_kind)
    if isinstance(per_kind, dict):
      factor = per_kind.get("partial_rotary_factor")
      if factor is not None:
        return factor
    factor = rope_parameters.get("partial_rotary_factor")
    if factor is not None:
      return factor
  return text_config.get("partial_rotary_factor")


def _infer_rotary_dim(
    text_config: Mapping[str, Any], layer_kind: str, head_dim: int
) -> int:
  """Infers the number of RoPE-rotated dimensions per attention head.

  Without a partial rotary factor the whole head rotates. Otherwise
  `head_dim * factor` is rounded to the nearest even integer, because RoPE
  rotates dimension pairs; a product that is exactly an odd integer rounds
  down. The result is clamped to `[2, head_dim]` (to the largest even value
  below an odd `head_dim`), matching what `validate_catalog` accepts.

  Args:
    text_config: HF text decoder configuration.
    layer_kind: HF `layer_types` entry, e.g. `full_attention`.
    head_dim: Per-head width of the layer.

  Returns:
    `head_dim` when no factor is configured, else an even integer in
    `[2, head_dim]`.

  Raises:
    ValueError: If a factor is configured but is not a positive finite
      number, or `head_dim` is smaller than 2.
  """
  factor = _partial_rotary_factor(text_config, layer_kind)
  if factor is None:
    return head_dim
  if not _is_pos_finite_num(factor):
    raise ValueError(
        "partial_rotary_factor must be a positive finite number, got "
        f"{factor!r}"
    )
  if head_dim < 2:
    raise ValueError(
        f"partial_rotary_factor requires head_dim >= 2, got {head_dim}"
    )
  nearest_even = 2 * math.ceil(head_dim * factor / 2.0 - 0.5)
  return min(max(nearest_even, 2), head_dim - head_dim % 2)


def _explicit_flag(config: Mapping[str, Any], *keys: str) -> bool | None:
  """Reads the first boolean flag that a configuration sets explicitly.

  Args:
    config: Configuration dictionary.
    *keys: Candidate keys, by decreasing priority.

  Returns:
    The truthiness of the first of `keys` present in `config` (an explicit
    null counts as `False`), or `None` when none of them is present.
  """
  for key in keys:
    value = config.get(key, _UNSET)
    if value is not _UNSET:
      return bool(value)
  return None


def _infer_norm_flags(
    text_config: Mapping[str, Any], family: str
) -> tuple[bool, bool]:
  """Infers `(qk_norm, post_norms)` from HF config keys and model_type.

  Args:
    text_config: HF text decoder configuration.
    family: HF `model_type`; decides the flags the config leaves unset.

  Returns:
    Tuple `(qk_norm, post_norms)`. Explicit `qk_norm`/`attn_qk_norm` and
    `post_norms` keys win; otherwise four block norms imply post-norms, and
    the model family supplies the remaining defaults.
  """
  family_lower = family.lower()
  qk_norm = _explicit_flag(text_config, "qk_norm", "attn_qk_norm")
  if qk_norm is None:
    qk_norm = any(tag in family_lower for tag in ("gemma3", "gemma4", "qwen3"))
  post_norms = _explicit_flag(text_config, "post_norms")
  if post_norms is None:
    post_norms = int(text_config.get("num_block_norms") or 0) >= 4 or any(
        tag in family_lower for tag in ("gemma2", "gemma3", "gemma4")
    )
  return qk_norm, post_norms


def _require_pos_int(text_config: Mapping[str, Any], key: str) -> int:
  """Returns the positive integer `text_config[key]`.

  Args:
    text_config: HF text decoder configuration.
    key: Required configuration key.

  Returns:
    The value of `key`.

  Raises:
    ValueError: If `key` is missing or is not a positive integer.
  """
  value = text_config.get(key)
  if not _is_pos_int(value):
    raise ValueError(
        f"HF config requires a positive integer {key!r}, got {value!r}"
    )
  return value


def _parse_text_dims(text_config: Mapping[str, Any], family: str) -> _TextDims:
  """Extracts decoder dimensions from an HF text configuration.

  Args:
    text_config: HF text decoder configuration.
    family: HF `model_type`, used to infer normalization defaults.

  Returns:
    The decoder dimensions, with HF defaults applied to null or missing keys.

  Raises:
    ValueError: If a required dimension is missing or not a positive integer.
  """
  embed_dim = _require_pos_int(text_config, "hidden_size")
  num_heads = _require_pos_int(text_config, "num_attention_heads")
  num_kv_heads = int(text_config.get("num_key_value_heads") or num_heads)
  head_dim = int(text_config.get("head_dim") or (embed_dim // num_heads))
  raw_activation = str(
      text_config.get("hidden_activation")
      or text_config.get("hidden_act")
      or "silu"
  ).lower()
  qk_norm, post_norms = _infer_norm_flags(text_config, family)
  return _TextDims(
      embed_dim=embed_dim,
      hidden_dim=_require_pos_int(text_config, "intermediate_size"),
      num_layers=_require_pos_int(text_config, "num_hidden_layers"),
      mlp_activation="GeGLU" if "gelu" in raw_activation else "SwiGLU",
      num_heads=num_heads,
      num_kv_heads=num_kv_heads,
      global_kv_heads=int(
          text_config.get("num_global_key_value_heads") or num_kv_heads
      ),
      head_dim=head_dim,
      global_head_dim=int(text_config.get("global_head_dim") or head_dim),
      sliding_window=text_config.get("sliding_window"),
      ple_dim=int(text_config.get("hidden_size_per_layer_input") or 0),
      kv_shared_layers=int(text_config.get("num_kv_shared_layers") or 0),
      k_eq_v=bool(text_config.get("attention_k_eq_v", False)),
      gated_q=bool(text_config.get("attn_output_gate", False)),
      qk_norm=qk_norm,
      post_norms=post_norms,
      eps=float(text_config.get("rms_norm_eps") or _DEFAULT_EPS),
  )


def _resolve_layer_types(
    text_config: Mapping[str, Any], dims: _TextDims
) -> list[str]:
  """Determines the attention kind of every decoder layer.

  Args:
    text_config: HF text decoder configuration.
    dims: Decoder dimensions of the model.

  Returns:
    One `layer_types` entry per layer, in depth order: the config's own list
    when it has one entry per layer; else every `sliding_window_pattern`-th
    layer is `full_attention` and the rest `sliding_attention`; else all
    layers are `full_attention`.
  """
  layer_types = text_config.get("layer_types")
  if isinstance(layer_types, list) and len(layer_types) == dims.num_layers:
    return layer_types
  pattern = int(text_config.get("sliding_window_pattern") or 0)
  if pattern > 1 and _is_pos_int(dims.sliding_window):
    return [
        "full_attention" if (index + 1) % pattern == 0 else "sliding_attention"
        for index in range(dims.num_layers)
    ]
  return ["full_attention"] * dims.num_layers


def _base_block_params(dims: _TextDims) -> dict[str, Any]:
  """Builds block parameters with the multi-head attention fields unset.

  Args:
    dims: Decoder dimensions of the model.

  Returns:
    Parameter dictionary of a linear-attention block; attention blocks
    overwrite its null and false attention fields.
  """
  return {
      "embed_dim": dims.embed_dim,
      "hidden_dim": dims.hidden_dim,
      "mlp_activation": dims.mlp_activation,
      "num_heads": None,
      "num_kv_heads": None,
      "head_dim": None,
      "rotary_dim": None,
      "sliding_window": None,
      "gated_q_proj": False,
      "shared_kv": False,
      "k_eq_v": False,
      "qk_norm": False,
      "post_norms": dims.post_norms,
      "num_block_norms": 4 if dims.post_norms else 2,
      "ple_dim": dims.ple_dim,
      "eps": dims.eps,
  }


def _attention_block_params(
    dims: _TextDims,
    text_config: Mapping[str, Any],
    layer_kind: str,
    is_shared: bool,
) -> dict[str, Any]:
  """Returns block parameters of a multi-head attention layer.

  Args:
    dims: Decoder dimensions of the model.
    text_config: HF text decoder configuration (for RoPE settings).
    layer_kind: HF `layer_types` entry; `sliding_attention` selects the local
      head shape, anything else the global one.
    is_shared: Whether the layer reuses an earlier layer's KV cache.

  Returns:
    Parameter dictionary for a `composite_blocks` entry.

  Raises:
    ValueError: If the configured partial rotary factor is invalid.
  """
  is_local = layer_kind == "sliding_attention"
  head_dim = dims.head_dim if is_local else dims.global_head_dim
  has_window = is_local and _is_pos_int(dims.sliding_window)
  params = _base_block_params(dims)
  params.update(
      num_heads=dims.num_heads,
      num_kv_heads=dims.num_kv_heads if is_local else dims.global_kv_heads,
      head_dim=head_dim,
      rotary_dim=_infer_rotary_dim(text_config, layer_kind, head_dim),
      sliding_window=int(dims.sliding_window) if has_window else None,
      gated_q_proj=dims.gated_q,
      shared_kv=is_shared,
      k_eq_v=dims.k_eq_v and not is_local,
      qk_norm=dims.qk_norm,
  )
  return params


def _attention_archetype(
    dims: _TextDims, is_local: bool, is_shared: bool, mixed_kinds: bool
) -> str:
  """Names the block archetype of a multi-head attention layer.

  Args:
    dims: Decoder dimensions of the model.
    is_local: Whether the layer uses sliding-window attention.
    is_shared: Whether the layer reuses an earlier layer's KV cache.
    mixed_kinds: Whether the model has any layer that is not full attention.

  Returns:
    Archetype suffix used in block IDs and preset names.
  """
  if dims.kv_shared_layers > 0:
    scope = "Local" if is_local else "Global"
    return scope + ("SharedKV" if is_shared else "Producer")
  if is_local:
    return "LocalSliding"
  if dims.gated_q:
    return "GatedFullAttention"
  if mixed_kinds:
    return "GlobalFull"
  return "FullAttention"


def _scout_archetypes(
    text_config: Mapping[str, Any], dims: _TextDims
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
  """Groups a model's layers into block archetypes.

  Args:
    text_config: HF text decoder configuration.
    dims: Decoder dimensions of the model.

  Returns:
    Tuple `(counts, params)` keyed by archetype name in first-occurrence
    order: the number of layers per archetype and its block parameters.

  Raises:
    ValueError: If the configured partial rotary factor is invalid.
  """
  layer_types = _resolve_layer_types(text_config, dims)
  mixed_kinds = any(kind != "full_attention" for kind in layer_types)
  if dims.kv_shared_layers > 0:
    first_shared_index = dims.num_layers - dims.kv_shared_layers
  else:
    first_shared_index = dims.num_layers
  counts: dict[str, int] = {}
  params_by_archetype: dict[str, dict[str, Any]] = {}
  for index, layer_kind in enumerate(layer_types):
    is_shared = index >= first_shared_index
    if layer_kind == "linear_attention":
      archetype = "GatedDeltaNet"
      params = _base_block_params(dims)
    else:
      archetype = _attention_archetype(
          dims, layer_kind == "sliding_attention", is_shared, mixed_kinds
      )
      params = _attention_block_params(dims, text_config, layer_kind, is_shared)
    counts[archetype] = counts.get(archetype, 0) + 1
    params_by_archetype[archetype] = params
  return counts, params_by_archetype


def _token_stream_shape(embed_dim: int) -> list[Any]:
  """Returns the symbolic `[B, Sq, embed_dim]` activation shape."""
  return ["B", "Sq", embed_dim]


def _block_entry(
    model_name: str,
    archetype: str,
    is_first: bool,
    count: int,
    params: Mapping[str, Any],
) -> dict[str, Any]:
  """Builds the `composite_blocks` entry of one archetype.

  Args:
    model_name: Canonical model identifier.
    archetype: Archetype name of the block.
    is_first: Whether this is the model's first archetype, which takes the bare
      model name as its preset name.
    count: Number of layers of this archetype in the model.
    params: Block parameters of the archetype.

  Returns:
    The catalog entry, including a `roofline_ref_p1024` reference when the
    block uses multi-head attention.
  """
  embed_dim = params["embed_dim"]
  entry: dict[str, Any] = {
      "id": f"block.{model_name}.{archetype}",
      "preset_name": model_name if is_first else f"{model_name}-{archetype}",
      "layer_type": "TransformerBlock",
      "models": [{
          "model": model_name,
          "archetype": f"{model_name}.{archetype}",
          "layers": count,
      }],
      "params": params,
      "inputs": {"x": _token_stream_shape(embed_dim)},
      "outputs": {"y": _token_stream_shape(embed_dim)},
  }
  if params["num_heads"] is not None:
    entry["roofline_ref_p1024"] = estimate_block_roofline_work(
        params, batch_size=1, seq_len_q=1024, seq_len_kv=1024
    )
  return entry


def _attention_layer_id(params: Mapping[str, Any]) -> str:
  """Returns the shape-derived ID that deduplicates attention layers."""
  window = params["sliding_window"]
  window_tag = f"W{window}" if window else "Full"
  suffix = "_SharedKV" if params["shared_kv"] else ""
  if params["k_eq_v"]:
    suffix += "_KeqV"
  if params["qk_norm"]:
    suffix += "_QKNorm"
  if params["gated_q_proj"]:
    suffix += "_GatedQ"
  return (
      f"attn.M{params['embed_dim']}_Hq{params['num_heads']}_"
      f"Hkv{params['num_kv_heads']}_D{params['head_dim']}_"
      f"R{params['rotary_dim']}_{window_tag}{suffix}"
  )


def _attention_entry(
    layer_id: str,
    params: Mapping[str, Any],
    occurrence: Mapping[str, Any],
) -> dict[str, Any]:
  """Builds the `attention_layers` entry of a multi-head attention shape.

  Args:
    layer_id: ID from `_attention_layer_id`.
    params: Block parameters the attention layer belongs to.
    occurrence: Model occurrence record of the owning block.

  Returns:
    A `SelfAttention` catalog entry.
  """
  embed_dim = params["embed_dim"]
  return {
      "id": layer_id,
      "layer_type": "SelfAttention",
      "params": {
          "embed_dim": embed_dim,
          "num_heads": params["num_heads"],
          "num_kv_heads": params["num_kv_heads"],
          "gqa_ratio": params["num_heads"] // params["num_kv_heads"],
          "head_dim": params["head_dim"],
          "rotary_dim": params["rotary_dim"],
          "sliding_window": params["sliding_window"],
          "gated_q_proj": params["gated_q_proj"],
          "shared_kv": params["shared_kv"],
          "k_eq_v": params["k_eq_v"],
          "qk_norm": params["qk_norm"],
      },
      "inputs": {"x": _token_stream_shape(embed_dim)},
      "outputs": {"y": _token_stream_shape(embed_dim)},
      "models": [occurrence],
  }


def _linear_attention_entry(
    model_name: str, embed_dim: int, occurrence: Mapping[str, Any]
) -> dict[str, Any]:
  """Builds the `attention_layers` entry of a linear-attention block."""
  return {
      "id": f"attn.gdn_M{embed_dim}_{model_name}",
      "layer_type": "GatedDeltaNet",
      "params": {"embed_dim": embed_dim},
      "inputs": {"x": _token_stream_shape(embed_dim)},
      "outputs": {"y": _token_stream_shape(embed_dim)},
      "models": [occurrence],
  }


def _archetype_entries(
    model_name: str,
    counts: Mapping[str, int],
    params_by_archetype: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
  """Builds the catalog entries of a model's block archetypes.

  Args:
    model_name: Canonical model identifier.
    counts: Number of layers per archetype, in first-occurrence order.
    params_by_archetype: Block parameters per archetype.

  Returns:
    Tuple `(blocks, attention_layers, composition)`: `composite_blocks`
    entries, attention entries deduplicated by shape, and the model's
    `block_composition`.
  """
  blocks: list[dict[str, Any]] = []
  attention_layers: list[dict[str, Any]] = []
  attention_by_id: dict[str, dict[str, Any]] = {}
  composition: list[dict[str, Any]] = []
  for index, (archetype, count) in enumerate(counts.items()):
    params = params_by_archetype[archetype]
    block = _block_entry(model_name, archetype, index == 0, count, params)
    blocks.append(block)
    composition.append({"layer_id": block["id"], "count": count})
    occurrence = {
        "model": model_name,
        "archetype": f"{model_name}.{archetype}",
        "layers": count,
    }
    if params["num_heads"] is None:
      attention_layers.append(
          _linear_attention_entry(model_name, params["embed_dim"], occurrence)
      )
      continue
    attention_id = _attention_layer_id(params)
    existing = attention_by_id.get(attention_id)
    if existing is not None:
      existing["models"].append(occurrence)
    else:
      entry = _attention_entry(attention_id, params, occurrence)
      attention_by_id[attention_id] = entry
      attention_layers.append(entry)
  return blocks, attention_layers, composition


def _mlp_entry(model_name: str, dims: _TextDims) -> dict[str, Any]:
  """Builds the `mlp_layers` entry shared by all blocks of a model."""
  embed_dim = dims.embed_dim
  hidden_dim = dims.hidden_dim
  return {
      "id": f"mlp.{dims.mlp_activation.lower()}_M{embed_dim}_F{hidden_dim}",
      "layer_type": "GatedMlp",
      "params": {
          "embed_dim": embed_dim,
          "hidden_dim": hidden_dim,
          "activation": dims.mlp_activation,
      },
      "inputs": {"x": _token_stream_shape(embed_dim)},
      "weights": {
          "gate_proj": [hidden_dim, embed_dim],
          "up_proj": [hidden_dim, embed_dim],
          "down_proj": [embed_dim, hidden_dim],
      },
      "outputs": {"y": _token_stream_shape(embed_dim)},
      "models": [{"model": model_name, "layers": dims.num_layers}],
  }


def _first_not_none(*values: Any) -> Any:
  """Returns the first argument that is not `None`, or `None` if all are."""
  for value in values:
    if value is not None:
      return value
  return None


def _model_entry(
    hf_config: Mapping[str, Any],
    text_config: Mapping[str, Any],
    family: str,
    hf_repo: str,
    dims: _TextDims,
    composition: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  """Builds the `models` entry of a scouted model.

  Args:
    hf_config: Parsed Hugging Face `config.json` dictionary.
    text_config: Its text decoder configuration, which takes precedence.
    family: HF `model_type` of the text decoder.
    hf_repo: Hugging Face repository slug, or an empty string.
    dims: Decoder dimensions of the model.
    composition: Block multiplicities of the model.

  Returns:
    The catalog model entry; word embeddings count as tied and the vocabulary
    size falls back to a default when the config sets neither.
  """
  tied = _first_not_none(
      text_config.get("tie_word_embeddings"),
      hf_config.get("tie_word_embeddings"),
      True,
  )
  vocab_size = (
      text_config.get("vocab_size")
      or hf_config.get("vocab_size")
      or _DEFAULT_VOCAB_SIZE
  )
  return {
      "family": family,
      "hf_repo": hf_repo,
      "total_layers": dims.num_layers,
      "embed_dim": dims.embed_dim,
      "vocab_size": int(vocab_size),
      "tie_word_embeddings": bool(tied),
      "eps": dims.eps,
      "block_composition": composition,
  }


def scout_hf_config(
    hf_config: Mapping[str, Any], model_name: str, hf_repo: str = ""
) -> dict[str, Any]:
  """Extracts deduplicated Level 2 catalog entries from an HF `config.json`.

  Handles nested `text_config` wrappers (Gemma 3/4, Qwen 3.5), hybrid local/
  global `layer_types`, `num_kv_shared_layers`, `attention_k_eq_v`, `ple_dim`,
  `qk_norm`, `post_norms`, and linear-attention (`GatedDeltaNet`) blocks
  without loading weight files.

  Args:
    hf_config: Parsed Hugging Face `config.json` dictionary.
    model_name: Canonical model identifier for catalog entries.
    hf_repo: Optional Hugging Face repository slug for provenance metadata.

  Returns:
    A valid `aeb.layer.catalog/1` dictionary containing `models`,
    `composite_blocks`, `attention_layers`, and `mlp_layers`.

  Raises:
    ValueError: If `model_name` is empty, required dimensions are missing, or
      the configuration cannot be expressed as a valid catalog (e.g. an odd
      `head_dim`).
  """
  if not model_name:
    raise ValueError("model_name must be a non-empty string")
  text_config = hf_config.get("text_config")
  if not isinstance(text_config, dict):
    text_config = hf_config
  family = str(
      text_config.get("model_type")
      or hf_config.get("model_type")
      or "transformer"
  )
  dims = _parse_text_dims(text_config, family)
  counts, params_by_archetype = _scout_archetypes(text_config, dims)
  blocks, attention_layers, composition = _archetype_entries(
      model_name, counts, params_by_archetype
  )
  scouted = {
      "schema": CATALOG_SCHEMA,
      "catalog_id": f"scouted-{model_name.lower()}",
      "description": f"Scouted Level 2 layer catalog for {model_name}.",
      "models": {
          model_name: _model_entry(
              hf_config, text_config, family, hf_repo, dims, composition
          ),
      },
      "composite_blocks": blocks,
      "attention_layers": attention_layers,
      "mlp_layers": [_mlp_entry(model_name, dims)],
  }
  validate_catalog(scouted, source=f"scouted:{model_name}")
  return scouted


def _merge_model_occurrences(
    existing: MutableMapping[str, Any], scouted_entry: Mapping[str, Any]
) -> None:
  """Adds the model occurrences of a scouted entry to an existing entry.

  Occurrences are keyed by `(model, archetype)`, so re-merging the same
  scouted model replaces its record instead of duplicating it.

  Args:
    existing: Catalog entry whose `models` list is updated in place.
    scouted_entry: Scouted entry with the same layer ID.
  """
  occurrences = {
      (item["model"], item.get("archetype", "")): item
      for item in existing.get("models", [])
  }
  for item in scouted_entry.get("models", []):
    occurrences[(item["model"], item.get("archetype", ""))] = item
  existing["models"] = list(occurrences.values())


def merge_scouted_into_catalog(
    catalog: Mapping[str, Any], scouted: Mapping[str, Any]
) -> dict[str, Any]:
  """Idempotently merges `scout_hf_config` output into an existing catalog.

  Args:
    catalog: Existing `aeb.layer.catalog/1` dictionary (not mutated).
    scouted: Output dictionary from `scout_hf_config`.

  Returns:
    A new validated `aeb.layer.catalog/1` dictionary containing both inputs.

  Raises:
    ValueError: If either input or the merged catalog violates
      `validate_catalog` invariants.
  """
  validate_catalog(catalog, source="base_catalog")
  validate_catalog(scouted, source="scouted_catalog")
  merged = copy.deepcopy(dict(catalog))
  merged["models"].update(scouted["models"])

  for table in ("composite_blocks", "attention_layers", "mlp_layers"):
    existing_by_id = {entry["id"]: entry for entry in merged.get(table, [])}
    for scouted_entry in scouted.get(table, []):
      existing = existing_by_id.get(scouted_entry["id"])
      if existing is None:
        merged.setdefault(table, []).append(copy.deepcopy(scouted_entry))
      else:
        _merge_model_occurrences(existing, scouted_entry)
  validate_catalog(merged, source="merged_catalog")
  return merged


def _approximated_features(
    params: Mapping[str, Any], head_dim: int
) -> list[str]:
  """Lists block features that `layer_bench` presets cannot express."""
  approximated: list[str] = []
  for feature in ("shared_kv", "k_eq_v", "gated_q_proj"):
    if params.get(feature):
      approximated.append(feature)
  ple_dim = int(params.get("ple_dim") or 0)
  if ple_dim > 0:
    approximated.append(f"ple_dim={ple_dim}")
  rotary_dim = params.get("rotary_dim")
  if _is_pos_int(rotary_dim) and rotary_dim < head_dim:
    approximated.append(f"partial_rotary_dim={rotary_dim}/{head_dim}")
  return approximated


def block_to_preset(block: Mapping[str, Any]) -> dict[str, Any] | None:
  """Converts a `composite_blocks` entry into a C++ `layer_bench` preset.

  Args:
    block: Entry from a catalog's `composite_blocks` table.

  Returns:
    Preset dictionary consumable by `layer_bench --suite_file`, or `None` when
    the block has no standard multi-head attention parameters (e.g., linear
    attention `GatedDeltaNet` blocks where `num_heads` is `None`).
  """
  params = block["params"]
  if params.get("num_heads") is None:
    return None
  raw_name = block.get("preset_name") or block["id"].removeprefix("block.")
  head_dim = int(params["head_dim"])
  window = params.get("sliding_window")
  preset: dict[str, Any] = {
      "name": raw_name.replace("/", "_"),
      "layer_id": block["id"],
      "num_heads": int(params["num_heads"]),
      "head_dim": head_dim,
      "hidden_dim": int(params["hidden_dim"]),
      "num_kv_heads": int(params["num_kv_heads"]),
      "embed_dim": int(params["embed_dim"]),
      "sliding_window": int(window) if _is_pos_int(window) else 0,
      "eps": float(params.get("eps", _DEFAULT_EPS)),
      "mlp_activation": str(params.get("mlp_activation", "SwiGLU")).lower(),
      "qk_norm": bool(params.get("qk_norm", False)),
      "post_norms": _has_post_norms(params),
  }
  approximated = _approximated_features(params, head_dim)
  if approximated:
    preset["approximated_features"] = approximated
  return preset


def preset_name(preset: Mapping[str, Any] | str) -> str:
  """Returns the name of a suite preset.

  Args:
    preset: Either the name of a preset built into `layer_bench`, or an inline
      preset dictionary.

  Returns:
    The preset's name.
  """
  return preset if isinstance(preset, str) else preset["name"]


def _cliff_variants(
    base: Mapping[str, Any], target_dims: Sequence[str], deltas: Sequence[int]
) -> list[dict[str, Any]]:
  """Builds the `(+-Delta)` variants of one preset.

  Args:
    base: Inline preset dictionary to perturb.
    target_dims: Preset keys whose values are perturbed.
    deltas: Signed offsets applied to each target dimension; 0 is skipped.

  Returns:
    One variant per `(dimension, delta)` pair that keeps the dimension
    positive, ordered by dimension then delta. Each variant is named
    `<base>__<dim>_p<delta>` or `<base>__<dim>_m<delta>` and records
    `cliff_parent`, `cliff_dim`, and `cliff_delta`.
  """
  variants: list[dict[str, Any]] = []
  for dim_key in target_dims:
    base_value = base.get(dim_key)
    if base_value is None or base_value <= 0:
      continue
    for delta in deltas:
      value = int(base_value) + delta
      if delta == 0 or value <= 0:
        continue
      sign = f"p{delta}" if delta > 0 else f"m{abs(delta)}"
      variant = copy.deepcopy(dict(base))
      variant[dim_key] = value
      variant["name"] = f"{base['name']}__{dim_key}_{sign}"
      variant["cliff_parent"] = base["name"]
      variant["cliff_dim"] = dim_key
      variant["cliff_delta"] = delta
      variants.append(variant)
  return variants


def _expand_cliff_presets(
    presets: Sequence[dict[str, Any] | str], sweep: Mapping[str, Any]
) -> list[dict[str, Any] | str]:
  """Expands inline presets with `(+-Delta)` variants around target dims.

  Args:
    presets: Suite presets. Preset names (strings) cannot be perturbed and pass
      through unchanged.
    sweep: Cliff-sweep profile providing `deltas` and `target_dims`.

  Returns:
    The presets in input order, each inline preset followed by its variants,
    with repeated names dropped.
  """
  deltas = [int(delta) for delta in sweep.get("deltas", [0])]
  target_dims = list(sweep.get("target_dims", []))
  if deltas == [0] or not target_dims:
    return list(presets)

  expanded: list[dict[str, Any] | str] = []
  seen_names: set[str] = set()
  for base in presets:
    if isinstance(base, str):
      candidates: list[dict[str, Any] | str] = [base]
    else:
      candidates = [base, *_cliff_variants(base, target_dims, deltas)]
    for candidate in candidates:
      name = preset_name(candidate)
      if name not in seen_names:
        expanded.append(candidate)
        seen_names.add(name)
  return expanded


def is_catalog_suite(suite_data: Mapping[str, Any]) -> bool:
  """Returns whether `suite_data` references catalog or hyperparam definitions.

  Args:
    suite_data: Parsed JSON dictionary for a layer benchmark suite.

  Returns:
    `True` if `suite_data` specifies `catalog`, `models`, `layer_ids`, or
    `hyperparams`; `False` otherwise.
  """
  return bool(
      suite_data.get("catalog")
      or suite_data.get("models")
      or suite_data.get("layer_ids")
      or suite_data.get("hyperparams")
  )


def _apply_regime(
    resolved: MutableMapping[str, Any],
    suite_data: Mapping[str, Any],
    registry: _Registry,
    override: str,
) -> None:
  """Binds a regime's operating points and mode-filtered families.

  Args:
    resolved: Suite being resolved; updated in place.
    suite_data: Suite as authored, providing the default `regime`.
    registry: Hyperparameter registry to look the regime up in.
    override: Regime name that replaces the suite's values; empty keeps sweep
      points the suite already defines.

  Raises:
    UnknownProfileError: If the regime is not defined in `registry`.
    ValueError: If no benchmark family matches the regime's mode.
  """
  name = override or suite_data.get("regime", "")
  if not name:
    return
  regime = registry.profile("regimes", "regime", name)
  resolved["regime"] = name
  if override:
    resolved["batch_sizes"] = list(regime["batch_sizes"])
    resolved["seq_lengths"] = list(regime["seq_lengths"])
  else:
    resolved.setdefault("batch_sizes", list(regime["batch_sizes"]))
    resolved.setdefault("seq_lengths", list(regime["seq_lengths"]))
  families = filter_families_for_mode(
      list(resolved.get("families") or DEFAULT_SUITE_FAMILIES),
      str(regime.get("mode", "both")),
  )
  if not families:
    raise ValueError(
        f"Suite {suite_data.get('name')} has zero families matching "
        f"regime {name!r} (mode={regime.get('mode')!r})"
    )
  resolved["families"] = families
  resolved["regime_profile"] = {"name": name, **regime}


def _apply_quant_scheme(
    resolved: MutableMapping[str, Any],
    suite_data: Mapping[str, Any],
    registry: _Registry,
    override: str,
) -> None:
  """Binds a quantization scheme's dtype and quality gate.

  Catalog-backed suites default to the `fp32` scheme.

  Args:
    resolved: Suite being resolved; updated in place.
    suite_data: Suite as authored, providing the default `quant_scheme`.
    registry: Hyperparameter registry to look the scheme up in.
    override: Scheme name that replaces the suite's values; empty keeps a dtype
      and quality gate the suite already defines.

  Raises:
    UnknownProfileError: If the scheme is not defined in `registry`.
  """
  name = override or suite_data.get("quant_scheme", "")
  if not name and is_catalog_suite(suite_data):
    name = "fp32"
  if not name:
    return
  scheme = registry.profile("quant_schemes", "quant_scheme", name)
  resolved["quant_scheme"] = name
  quality_gate = copy.deepcopy(scheme["quality_gate"])
  if override:
    resolved["dtype"] = scheme["weight_dtype"]
    resolved["quality_gate"] = quality_gate
  else:
    resolved.setdefault("dtype", scheme["weight_dtype"])
    resolved.setdefault("quality_gate", quality_gate)
  resolved["quant_profile"] = {"name": name, **scheme}


def _apply_machine_profile(
    resolved: MutableMapping[str, Any],
    suite_data: Mapping[str, Any],
    registry: _Registry,
    override: str,
) -> None:
  """Binds a host's thread counts and Roofline ceilings file.

  Without a machine, default thread counts are used.

  Args:
    resolved: Suite being resolved; updated in place.
    suite_data: Suite as authored, providing the default `machine`.
    registry: Hyperparameter registry to look the machine up in.
    override: Machine name that replaces the suite's values, including its
      ceilings file; empty keeps values the suite already defines.

  Raises:
    UnknownProfileError: If the machine is not defined in `registry`.
  """
  name = override or suite_data.get("machine", "")
  if not name:
    resolved.setdefault("thread_counts", list(_DEFAULT_THREAD_COUNTS))
    return
  machine = registry.profile("machine_threading", "machine_threading", name)
  resolved["machine"] = name
  ceilings_file = ""
  if machine.get("ceilings_file"):
    ceilings_file = str(
        (registry.path.parent / machine["ceilings_file"]).resolve()
    )
  if override:
    resolved["thread_counts"] = list(machine["thread_counts"])
    resolved["ceilings_file"] = ceilings_file
  else:
    resolved.setdefault("thread_counts", list(machine["thread_counts"]))
    if not resolved.get("ceilings_file") and ceilings_file:
      resolved["ceilings_file"] = ceilings_file
  resolved["machine_profile"] = {"name": name, **machine}


def _apply_cliff_sweep(
    resolved: MutableMapping[str, Any],
    suite_data: Mapping[str, Any],
    registry: _Registry,
    override: str,
) -> None:
  """Expands inline presets with a cliff sweep's perturbations.

  Catalog-backed suites default to the `none` sweep.

  Args:
    resolved: Suite being resolved; updated in place.
    suite_data: Suite as authored, providing the default `cliff_sweep`.
    registry: Hyperparameter registry to look the sweep up in.
    override: Sweep name that replaces the suite's; empty uses the suite's.

  Raises:
    UnknownProfileError: If the sweep is not defined in `registry`.
  """
  name = override or suite_data.get("cliff_sweep", "")
  if not name and is_catalog_suite(suite_data):
    name = "none"
  if not name:
    return
  sweep = registry.profile("cliff_sweeps", "cliff_sweep", name)
  presets = resolved.get("presets")
  if isinstance(presets, list) and any(
      isinstance(preset, dict) for preset in presets
  ):
    resolved["presets"] = _expand_cliff_presets(presets, sweep)
  resolved["cliff_profile"] = {"name": name, **sweep}


def _apply_hyperparam_profiles(
    resolved: MutableMapping[str, Any],
    suite_data: Mapping[str, Any],
    registry: _Registry,
    machine_override: str = "",
    regime_override: str = "",
    quant_override: str = "",
    cliff_override: str = "",
) -> None:
  """Applies regime, quantization, machine, and cliff profiles onto resolved.

  Args:
    resolved: Suite being resolved; updated in place.
    suite_data: Suite as authored, providing default profile names.
    registry: Hyperparameter registry defining the profiles.
    machine_override: Optional machine profile replacing the suite's.
    regime_override: Optional regime profile replacing the suite's.
    quant_override: Optional quantization scheme replacing the suite's.
    cliff_override: Optional cliff sweep replacing the suite's.

  Raises:
    UnknownProfileError: If a named profile is not defined in `registry`.
    ValueError: If no benchmark family matches the regime's mode.
  """
  _apply_regime(resolved, suite_data, registry, regime_override)
  _apply_quant_scheme(resolved, suite_data, registry, quant_override)
  _apply_machine_profile(resolved, suite_data, registry, machine_override)
  _apply_cliff_sweep(resolved, suite_data, registry, cliff_override)


def _load_registry(suite_data: Mapping[str, Any], suite_dir: Path) -> _Registry:
  """Loads and validates the hyperparameter registry a suite refers to.

  Args:
    suite_data: Suite whose `hyperparams` path (relative to `suite_dir`) names
      the registry; an unset, null, or empty path selects
      `DEFAULT_HYPERPARAMS_FILE`.
    suite_dir: Directory containing the suite file.

  Returns:
    The validated registry.

  Raises:
    OSError: If the registry file cannot be read.
    ValueError: If the registry is not valid `aeb.layer.hyperparams/1` JSON.
  """
  relative = suite_data.get("hyperparams") or DEFAULT_HYPERPARAMS_FILE
  path = (suite_dir / relative).resolve()
  hyperparams = load_json_object(path)
  validate_hyperparams(hyperparams, source=str(path), base_dir=path.parent)
  return _Registry(tables=hyperparams, path=path)


def _block_weight_entry(block: Mapping[str, Any], count: int) -> dict[str, Any]:
  """Builds the `model_layer_weights` record of one block multiplicity.

  Args:
    block: `composite_blocks` entry the model repeats.
    count: Number of times the model repeats `block`.

  Returns:
    Record naming the block's preset, or flagged `unsupported` with a null
    preset when the block has no `layer_bench` preset.
  """
  preset = block_to_preset(block)
  if preset is None:
    return {
        "layer_id": block["id"],
        "preset": None,
        "count": count,
        "unsupported": True,
    }
  entry: dict[str, Any] = {
      "layer_id": block["id"],
      "preset": preset["name"],
      "count": count,
  }
  approximated = preset.get("approximated_features")
  if approximated:
    entry["approximated_features"] = list(approximated)
  return entry


def _collect_model_layer_weights(
    suite_data: Mapping[str, Any],
    catalog: Mapping[str, Any],
    blocks_by_id: Mapping[str, Mapping[str, Any]],
    catalog_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
  """Expands a suite's models into block multiplicities and block IDs.

  Args:
    suite_data: Suite naming `models` and optional explicit `layer_ids`.
    catalog: Validated catalog defining the models.
    blocks_by_id: The catalog's `composite_blocks` keyed by layer ID.
    catalog_path: Path of the catalog file, for error messages.

  Returns:
    Tuple `(model_layer_weights, requested_ids)`: per-model block
    multiplicity records, and the suite's `layer_ids` followed by every block
    the models use, without repeats.

  Raises:
    ValueError: If the suite names a model the catalog does not define.
  """
  requested_ids: list[str] = list(suite_data.get("layer_ids", []))
  model_layer_weights: dict[str, list[dict[str, Any]]] = {}
  for model_name in suite_data.get("models", []):
    model_info = catalog["models"].get(model_name)
    if model_info is None:
      raise ValueError(
          f"Suite {suite_data.get('name')} references unknown model "
          f"{model_name!r} in {catalog_path}"
      )
    weights: list[dict[str, Any]] = []
    for item in model_info["block_composition"]:
      block_id = item["layer_id"]
      if block_id not in requested_ids:
        requested_ids.append(block_id)
      weights.append(
          _block_weight_entry(blocks_by_id[block_id], int(item["count"]))
      )
    model_layer_weights[model_name] = weights
  return model_layer_weights, requested_ids


def _preset_shape_signature(preset: Mapping[str, Any]) -> tuple[Any, ...]:
  """Returns the fields that make two presets benchmark-identical."""
  return (
      preset["num_heads"],
      preset["head_dim"],
      preset["hidden_dim"],
      preset["num_kv_heads"],
      preset["embed_dim"],
      preset["mlp_activation"],
      preset["sliding_window"],
      preset["qk_norm"],
      preset["post_norms"],
      round(float(preset["eps"]), 12),
  )


def _resolve_catalog_presets(
    suite_data: Mapping[str, Any],
    requested_ids: Sequence[str],
    blocks_by_id: Mapping[str, Mapping[str, Any]],
) -> _CatalogPresets:
  """Converts the requested catalog blocks into `layer_bench` presets.

  Args:
    suite_data: Suite providing explicit `presets` and the
      `deduplicate_identical_presets` switch.
    requested_ids: Block IDs to benchmark, in order.
    blocks_by_id: The catalog's `composite_blocks` keyed by layer ID.

  Returns:
    The suite's own presets followed by one preset per runnable block. With
    deduplication on, blocks whose shape matches an earlier block are dropped
    and recorded in `remap`.

  Raises:
    ValueError: If a requested block ID is not in the catalog.
  """
  result = _CatalogPresets(presets=list(suite_data.get("presets", [])))
  seen_names = {preset_name(preset) for preset in result.presets}
  deduplicate = bool(suite_data.get("deduplicate_identical_presets", False))
  canonical_by_shape: dict[tuple[Any, ...], str] = {}
  for layer_id in requested_ids:
    block = blocks_by_id.get(layer_id)
    if block is None:
      raise ValueError(f"Unknown composite_block layer_id {layer_id!r}")
    preset = block_to_preset(block)
    if preset is None:
      result.skipped_layers.append(layer_id)
      continue
    name = preset["name"]
    signature = _preset_shape_signature(preset)
    canonical = canonical_by_shape.get(signature)
    if deduplicate and canonical is not None:
      result.remap[name] = canonical
      continue
    canonical_by_shape[signature] = name
    approximated = preset.get("approximated_features")
    if approximated:
      result.approximated[name] = list(approximated)
    if name not in seen_names:
      result.presets.append(preset)
      seen_names.add(name)
  return result


def _remap_deduplicated_weights(
    model_layer_weights: Mapping[str, list[dict[str, Any]]],
    remap: Mapping[str, str],
) -> None:
  """Points block multiplicities at the canonical deduplicated presets.

  Args:
    model_layer_weights: Per-model multiplicity records; updated in place with
      the canonical `preset` and the original name in `deduplicated_from`.
    remap: Deduplicated preset name to its canonical replacement.
  """
  for weights in model_layer_weights.values():
    for weight in weights:
      canonical = remap.get(weight.get("preset") or "")
      if canonical is not None:
        weight["deduplicated_from"] = weight["preset"]
        weight["preset"] = canonical


def _check_suite_fields(suite_data: Mapping[str, Any]) -> None:
  """Checks the types of the suite fields that resolution relies on.

  Args:
    suite_data: Parsed suite dictionary.

  Raises:
    ValueError: If the suite is not an object, `presets` holds anything but
      names and named objects or names a preset twice, `families`, `models`,
      or `layer_ids` is not a list of strings, a file path or profile name is
      neither a string nor null, or a present sweep axis is not a non-empty
      list of positive integers.
  """
  if not isinstance(suite_data, Mapping):
    raise ValueError("Suite must be a JSON object")
  suite_name = suite_data.get("name")
  presets = suite_data.get("presets", [])
  if not isinstance(presets, list):
    raise ValueError(f"Suite {suite_name} 'presets' must be a list")
  seen_presets: set[str] = set()
  for preset in presets:
    named_object = isinstance(preset, dict) and isinstance(
        preset.get("name"), str
    )
    if not isinstance(preset, str) and not named_object:
      raise ValueError(
          f"Suite {suite_name} preset must be a name or an object with a "
          f"'name': {preset!r}"
      )
    name = preset_name(preset)
    # `layer_bench` keys its benchmarks by preset name and rejects repeats.
    if name in seen_presets:
      raise ValueError(
          f"Suite {suite_name} names preset {name!r} more than once"
      )
    seen_presets.add(name)
  for key in _SUITE_NAME_LIST_KEYS:
    names = suite_data.get(key, [])
    if not isinstance(names, list) or not all(
        isinstance(name, str) for name in names
    ):
      raise ValueError(f"Suite {suite_name} {key!r} must be a list of names")
  for key in _SUITE_STRING_KEYS:
    value = suite_data.get(key)
    if value is not None and not isinstance(value, str):
      raise ValueError(
          f"Suite {suite_name} {key!r} must be a string: {value!r}"
      )
  for key in ("batch_sizes", "seq_lengths", "thread_counts"):
    values = suite_data.get(key, _UNSET)
    if values is not _UNSET:
      _check_pos_int_list(values, f"Suite {suite_name}:{key}")


def _resolve_catalog_suite(
    resolved: MutableMapping[str, Any],
    suite_data: Mapping[str, Any],
    suite_dir: Path,
    overrides: Mapping[str, str],
) -> None:
  """Resolves a catalog-backed suite into presets and bound profiles.

  Args:
    resolved: Suite being resolved; receives `presets`, the hyperparameter
      profiles, `catalog_id`, `model_layer_weights`, and (when non-empty)
      `skipped_layers` and `approximated_presets`.
    suite_data: Catalog-backed suite as authored; an unset, null, or empty
      `catalog` path selects `DEFAULT_CATALOG_FILE`.
    suite_dir: Directory containing the suite file.
    overrides: Profile override keyword arguments for
      `_apply_hyperparam_profiles`.

  Raises:
    OSError: If the catalog or hyperparameter file cannot be read.
    UnknownProfileError: If a named profile is not defined in the registry.
    ValueError: If a referenced file is invalid, the suite references unknown
      models or layers, or it resolves to zero presets or families.
  """
  catalog_path = (
      suite_dir / (suite_data.get("catalog") or DEFAULT_CATALOG_FILE)
  ).resolve()
  catalog = load_json_object(catalog_path)
  validate_catalog(catalog, source=str(catalog_path))
  registry = _load_registry(suite_data, suite_dir)
  blocks_by_id = {
      block["id"]: block for block in catalog.get("composite_blocks", [])
  }
  model_layer_weights, requested_ids = _collect_model_layer_weights(
      suite_data, catalog, blocks_by_id, catalog_path
  )
  selection = _resolve_catalog_presets(suite_data, requested_ids, blocks_by_id)
  _remap_deduplicated_weights(model_layer_weights, selection.remap)
  if not selection.presets:
    raise ValueError(
        f"Suite {suite_data.get('name')} resolved to zero runnable presets"
    )
  resolved["presets"] = selection.presets
  _apply_hyperparam_profiles(resolved, suite_data, registry, **overrides)
  resolved["catalog_id"] = catalog["catalog_id"]
  resolved["model_layer_weights"] = model_layer_weights
  if selection.skipped_layers:
    resolved["skipped_layers"] = selection.skipped_layers
  if selection.approximated:
    resolved["approximated_presets"] = selection.approximated


def resolve_suite(
    suite_data: Mapping[str, Any],
    suite_dir: Path,
    machine_override: str = "",
    regime_override: str = "",
    quant_override: str = "",
    cliff_override: str = "",
) -> dict[str, Any]:
  """Resolves a catalog-backed or inline suite into a `layer_bench` JSON spec.

  A suite that references the catalog is expanded from it. Any other suite
  is returned as authored unless it or an override names a hyperparameter
  profile (`regime`, `quant_scheme`, `machine`, or `cliff_sweep`), which is
  then bound from the registry.

  Args:
    suite_data: Parsed JSON from `layer/suites/<name>.json`.
    suite_dir: Directory containing the suite file for relative path resolution.
    machine_override: Optional host profile key in `hyperparams.json`.
    regime_override: Optional regime profile key in `hyperparams.json`.
    quant_override: Optional quantization profile key in `hyperparams.json`.
    cliff_override: Optional cliff-sweep profile key in `hyperparams.json`.

  Returns:
    A resolved dictionary containing concrete `presets`, `families`,
    `batch_sizes`, `seq_lengths`, `thread_counts`, `quality_gate`,
    `ceilings_file`, and `model_layer_weights` consumable by both C++
    `layer_bench --suite_file=...` and Python `aeb.layer`.

  Raises:
    OSError: If a referenced catalog or hyperparameter file cannot be read.
    UnknownProfileError: If the suite or an override names a profile the
      hyperparameter registry does not define.
    ValueError: If the suite or a referenced file is invalid, the suite
      references unknown models or layers, or it resolves to zero presets or
      families.
  """
  _check_suite_fields(suite_data)
  resolved = copy.deepcopy(dict(suite_data))
  overrides = {
      "machine_override": machine_override,
      "regime_override": regime_override,
      "quant_override": quant_override,
      "cliff_override": cliff_override,
  }
  if is_catalog_suite(suite_data):
    _resolve_catalog_suite(resolved, suite_data, suite_dir, overrides)
  elif any(overrides.values()) or any(
      suite_data.get(key) for key in _SUITE_PROFILE_KEYS
  ):
    # An inline suite that names a profile gets it from the default registry
    # instead of having the name silently ignored.
    registry = _load_registry(suite_data, suite_dir)
    _apply_hyperparam_profiles(resolved, suite_data, registry, **overrides)
  return resolved


def _validate_catalog_dir(catalog_dir: Path) -> tuple[int, int]:
  """Validates every registry JSON file in a `layer/catalog` directory.

  Args:
    catalog_dir: Directory holding catalog and hyperparameter JSON files;
      `schema.json` is skipped.

  Returns:
    Tuple `(n_catalogs, n_hyperparams)` of validated files.

  Raises:
    OSError: If a file cannot be read.
    ValueError: If a file is invalid or declares an unrecognized schema.
  """
  n_catalogs = 0
  n_hyperparams = 0
  for path in sorted(catalog_dir.glob("*.json")):
    if path.name == "schema.json":
      continue
    document = load_json_object(path)
    schema = document.get("schema", "")
    if schema == CATALOG_SCHEMA:
      validate_catalog(document, source=str(path))
      n_catalogs += 1
    elif schema == HYPERPARAMS_SCHEMA:
      validate_hyperparams(document, source=str(path), base_dir=path.parent)
      n_hyperparams += 1
    else:
      raise ValueError(f"Unrecognized schema {schema!r} in {path}")
  return n_catalogs, n_hyperparams


def _validate_suite_file(suite_path: Path) -> None:
  """Checks that a suite file resolves to a runnable `layer_bench` spec.

  Args:
    suite_path: Suite JSON file to resolve.

  Raises:
    OSError: If the suite or a file it references cannot be read.
    ValueError: If the suite does not resolve to presets, families, positive
      operating points, and an existing ceilings file (when one is named).
  """
  resolved = resolve_suite(load_json_object(suite_path), suite_path.parent)
  if not resolved.get("presets") or not resolved.get("families"):
    raise ValueError(
        f"Suite {suite_path} missing presets or families after resolve"
    )
  for key in ("batch_sizes", "seq_lengths", "thread_counts"):
    _check_pos_int_list(resolved.get(key), f"{suite_path}:{key}")
  ceilings_file = resolved.get("ceilings_file", "")
  if ceilings_file:
    ceilings_path = Path(ceilings_file)
    if not ceilings_path.is_absolute():
      ceilings_path = (suite_path.parent / ceilings_file).resolve()
    if not ceilings_path.is_file():
      raise ValueError(
          f"Suite {suite_path} ceilings_file does not exist: {ceilings_path}"
      )


def validate_repo_layer_configs(layer_root: Path) -> dict[str, int]:
  """Validates all catalog, hyperparams, and suite JSONs under `layer_root`.

  Args:
    layer_root: Path to the repository's `layer/` directory.

  Returns:
    Dictionary with integer counts for keys `catalogs`, `hyperparams`, and
    `suites`.

  Raises:
    OSError: If a catalog, hyperparameter registry, or suite cannot be read.
    ValueError: If `layer_root` holds no catalog, or any catalog,
      hyperparameter registry, or suite fails schema or resolution checks.
  """
  catalog_dir = layer_root / "catalog"
  n_catalogs, n_hyperparams = _validate_catalog_dir(catalog_dir)
  if n_catalogs == 0:
    raise ValueError(f"No layer catalog JSON found in {catalog_dir}")
  suite_paths = sorted((layer_root / "suites").glob("*.json"))
  for suite_path in suite_paths:
    _validate_suite_file(suite_path)
  return {
      "catalogs": n_catalogs,
      "hyperparams": n_hyperparams,
      "suites": len(suite_paths),
  }


def _default_layer_root() -> Path:
  """Returns the repository `layer/` directory, or `/layer` in containers."""
  root = Path(__file__).resolve().parents[4] / "layer"
  return root if root.is_dir() else Path("/layer")


def _add_scout_flags(parser: argparse.ArgumentParser) -> None:
  """Registers the flags of the `scout` subcommand."""
  parser.add_argument("--hf-config", required=True, help="Path to config.json")
  parser.add_argument("--model-name", required=True, help="Canonical model ID")
  parser.add_argument("--hf-repo", default="", help="Optional HF repo slug")
  parser.add_argument(
      "--merge-into",
      default="",
      help="Optional existing catalog JSON path to update in-place.",
  )
  parser.add_argument("--out", default="", help="Optional output JSON path.")


def _add_resolve_flags(parser: argparse.ArgumentParser) -> None:
  """Registers the flags of the `resolve` subcommand."""
  parser.add_argument("--suite", required=True, help="Path to suite JSON")
  parser.add_argument("--machine", default="", help="Override machine profile")
  parser.add_argument("--regime", default="", help="Override regime profile")
  parser.add_argument("--quant", default="", help="Override quant_scheme")
  parser.add_argument("--cliff-sweep", default="", help="Override cliff_sweep")
  parser.add_argument("--out", default="", help="Write resolved JSON to path")


def _build_arg_parser() -> argparse.ArgumentParser:
  """Builds the `validate | summary | scout | resolve` command-line parser."""
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  subparsers = parser.add_subparsers(dest="cmd", required=True)
  validate_parser = subparsers.add_parser(
      "validate", help="Validate all layer/catalog and layer/suites JSON files."
  )
  validate_parser.add_argument(
      "--layer-root",
      default="",
      help="Path to layer/ directory (auto-detected if omitted).",
  )
  summary_parser = subparsers.add_parser(
      "summary", help="Print model and canonical layer counts in a catalog."
  )
  summary_parser.add_argument(
      "--catalog",
      default="",
      help="Path to catalog JSON (defaults to gemma_qwen_edge.json).",
  )
  scout_parser = subparsers.add_parser(
      "scout", help="Extract canonical Level 2 layers from an HF config.json."
  )
  _add_scout_flags(scout_parser)
  resolve_parser = subparsers.add_parser(
      "resolve",
      help="Resolve a suite + catalog + hyperparams into a layer_bench JSON.",
  )
  _add_resolve_flags(resolve_parser)
  return parser


def _cmd_validate(args: argparse.Namespace) -> int:
  """Validates every catalog, registry, and suite; prints the counts."""
  layer_root = (
      Path(args.layer_root) if args.layer_root else _default_layer_root()
  )
  counts = validate_repo_layer_configs(layer_root)
  print(
      f"[aeb.catalog] VALIDATED {counts['catalogs']} catalog(s), "
      f"{counts['hyperparams']} hyperparam registry, "
      f"{counts['suites']} suite(s) under {layer_root}"
  )
  return 0


def _cmd_summary(args: argparse.Namespace) -> int:
  """Prints per-table layer counts and per-model block compositions."""
  if args.catalog:
    catalog_path = Path(args.catalog)
  else:
    catalog_path = _default_layer_root() / "catalog" / "gemma_qwen_edge.json"
  catalog = load_json_object(catalog_path)
  validate_catalog(catalog, source=str(catalog_path))
  models = catalog["models"]
  print(f"Catalog: {catalog['catalog_id']} ({len(models)} models)")
  for table in LAYER_TABLES:
    entries = catalog.get(table)
    if entries is not None:
      print(f"  {table:20s}: {len(entries):3d}")
  for model_name, model_info in models.items():
    parts = ", ".join(
        f"{item['count']}x {item['layer_id'].removeprefix('block.')}"
        for item in model_info["block_composition"]
    )
    total_layers = model_info["total_layers"]
    print(f"  model {model_name:14s} ({total_layers:2d}L): {parts}")
  return 0


def _cmd_scout(args: argparse.Namespace) -> int:
  """Scouts an HF config; merges it into a catalog, writes, or prints it."""
  hf_config = load_json_object(Path(args.hf_config))
  scouted = scout_hf_config(hf_config, args.model_name, hf_repo=args.hf_repo)
  if args.merge_into:
    base_path = Path(args.merge_into)
    merged = merge_scouted_into_catalog(load_json_object(base_path), scouted)
    target = Path(args.out) if args.out else base_path
    target.write_text(_dump_json(merged), encoding="utf-8")
    print(f"[aeb.catalog] Merged {args.model_name} into {target}")
  elif args.out:
    Path(args.out).write_text(_dump_json(scouted), encoding="utf-8")
  else:
    sys.stdout.write(_dump_json(scouted))
  return 0


def _cmd_resolve(args: argparse.Namespace) -> int:
  """Resolves a suite with optional overrides; writes or prints the JSON."""
  suite_path = Path(args.suite)
  resolved = resolve_suite(
      load_json_object(suite_path),
      suite_path.parent,
      machine_override=args.machine,
      regime_override=args.regime,
      quant_override=args.quant,
      cliff_override=args.cliff_sweep,
  )
  text = _dump_json(resolved)
  if args.out:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
  else:
    sys.stdout.write(text)
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint for validating, summarizing, scouting, and resolving.

  Args:
    argv: Command-line arguments without the program name; `None` reads
      `sys.argv[1:]`.

  Returns:
    Process exit code: 0 on success, 1 when an input file is unreadable or
    invalid (the reason is printed to stderr).

  Raises:
    SystemExit: If `argv` asks for `--help` or is not a valid command line;
      argparse prints the usage and exits with status 0 or 2.
  """
  args = _build_arg_parser().parse_args(argv)
  commands = {
      "validate": _cmd_validate,
      "summary": _cmd_summary,
      "scout": _cmd_scout,
      "resolve": _cmd_resolve,
  }
  try:
    return commands[args.cmd](args)
  except (OSError, ValueError) as error:
    print(f"[aeb.catalog] ERROR: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())
