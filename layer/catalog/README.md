# Level 2 Layer Catalog & Hyperparameter Registry (`layer/catalog/`)

`layer/catalog/` separates **immutable architectural layer shapes and model multiplicities** from **execution operating points, quantization schemes, cliff sweeps, and host CPU threading profiles**. Benchmark suites in `layer/suites/*.json` bind a subset of catalog models or layer IDs to named hyperparameter profiles without duplicating shape tuples across machines or studies.

## Two-File Registry Design

| File | Schema | Responsibility |
| :--- | :--- | :--- |
| [schema.json](schema.json) | Draft 2020-12 | Formal JSON Schema definitions for `aeb.layer.catalog/1` and `aeb.layer.hyperparams/1`. |
| [gemma_qwen_edge.json](gemma_qwen_edge.json) | `aeb.layer.catalog/1` | Canonical layer & subgraph definitions (`composite_blocks`, `attention_layers`, `mlp_layers`, `ple_layers`, `norm_layers`, `rope_layers`, `linear_fc_layers`) across Gemma 3 (`270M`, `1B`, `4B`), Gemma 4 (`E2B`, `E4B`, `31B`), Qwen 3 (`0.6B`, `1.7B`, `4B`), and Qwen 3.5 (`0.8B`), plus per-model `block_composition` layer counts (`w_L`). |
| [hyperparams.json](hyperparams.json) | `aeb.layer.hyperparams/1` | Reusable `regimes` (`p1_d1`, `p128_d128`, `p1024_d128`, `decode_ctx8192`, ...), `quant_schemes` (`fp32`, `bf16`, `fp16`, `qc8w_qd8`, `qb4w_b32`, `qb4w_b128`, `qc2w_b32`, `fp32_kv_q8_0`) with numerical fidelity gates (`rtol`, `atol`, `min_cosine`, `max_l2_rel_err`, `max_kl_divergence`), `cliff_sweeps` (`tile_boundaries`, `cacheline_strides`), and `machine_threading` profiles. |

## Why Per-Model Layer Multiplicities (`block_composition`) Live in the Catalog

Modern edge and workstation models are heterogeneous within a single checkpoint:

- **Gemma 4 E2B (`35` layers):** `12x` Local Producer (`D=256, W=512`), `16x` Local SharedKV (`D=256, W=512`), `3x` Global Producer (`D=512, R=128`), and `4x` Global SharedKV (`D=512, R=128`), plus `PLE` (`P=256`).
- **Gemma 4 31B (`60` layers):** `50x` Local Sliding-Window (`H_q=32, H_kv=16, D=256, W=1024`) + `10x` Global Full-Attention (`H_q=32, H_kv=4, D=512, R=128, K=V`).
- **Qwen 3.5 0.8B (`24` layers):** `18x` `GatedDeltaNet` linear-attention blocks + `6x` `GatedFullAttention` blocks (`3:1` hybrid).

Storing `block_composition` (`[{layer_id, count}, ...]`) directly in `gemma_qwen_edge.json` allows `aeb.layer` to compute whole-model summed block latency:

```python
T_blocks_est(M)   = sum(count(L, M) * T_layer(L) for L in blocks(M))
DeltaT_runtime(M) = T_e2e(M) - T_blocks_est(M)
```

and emit `estimated_model_blocks_ms` in every `summary.json` run artifact.

## CLI Workflow (`layer/scripts/layer_catalog.py`)

Validate all catalog, hyperparameter, and suite JSON invariants (run automatically by `make -C layer test`):

```bash
python3 layer/scripts/layer_catalog.py validate
python3 layer/scripts/layer_catalog.py summary
```

Scout a new Hugging Face `config.json` offline (without downloading multi-GB `.safetensors` weights) and merge deduplicated archetypes into the catalog:

```bash
python3 layer/scripts/layer_catalog.py scout \
  --hf-config /path/to/config.json \
  --model-name Gemma4-NewVariant \
  --hf-repo google/gemma-4-new \
  --merge-into layer/catalog/gemma_qwen_edge.json
```

Resolve a declarative suite (with optional machine, regime, quantization, or `(+-Delta)` register-tile cliff overrides) into a concrete C++ `layer_bench --suite_file=...` payload:

```bash
python3 layer/scripts/layer_catalog.py resolve \
  --suite layer/suites/gemma4-e2b-cpu.json \
  --machine sapphire-rapids \
  --quant qb4w_b32 \
  --cliff-sweep tile_boundaries \
  --out /tmp/resolved_suite.json
```
