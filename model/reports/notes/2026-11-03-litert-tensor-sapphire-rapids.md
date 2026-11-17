# LiteRT C++ Tensor API (`XnnpackRunner` & Native `YnnpackRunner`) Sapphire Rapids Evaluation (`ddct8` & `qtbhw10`)

## 1. Evaluation Environment & Architecture (`ddct8.prod.google.com` & `qtbhw10.prod.google.com`)

All benchmark measurements and parity verifications reported below were executed end-to-end inside Dockerized `ubuntu:24.04` container bundles via [`setup_amx_machine.sh`](file://../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/scratch/ai-edge-bench/scripts/setup_amx_machine.sh), [`dock_shim.py`](file://../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/scratch/ai-edge-bench/scripts/dock_shim.py) (`unshare --mount chroot` + `numactl --physcpubind=0-55 --localalloc`), and [`run_amx_benchmarks.sh`](file://../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/scratch/ai-edge-bench/scripts/run_amx_benchmarks.sh) on dedicated Google `devrez` Sapphire Rapids machines **`root@ddct8.prod.google.com`** (`litert-tensor/fp32-xnnpack`) and **`root@qtbhw10.prod.google.com`** (`litert-tensor/fp32-ynnpack`):

| Parameter | Verified Value |
| --- | --- |
| **Host FQDNs** | `ddct8.prod.google.com` (`XnnpackRunner`) & `qtbhw10.prod.google.com` (native `YnnpackRunner`) |
| **CPU Model** | Dual-socket `Intel(R) Xeon(R) Platinum 8481C` (Sapphire Rapids, `56` physical cores/socket, `224` logical CPUs) |
| **NUMA & Pinning Policy** | Pinned to Socket 0 physical cores (`numactl --physcpubind=0-55 --localalloc`, `56` physical cores) |
| **ISA Extensions** | `avx512f`, `avx512_vnni`, `avx512_bf16`, `avx512_fp16`, `amx_tile`, `amx_bf16`, `amx_int8` |
| **Native Backend Dispatch** | `XnnpackMixinTag` + `XnnpackRunner` vs. `YnnpackMixinTag` + `YnnpackRunner` (`aeb_driver.h`, `litert_tensor_backend.cc`) |
| **Model Checkpoint** | `google/gemma-4-E2B-it` (`model.safetensors` SHA-256: `2db5482b20d7...`, full 35-layer decoder) |
| **Harness Unit Tests** | **79 / 79 passed** (`make MACHINE=sapphire-rapids test`, `OK (skipped=1)`) |
| **Generated Combined Report** | [`2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/README.md`](file://../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/scratch/ai-edge-bench/reports/2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/README.md) & [`dashboards/index.html`](file://../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/scratch/ai-edge-bench/dashboards/index.html) |

---

## 2. Full 35-Layer Gemma 4 E2B End-to-End Performance (`Socket 0`)

### 2.1 Headline Stateless Cold-KV Performance (`1024` Prompt → `128` Decode Tokens, `2 Rounds × 3 Reps = 6 Measured Runs`)

| Metric | `litert-tensor/fp32-xnnpack` (`XnnpackRunner`, `t = 32`) | `litert-tensor/fp32-ynnpack` (`YnnpackRunner`, `t = 32`) | Relative Comparison (`XnnpackRunner` vs. `YnnpackRunner`) |
| --- | ---: | ---: | --- |
| **TTFT (ms) ↓** | **1,432 ms** `[1,415–1,455]` | **1,547 ms** `[1,538–1,569]` | XNNPACK **1.08× faster** (`+7.3%` faster than prior XNNPACK-shim YNNPACK) |
| **Prefill Throughput (tok/s) ↑** | **715.3 tok/s** `[703.6–723.9]` | **661.9 tok/s** `[652.5–665.8]` | XNNPACK **1.08× higher** |
| **Decode Throughput (tok/s) ↑** | **19.65 tok/s** `[19.24–19.75]` | **10.95 tok/s** `[10.82–10.99]` | XNNPACK **1.79× higher** (`+12.3%` faster than prior XNNPACK-shim YNNPACK) |
| **End-to-End Latency (ms) ↓** | **7,896 ms** `[7,853–8,030]` | **13,162 ms** `[13,092–13,273]` | XNNPACK **1.67× faster** |
| **Inter-token p99 Latency (ms) ↓** | **57.3 ms** `[51.6–66.8]` | **93.4 ms** `[92.0–104.5]` | Native `YnnpackRunner` cut p99 by **24.2%** vs. shim (`123.3 ms`) |
| **Active CPU Cores (Prefill)** | **30.03** `[29.76–30.08]` | **28.04** `[27.85–28.29]` | `93.8%` vs. `87.6%` thread efficiency |
| **Active CPU Cores (Decode)** | **27.84** `[27.47–27.92]` | **11.51** `[11.42–11.54]` | `87.0%` vs. `36.0%` thread efficiency |
| **Peak RSS (Request, MB) ↓** | **37,744 MB** `[37,741–37,753]` | **37,275 MB** `[37,230–37,289]` | **YNNPACK uses 469 MB less RAM** |
| **Model Load, Warm Cache (ms) ↓** | **9,439 ms** | **6,545 ms** | **YNNPACK loads & compiles 1.44× faster** (`5.0×` faster than shim's `32,584 ms`) |
| **Measured Requests (`n`)** | **6** (`status=ok`, `CV=1.1%`) | **6** (`status=ok`, `CV=0.6%`) | 100% pass rate & bit-identical greedy text |

````carousel
![Headline Throughput (XnnpackRunner vs. YnnpackRunner)](../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/throughput.png)
<!-- slide -->
![Headline Latency (XnnpackRunner vs. YnnpackRunner)](../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/latency.png)
<!-- slide -->
![Prefix Caching Warm vs. Cold KV (XnnpackRunner vs. YnnpackRunner)](../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/prefix_cache.png)
<!-- slide -->
![Thread Scaling Sweep on Socket 0 (XnnpackRunner vs. YnnpackRunner)](../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/thread_scaling.png)
````

---

### 2.2 Warm-KV Cached-Prefix Performance (`896` Cached Prefix + `128` Suffix → `128` Decode Tokens)

| Config | Cold TTFT (ms) ↓ | Warm TTFT (ms) ↓ | TTFT Speedup ↑ | Cold Prefill (tok/s) ↑ | Warm Effective Prefill (tok/s) ↑ | Warm Suffix Prefill (tok/s) ↑ | Cold E2E (ms) ↓ | Warm E2E (ms) ↓ | E2E Speedup ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`litert-tensor/fp32-xnnpack`** | 1,432 `[1,415–1,455]` | **273** `[270–286]` | **5.24×** `[4.97–5.32×]` | 715.3 | **3,750.9** `[3,577.9–3,788.0]` | **468.9** | 7,896 | **6,734** | **1.17×** |
| **`litert-tensor/fp32-ynnpack`** | 1,547 `[1,538–1,569]` | **313** `[311–318]` | **4.95×** `[4.84–5.02×]` | 661.9 | **3,274.3** `[3,216.0–3,291.0]` | **409.3** | 13,162 | **11,927** | **1.10×** |

---

### 2.3 Socket 0 Thread-Scaling Sweep (`1024` Prompt → `64` Decode Tokens, `t ∈ {8, 16, 32, 40}`)

| Threads (`t`) | `fp32-xnnpack` TTFT (ms) | `fp32-xnnpack` Decode (tok/s) | `fp32-xnnpack` E2E (ms) | `fp32-ynnpack` TTFT (ms) | `fp32-ynnpack` Decode (tok/s) | `fp32-ynnpack` E2E (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`t = 8`** | 3,822.2 ms | 9.36 tok/s | 10,555.6 ms | 5,159.9 ms | 8.26 tok/s | 12,782.6 ms |
| **`t = 16`** | 2,136.4 ms | 16.33 tok/s | 5,994.9 ms | 2,633.5 ms | 10.17 tok/s | 8,828.4 ms |
| **`t = 32` (Selected Winner)** | **1,424.1 ms** | **19.71 tok/s** | **4,619.9 ms** | **1,532.8 ms** | **10.98 tok/s** | **7,269.2 ms** |
| **`t = 40`** | 1,361.0 ms | 19.02 tok/s | 4,674.4 ms | 1,438.8 ms | 10.78 tok/s | 7,284.3 ms |

> [!TIP]
> Scaling from `t = 8` to `t = 32` on Socket 0 delivers a **2.68× TTFT reduction** (`3,822 ms → 1,424 ms`) on `litert-tensor/fp32-xnnpack` and a **3.37× TTFT reduction** (`5,160 ms → 1,533 ms`) on `litert-tensor/fp32-ynnpack`. Both backends achieve minimum E2E latency at `t = 32`.

---

## 3. Layer-Level Microbenchmarks, Roofline & `LMLB` Binary Tensor Parity (`18 / 18 PASS`)

Executed 10 full layer benchmark runs (**180 individual benchmark measurements** across `t ∈ {1, 4, 8, 16, 32}`, `Gemma4-E2B` and `Gemma4-E2B-WideMLP`, `litert-tensor/xnnpack` and native `litert-tensor/ynnpack`) with L2/LLC eviction buffer wipes between timed iterations.

### 3.1 `Gemma4-E2B` & `Gemma4-E2B-WideMLP` Breakdown (`B=1, S=1024, Threads=16`)

> [!IMPORTANT]
> Switching from the XNNPACK-wrapped shim to the native `YnnpackRunner` + `YnnpackMixinTag` makes `litert-tensor/ynnpack` **2.82× to 3.35× faster than `XnnpackRunner` on `RoPE`** (`0.156–0.158 ms` / `120.8 GB/s` vs. `0.445–0.523 ms` / `36.1–42.4 GB/s`) and **1.14× faster on `RmsNorm`** (`0.075 ms` / `168.5 GB/s` vs. `0.085 ms` / `148.0 GB/s`), while boosting `TransformerBlock_Decode` throughput by **+47%** over the shim (`62.63 GFLOP/s` vs. `42.61 GFLOP/s`).

| Preset | Layer | Arithmetic Intensity (FLOP/B) | `xnnpack` Latency (ms) ↓ | `xnnpack` GFLOP/s ↑ | `ynnpack` Latency (ms) ↓ | `ynnpack` GFLOP/s ↑ | Speedup (`xnnpack ÷ ynnpack`) | Parity Max Abs Err | Parity Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Gemma4-E2B` | `RmsNorm` | 0.50 | 0.085 ms | 73.97 | **0.075 ms** | **84.19** | **1.14× (YNNPACK wins)** | `4.47e-08` | **PASS (`0` mismatches)** |
| `Gemma4-E2B` | `RoPE` | 0.33 | 0.445 ms | 14.15 | **0.158 ms** | **39.95** | **2.82× (YNNPACK wins)** | `0.00e+00` | **PASS (`0` mismatches)** |
| `Gemma4-E2B` | `GatedMlp_SwiGLU` | 460.95 | **24.532 ms** | **2,364.26** | 31.143 ms | 1,862.41 | 0.79× | `7.75e-07` | **PASS (`0` mismatches)** |
| `Gemma4-E2B` | `GatedMlp_GeGLU` | 460.95 | **23.586 ms** | **2,459.14** | 30.914 ms | 1,876.19 | 0.76× | `6.71e-07` | **PASS (`0` mismatches)** |
| `Gemma4-E2B` | `StandardAttention_Prefill` | 456.44 | **3.534 ms** | **2,437.54** | 4.744 ms | 1,815.89 | 0.74× | `1.40e-08` | **PASS (`0` mismatches)** |
| `Gemma4-E2B` | `StandardAttention_Decode` | 3.98 | **0.158 ms** | **53.29** | 0.209 ms | 40.35 | 0.76× | `7.92e-09` | **PASS (`0` mismatches)** |
| `Gemma4-E2B` | `DecodeOptimizedAttention` | 3.98 | **0.181 ms** | **46.39** | 0.270 ms | 31.16 | 0.67× | `7.92e-09` | **PASS (`0` mismatches)** |
| `Gemma4-E2B` | `TransformerBlock_Prefill` | 519.21 | **34.245 ms** | **2,369.05** | 43.104 ms | 1,882.12 | 0.79× | `7.08e-07` | **PASS (`0` mismatches)** |
| `Gemma4-E2B` | `TransformerBlock_Decode` | 0.55 | **0.882 ms** | **89.81** | 1.265 ms | 62.63 | 0.70× | `4.02e-07` | **PASS (`0` mismatches)** |
| `Gemma4-E2B-WideMLP` | `RmsNorm` | 0.50 | 0.078 ms | 80.82 | 0.078 ms | 80.34 | 0.99× (Tie) | `4.47e-08` | **PASS (`0` mismatches)** |
| `Gemma4-E2B-WideMLP` | `RoPE` | 0.33 | 0.523 ms | 12.02 | **0.156 ms** | **40.26** | **3.35× (YNNPACK wins)** | `0.00e+00` | **PASS (`0` mismatches)** |
| `Gemma4-E2B-WideMLP` | `GatedMlp_SwiGLU` | 485.21 | **55.121 ms** | **2,104.49** | 62.901 ms | 1,844.20 | 0.88× | `1.51e-06` | **PASS (`0` mismatches)** |
| `Gemma4-E2B-WideMLP` | `GatedMlp_GeGLU` | 485.21 | **46.655 ms** | **2,486.36** | 60.935 ms | **1,903.69** | 0.77× | `1.31e-06` | **PASS (`0` mismatches)** |
| `Gemma4-E2B-WideMLP` | `StandardAttention_Prefill` | 456.44 | **3.535 ms** | **2,436.96** | 4.654 ms | 1,851.26 | 0.76× | `1.40e-08` | **PASS (`0` mismatches)** |
| `Gemma4-E2B-WideMLP` | `StandardAttention_Decode` | 3.98 | **0.157 ms** | **53.73** | 0.208 ms | 40.47 | 0.75× | `7.92e-09` | **PASS (`0` mismatches)** |
| `Gemma4-E2B-WideMLP` | `DecodeOptimizedAttention` | 3.98 | **0.179 ms** | **46.99** | 0.270 ms | 31.17 | 0.66× | `7.92e-09` | **PASS (`0` mismatches)** |
| `Gemma4-E2B-WideMLP` | `TransformerBlock_Prefill` | 516.25 | **56.271 ms** | **2,472.45** | 76.594 ms | 1,816.44 | 0.73× | `1.25e-06` | **PASS (`0` mismatches)** |
| `Gemma4-E2B-WideMLP` | `TransformerBlock_Decode` | 0.53 | **1.475 ms** | **92.12** | 2.197 ms | 61.84 | 0.67× | `6.93e-07` | **PASS (`0` mismatches)** |

````carousel
![Layer Latency Breakdown (XnnpackRunner vs. YnnpackRunner)](../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/layer_latency.png)
<!-- slide -->
![Layer Roofline Profile (XnnpackRunner vs. YnnpackRunner)](../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/layer_roofline.png)
<!-- slide -->
![CPU Utilization Breakdown (XnnpackRunner vs. YnnpackRunner)](../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/cpu.png)
<!-- slide -->
![Memory Footprint & Process Timeline (XnnpackRunner vs. YnnpackRunner)](../2026-11-03-gemma4-e2b-cpu-sapphire-rapids-sapphire-rapids/timeline.png)
````

---

## 4. Reproducibility (`100% PASS`) & Key Engineering Changes

1. **Output Reproducibility (`aeb.repro` — 100% PASS on Both Backends)**:
   - `litert-tensor/fp32-xnnpack`: **`reproducible per request`** (`24/24 identical` within process, `24/24 identical` across processes for both `greedy` and `seeded`, with `20/24` outputs changed on `seed+1` control).
   - `litert-tensor/fp32-ynnpack`: **`reproducible per request`** (`24/24 identical` within process, `24/24 identical` across processes for both `greedy` and `seeded`, with `20/24` outputs changed on `seed+1` control). Because native `YnnpackRunner` lowers compilation + load time by **5.0×** (`6,545 ms`), all 5 `aeb.repro` processes start and complete well within the 60s timeout.
2. **Native `YnnpackMixinTag` + `YnnpackRunner` Wiring**:
   - Added `transformer_ops_ynnpack.h` and `transformer_ops_ynnpack.cc` (`//tensor/examples/ops/transformer:transformer_ops_ynnpack`) so `Attention`, `RmsNorm`, and `ApplyRope` lower directly into `ynn_subgraph_t`.
   - Updated `aeb_driver.h`, `aeb_driver.cc`, `layer_builders.h`, and `litert_tensor_backend.cc` to instantiate `YnnpackRunner` (`ynn_create_subgraph` / `ynn_create_runtime`) when `--ynnpack` or `--backend=ynnpack` is selected.
3. **Slinky Compiler DAG Memoization (`simplify.cc`, `simplify_rules.h`)**:
   - Preserved reference-count DAG memoization and updated the 4 `can_flatten` rules in `simplify_rules.h` to use `when(predicate, replacement)` after syncing to HEAD (`cl/1012923423`).
