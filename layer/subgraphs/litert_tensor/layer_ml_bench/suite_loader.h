// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSOR_BENCHMARKS_LAYER_SUITE_LOADER_H_
#define TENSOR_BENCHMARKS_LAYER_SUITE_LOADER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensor/benchmarks/layer/layer_spec.h"
#include "tensor/benchmarks/layer/litert_tensor_backend.h"

namespace layer_ml_bench {

// Model architecture preset used to instantiate LayerSpec benchmarks.
struct ModelPreset {
  std::string name;
  int64_t num_heads = 0;
  int64_t head_dim = 0;
  int64_t hidden_dim = 0;
  int64_t num_kv_heads = 0;
  int64_t embed_dim = 0;
  float eps = 1e-5f;
  std::optional<MlpActivation> block_mlp_activation = std::nullopt;
  bool qk_norm = false;
  bool post_norms = false;
  int64_t sliding_window = 0;
};

// Mapping from a Google Benchmark family prefix (e.g., "BM_RmsNorm") to its
// LayerSpec execution configuration.
struct FamilyConfig {
  std::string family_name;
  LayerType layer_type;
  AttentionMode attention_mode;
  MlpActivation mlp_activation;
};

// Resolved benchmark suite matrix of presets, families, and sweep axes.
struct SuiteSpec {
  std::vector<ModelPreset> presets;
  std::vector<FamilyConfig> families;
  std::vector<int64_t> batch_sizes;
  std::vector<int64_t> seq_lengths;
  std::vector<size_t> thread_counts;
};

// One benchmark of a suite: a layer at a single operating point.
struct BenchmarkCase {
  // "<family>/<preset>/B:<batch_size>/S:<seq_len>/Threads:<num_threads>".
  std::string name;
  LayerSpec spec;
  size_t num_threads = 1;
};

// Returns the built-in public model presets registered when no suite overrides
// `presets`.
const std::vector<ModelPreset>& GetDefaultModelPresets();

// Returns all supported BM_* benchmark family definitions.
const std::vector<FamilyConfig>& GetFamilyConfigs();

// Parses and validates the `--backend` CLI flag (`""`, `"xnnpack"`, or
// `"ynnpack"`). When `backend_flag` is empty and `argv0` ends with
// `"_ynnpack"`, selects `RuntimeBackend::kYnnpack`; otherwise defaults to
// `kXnnpack`.
absl::StatusOr<litert::RuntimeBackend> ParseRuntimeBackend(
    absl::string_view backend_flag, absl::string_view argv0 = "");

// Parses a declarative JSON suite string into a validated `SuiteSpec`. Fields
// absent from the JSON keep their built-in defaults. `source` only labels
// error messages. Returns InvalidArgument for malformed JSON, a non-object
// root, unknown families or built-in preset names, duplicate preset names,
// and numeric fields outside their range: sweep values and required preset
// dimensions in [1, INT32_MAX], `embed_dim` and `sliding_window` in
// [0, INT32_MAX], and `eps` a positive finite float.
absl::StatusOr<SuiteSpec> ParseSuiteJson(absl::string_view json_text,
                                         absl::string_view source = "<suite>");

// Builds a `SuiteSpec` from defaults or by reading and validating `suite_file`.
absl::StatusOr<SuiteSpec> BuildSuiteSpec(absl::string_view suite_file);

// Expands `suite` into the cross product of its families, presets, batch
// sizes, sequence lengths, and thread counts, nested in that order. Decode
// families query one token against a context of the sequence length; prefill
// families query the whole sequence. A preset's own MLP activation applies to
// transformer blocks only; every other family keeps its own activation.
std::vector<BenchmarkCase> ExpandSuite(const SuiteSpec& suite);

}  // namespace layer_ml_bench

#endif  // TENSOR_BENCHMARKS_LAYER_SUITE_LOADER_H_
