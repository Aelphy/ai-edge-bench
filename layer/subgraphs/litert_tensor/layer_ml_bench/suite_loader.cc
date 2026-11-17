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

#include "tensor/benchmarks/layer/suite_loader.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensor/benchmarks/layer/layer_spec.h"
#include "tensor/benchmarks/layer/litert_tensor_backend.h"
#include "tensor/examples/utils/minijson.h"

namespace layer_ml_bench {
namespace {

// Largest integer accepted for a suite dimension or sweep value; layer builders
// narrow these to `int`.
constexpr int64_t kMaxSuiteInt = std::numeric_limits<int32_t>::max();

// Returns the value of `v` if it is an integral JSON number within
// [min_value, kMaxSuiteInt], and nullopt otherwise.
std::optional<int64_t> ToBoundedInt64(const minijson::value& v,
                                      int64_t min_value) {
  const minijson::number* num = v.as<minijson::number>();
  if (num == nullptr || !(*num >= static_cast<double>(min_value)) ||
      *num > static_cast<double>(kMaxSuiteInt) || std::floor(*num) != *num) {
    return std::nullopt;
  }
  return static_cast<int64_t>(*num);
}

// Replaces `dst` with the integers in array field `key` of `obj`, each of
// which must lie in [1, kMaxSuiteInt]. Leaves `dst` unchanged when the field is
// absent.
absl::Status ParseInt64Array(const minijson::object& obj,
                             const std::string& key,
                             std::vector<int64_t>& dst) {
  minijson::value val;
  if (!obj.at(key, &val)) {
    return absl::OkStatus();
  }
  const minijson::array* arr = val.as<minijson::array>();
  if (arr == nullptr || arr->empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Suite field '", key, "' must be a non-empty JSON array"));
  }
  std::vector<int64_t> parsed;
  parsed.reserve(arr->size());
  for (const minijson::value& item : *arr) {
    const std::optional<int64_t> n = ToBoundedInt64(item, /*min_value=*/1);
    if (!n.has_value()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Suite field '", key, "' must contain integers in [1, ",
                       kMaxSuiteInt, "]"));
    }
    parsed.push_back(*n);
  }
  dst = std::move(parsed);
  return absl::OkStatus();
}

// Replaces `thread_counts` with array field `thread_counts` of `root`. Leaves
// `thread_counts` unchanged when the field is absent.
absl::Status ParseThreadCounts(const minijson::object& root,
                               std::vector<size_t>& thread_counts) {
  std::vector<int64_t> parsed;
  absl::Status status = ParseInt64Array(root, "thread_counts", parsed);
  if (!status.ok()) return status;
  if (parsed.empty()) {
    return absl::OkStatus();
  }
  thread_counts.clear();
  thread_counts.reserve(parsed.size());
  for (int64_t count : parsed) {
    thread_counts.push_back(static_cast<size_t>(count));
  }
  return absl::OkStatus();
}

// Replaces `families` with the built-in families named by array field
// `families` of `root`, in built-in registration order. Leaves `families`
// unchanged when the field is absent.
absl::Status ParseFamilies(const minijson::object& root,
                           std::vector<FamilyConfig>& families) {
  minijson::value val;
  if (!root.at("families", &val)) {
    return absl::OkStatus();
  }
  const minijson::array* arr = val.as<minijson::array>();
  if (arr == nullptr || arr->empty()) {
    return absl::InvalidArgumentError(
        "Suite field 'families' must be a non-empty string array");
  }
  absl::flat_hash_set<std::string> wanted;
  for (const minijson::value& item : *arr) {
    const std::string* name = item.as<std::string>();
    if (name == nullptr || name->empty()) {
      return absl::InvalidArgumentError(
          "Suite 'families' entries must be non-empty strings");
    }
    wanted.insert(*name);
  }
  std::vector<FamilyConfig> selected;
  for (const FamilyConfig& family : GetFamilyConfigs()) {
    if (wanted.erase(family.family_name) > 0) {
      selected.push_back(family);
    }
  }
  if (!wanted.empty()) {
    std::vector<std::string> unknown(wanted.begin(), wanted.end());
    std::sort(unknown.begin(), unknown.end());
    return absl::InvalidArgumentError(absl::StrCat(
        "Unknown families in --suite_file: ", absl::StrJoin(unknown, ", ")));
  }
  families = std::move(selected);
  return absl::OkStatus();
}

// Stores required integer field `key` of preset `obj` in `dst`. The value must
// lie in [1, kMaxSuiteInt].
absl::Status GetRequiredPositiveInt64(const minijson::object& obj,
                                      const std::string& key, int64_t& dst) {
  minijson::value v;
  if (!obj.at(key, &v)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Preset missing required integer field '", key, "'"));
  }
  const std::optional<int64_t> n = ToBoundedInt64(v, /*min_value=*/1);
  if (!n.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Preset field '", key, "' must be an integer in [1, ",
                     kMaxSuiteInt, "]"));
  }
  dst = *n;
  return absl::OkStatus();
}

// Stores `v` in `dst` as preset field `key`. The value must be an integer in
// [0, kMaxSuiteInt].
absl::Status ParseNonNegativeInt64(const minijson::value& v,
                                   absl::string_view key, int64_t& dst) {
  const std::optional<int64_t> n = ToBoundedInt64(v, /*min_value=*/0);
  if (!n.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Preset '", key, "' must be an integer in [0, ", kMaxSuiteInt, "]"));
  }
  dst = *n;
  return absl::OkStatus();
}

// Stores boolean field `key` of preset `obj` in `dst`. Leaves `dst` unchanged
// when the field is absent.
absl::Status ParseOptionalBool(const minijson::object& obj,
                               const std::string& key, bool& dst) {
  minijson::value v;
  if (!obj.at(key, &v)) {
    return absl::OkStatus();
  }
  const minijson::boolean* flag = v.as<minijson::boolean>();
  if (flag == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Preset '", key, "' must be a boolean"));
  }
  dst = *flag;
  return absl::OkStatus();
}

// Maps a preset `mlp_activation` string (case-insensitive) to MlpActivation.
absl::StatusOr<MlpActivation> ParseMlpActivation(const minijson::value& v) {
  const std::string* name = v.as<std::string>();
  if (name == nullptr) {
    return absl::InvalidArgumentError(
        "Preset 'mlp_activation' must be a string");
  }
  const std::string lower = absl::AsciiStrToLower(*name);
  if (lower == "geglu" || lower == "gelu" || lower == "gelu_pytorch_tanh") {
    return MlpActivation::kGeGLU;
  }
  if (lower == "swiglu" || lower == "silu") {
    return MlpActivation::kSwiGLU;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported preset mlp_activation: ", *name));
}

// Parses one inline preset object from the suite `presets` array.
absl::StatusOr<ModelPreset> ParsePresetObject(const minijson::object& obj) {
  ModelPreset preset;
  minijson::value name_v;
  const std::string* name =
      obj.at("name", &name_v) ? name_v.as<std::string>() : nullptr;
  if (name == nullptr || name->empty()) {
    return absl::InvalidArgumentError(
        "Preset object must have non-empty string 'name'");
  }
  preset.name = *name;

  absl::Status status =
      GetRequiredPositiveInt64(obj, "num_heads", preset.num_heads);
  if (!status.ok()) return status;
  status = GetRequiredPositiveInt64(obj, "head_dim", preset.head_dim);
  if (!status.ok()) return status;
  status = GetRequiredPositiveInt64(obj, "hidden_dim", preset.hidden_dim);
  if (!status.ok()) return status;
  status = GetRequiredPositiveInt64(obj, "num_kv_heads", preset.num_kv_heads);
  if (!status.ok()) return status;

  minijson::value embed_v;
  if (obj.at("embed_dim", &embed_v)) {
    status = ParseNonNegativeInt64(embed_v, "embed_dim", preset.embed_dim);
    if (!status.ok()) return status;
  }

  minijson::value eps_v;
  if (obj.at("eps", &eps_v)) {
    const minijson::number* eps = eps_v.as<minijson::number>();
    // Bounds are checked in double so the narrowing below is always defined
    // and never rounds a positive value to zero.
    if (eps == nullptr || !(*eps >= std::numeric_limits<float>::min()) ||
        *eps > std::numeric_limits<float>::max()) {
      return absl::InvalidArgumentError(
          "Preset 'eps' must be a positive finite single-precision number");
    }
    preset.eps = static_cast<float>(*eps);
  }

  minijson::value activation_v;
  if (obj.at("mlp_activation", &activation_v)) {
    absl::StatusOr<MlpActivation> activation = ParseMlpActivation(activation_v);
    if (!activation.ok()) return activation.status();
    preset.block_mlp_activation = *activation;
  }

  status = ParseOptionalBool(obj, "qk_norm", preset.qk_norm);
  if (!status.ok()) return status;
  status = ParseOptionalBool(obj, "post_norms", preset.post_norms);
  if (!status.ok()) return status;

  minijson::value window_v;
  // A null sliding_window denotes global attention, the same as omitting it.
  if (obj.at("sliding_window", &window_v) && !window_v.is<minijson::null_t>()) {
    status = ParseNonNegativeInt64(window_v, "sliding_window",
                                   preset.sliding_window);
    if (!status.ok()) return status;
  }
  return preset;
}

// Resolves one `presets` entry, which is either the name of a built-in preset
// or an inline preset object.
absl::StatusOr<ModelPreset> ParsePresetEntry(const minijson::value& item) {
  if (const std::string* name = item.as<std::string>()) {
    for (const ModelPreset& builtin : GetDefaultModelPresets()) {
      if (builtin.name == *name) {
        return builtin;
      }
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown built-in preset name in --suite_file: ", *name));
  }
  const minijson::object* obj = item.as<minijson::object>();
  if (obj == nullptr) {
    return absl::InvalidArgumentError(
        "Each entry in 'presets' must be an object or preset name string");
  }
  return ParsePresetObject(*obj);
}

// Replaces `presets` with the entries of array field `presets` of `root`.
// Leaves `presets` unchanged when the field is absent. Preset names must be
// unique because they key the registered benchmark cases.
absl::Status ParsePresets(const minijson::object& root,
                          std::vector<ModelPreset>& presets) {
  minijson::value val;
  if (!root.at("presets", &val)) {
    return absl::OkStatus();
  }
  const minijson::array* arr = val.as<minijson::array>();
  if (arr == nullptr || arr->empty()) {
    return absl::InvalidArgumentError(
        "Suite field 'presets' must be a non-empty array");
  }
  std::vector<ModelPreset> parsed;
  parsed.reserve(arr->size());
  absl::flat_hash_set<std::string> seen_names;
  for (const minijson::value& item : *arr) {
    absl::StatusOr<ModelPreset> preset = ParsePresetEntry(item);
    if (!preset.ok()) return preset.status();
    if (!seen_names.insert(preset->name).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Duplicate preset name in --suite_file: ", preset->name));
    }
    parsed.push_back(std::move(*preset));
  }
  presets = std::move(parsed);
  return absl::OkStatus();
}

// Returns the layer that `family` measures on `preset` at one operating point.
LayerSpec MakeLayerSpec(const FamilyConfig& family, const ModelPreset& preset,
                        int64_t batch_size, int64_t seq_len) {
  LayerSpec spec;
  spec.model_name = preset.name;
  spec.layer_type = family.layer_type;
  spec.attention_mode = family.attention_mode;
  // Only a transformer block runs the preset's own MLP; the standalone MLP
  // families are named after the activation they measure.
  spec.mlp_activation = (family.layer_type == LayerType::kTransformerBlock &&
                         preset.block_mlp_activation.has_value())
                            ? *preset.block_mlp_activation
                            : family.mlp_activation;
  spec.batch_size = batch_size;
  spec.seq_len_kv = seq_len;
  spec.seq_len_q =
      (family.attention_mode == AttentionMode::kDecode) ? 1 : seq_len;
  spec.num_heads = preset.num_heads;
  spec.head_dim = preset.head_dim;
  spec.num_kv_heads = preset.num_kv_heads;
  spec.hidden_dim = preset.hidden_dim;
  spec.embed_dim = preset.embed_dim;
  spec.sliding_window = preset.sliding_window;
  spec.eps = preset.eps;
  spec.qk_norm = preset.qk_norm;
  spec.post_norms = preset.post_norms;
  spec.dtype = DataType::kFP32;
  return spec;
}

SuiteSpec MakeDefaultSuiteSpec() {
  SuiteSpec suite;
  suite.presets = GetDefaultModelPresets();
  suite.families = GetFamilyConfigs();
  suite.batch_sizes = {1, 4};
  suite.seq_lengths = {128, 512, 1024, 2048};
  suite.thread_counts = {1, 4, 8, 16, 32};
  return suite;
}

}  // namespace

const std::vector<ModelPreset>& GetDefaultModelPresets() {
  static const auto* presets = new std::vector<ModelPreset>{
      {"TinyCI", 4, 64, 512, 2, 0, 1e-5f, MlpActivation::kSwiGLU, false, false,
       0},
      {"Gemma2-2B", 8, 256, 9216, 4, 0, 1e-6f, MlpActivation::kGeGLU, false,
       true, 0},
      {"Gemma3-270M", 4, 256, 2048, 1, 640, 1e-6f, MlpActivation::kGeGLU, true,
       true, 512},
      {"Gemma3-1B", 4, 256, 6912, 1, 1152, 1e-6f, MlpActivation::kGeGLU, true,
       true, 512},
      {"Gemma3-4B", 8, 256, 10240, 4, 2560, 1e-6f, MlpActivation::kGeGLU, true,
       true, 1024},
      {"Gemma4-E2B", 8, 256, 6144, 1, 1536, 1e-6f, MlpActivation::kGeGLU, true,
       true, 512},
      {"Gemma4-E2B-WideMLP", 8, 256, 12288, 1, 1536, 1e-6f,
       MlpActivation::kGeGLU, true, true, 512},
      {"Gemma4-E4B", 8, 256, 10240, 2, 2560, 1e-6f, MlpActivation::kGeGLU, true,
       true, 512},
      {"Gemma4-31B-Local", 32, 256, 21504, 16, 5376, 1e-6f,
       MlpActivation::kGeGLU, true, true, 1024},
      {"Gemma4-31B-Global", 32, 512, 21504, 4, 5376, 1e-6f,
       MlpActivation::kGeGLU, true, true, 0},
      {"Qwen3-0.6B", 16, 128, 3072, 8, 1024, 1e-6f, MlpActivation::kSwiGLU,
       true, false, 0},
      {"Qwen3-1.7B", 16, 128, 6144, 8, 2048, 1e-6f, MlpActivation::kSwiGLU,
       true, false, 0},
      {"Qwen3-4B", 32, 128, 9728, 8, 2560, 1e-6f, MlpActivation::kSwiGLU, true,
       false, 0},
      {"LLaMA3-8B", 32, 128, 14336, 8, 0, 1e-5f, MlpActivation::kSwiGLU, false,
       false, 0},
  };
  return *presets;
}

const std::vector<FamilyConfig>& GetFamilyConfigs() {
  static const auto* configs = new std::vector<FamilyConfig>{
      {"BM_RmsNorm", LayerType::kRmsNorm, AttentionMode::kPrefill,
       MlpActivation::kSwiGLU},
      {"BM_RoPE", LayerType::kRoPE, AttentionMode::kPrefill,
       MlpActivation::kSwiGLU},
      {"BM_GatedMlp_SwiGLU", LayerType::kGatedMlp, AttentionMode::kPrefill,
       MlpActivation::kSwiGLU},
      {"BM_GatedMlp_GeGLU", LayerType::kGatedMlp, AttentionMode::kPrefill,
       MlpActivation::kGeGLU},
      {"BM_StandardAttention_Prefill", LayerType::kStandardAttention,
       AttentionMode::kPrefill, MlpActivation::kSwiGLU},
      {"BM_StandardAttention_Decode", LayerType::kStandardAttention,
       AttentionMode::kDecode, MlpActivation::kSwiGLU},
      {"BM_DecodeOptimizedAttention", LayerType::kDecodeOptimizedAttention,
       AttentionMode::kDecode, MlpActivation::kSwiGLU},
      {"BM_TransformerBlock_Prefill", LayerType::kTransformerBlock,
       AttentionMode::kPrefill, MlpActivation::kSwiGLU},
      {"BM_TransformerBlock_Decode", LayerType::kTransformerBlock,
       AttentionMode::kDecode, MlpActivation::kSwiGLU},
  };
  return *configs;
}

absl::StatusOr<litert::RuntimeBackend> ParseRuntimeBackend(
    absl::string_view backend_flag, absl::string_view argv0) {
  if (backend_flag.empty()) {
    return absl::EndsWith(argv0, "_ynnpack") ? litert::RuntimeBackend::kYnnpack
                                             : litert::RuntimeBackend::kXnnpack;
  }
  if (backend_flag == "xnnpack") {
    return litert::RuntimeBackend::kXnnpack;
  }
  if (backend_flag == "ynnpack") {
    return litert::RuntimeBackend::kYnnpack;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported --backend value '", backend_flag,
                   "' (expected '', 'xnnpack', or 'ynnpack')."));
}

absl::StatusOr<SuiteSpec> ParseSuiteJson(absl::string_view json_text,
                                         absl::string_view source) {
  const std::string text(json_text);
  const char* cursor = text.c_str();
  minijson::value root;
  minijson::error err = minijson::parse(cursor, root);
  if (err != minijson::no_error) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse JSON in ", source, ": ", minijson::errstr(err)));
  }
  const minijson::object* root_obj = root.as<minijson::object>();
  if (root_obj == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Suite JSON root in ", source, " must be an object"));
  }

  SuiteSpec suite = MakeDefaultSuiteSpec();
  absl::Status status =
      ParseInt64Array(*root_obj, "batch_sizes", suite.batch_sizes);
  if (!status.ok()) return status;
  status = ParseInt64Array(*root_obj, "seq_lengths", suite.seq_lengths);
  if (!status.ok()) return status;
  status = ParseThreadCounts(*root_obj, suite.thread_counts);
  if (!status.ok()) return status;
  status = ParseFamilies(*root_obj, suite.families);
  if (!status.ok()) return status;
  status = ParsePresets(*root_obj, suite.presets);
  if (!status.ok()) return status;
  return suite;
}

absl::StatusOr<SuiteSpec> BuildSuiteSpec(absl::string_view suite_file) {
  if (suite_file.empty()) {
    return MakeDefaultSuiteSpec();
  }
  std::ifstream in{std::string(suite_file)};
  if (!in.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open --suite_file: ", suite_file));
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return ParseSuiteJson(buffer.str(), suite_file);
}

std::vector<BenchmarkCase> ExpandSuite(const SuiteSpec& suite) {
  std::vector<BenchmarkCase> cases;
  cases.reserve(suite.families.size() * suite.presets.size() *
                suite.batch_sizes.size() * suite.seq_lengths.size() *
                suite.thread_counts.size());
  for (const FamilyConfig& family : suite.families) {
    for (const ModelPreset& preset : suite.presets) {
      for (int64_t batch_size : suite.batch_sizes) {
        for (int64_t seq_len : suite.seq_lengths) {
          const LayerSpec spec =
              MakeLayerSpec(family, preset, batch_size, seq_len);
          for (size_t num_threads : suite.thread_counts) {
            cases.push_back({
                .name = absl::StrFormat("%s/%s/B:%d/S:%d/Threads:%zu",
                                        family.family_name, preset.name,
                                        batch_size, seq_len, num_threads),
                .spec = spec,
                .num_threads = num_threads,
            });
          }
        }
      }
    }
  }
  return cases;
}

}  // namespace layer_ml_bench
