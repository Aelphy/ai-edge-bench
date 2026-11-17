/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_AEB_DRIVER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_AEB_DRIVER_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensor/examples/gemma3/tokenizer.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_graph.h"
#include "tensor/examples/gemma4/gemma4_runtime.h"
#include "tensor/examples/gemma4/helpers/quantized_embedding.h"
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/runners/common_nnpack/runner.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tflite/delegates/xnnpack/weight_cache.h"

namespace litert::tensor::examples::gemma4 {

struct SamplingParams {
  float temperature = 0.0f;
  int top_k = 0;
  float top_p = 1.0f;
  uint32_t seed = 0;
};

struct GenerateRequest {
  std::string prompt;
  std::vector<int32_t> prompt_ids;
  int max_tokens = 256;
  bool ignore_eos = false;
  bool return_text = true;
  bool return_prompt_ids = false;
  bool cache_prefix = false;
  int prefix_tokens = 0;
  absl::flat_hash_set<int32_t> stop_ids;
  SamplingParams sampling;
};

struct GenerateResult {
  std::vector<int32_t> prompt_ids;
  int n_prompt = 0;
  int n_cached = 0;
  int n_prefilled = 0;
  int n_gen = 0;
  bool stopped_on_eos = false;
  int64_t t_req_ns = 0;
  int64_t t_prefill_done_ns = 0;
  int64_t t_first_ns = 0;
  int64_t t_end_ns = 0;
  std::vector<int64_t> token_ns;
  std::vector<int32_t> gen_ids;
  std::string text;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
};

// Samples a token from `logits` using greedy argmax when `temperature <= 0`
// or top-k / top-p nucleus sampling with `rng` when `temperature > 0`.
int32_t SampleToken(absl::Span<const float> logits,
                    const SamplingParams& sampling, std::mt19937* rng);

// Creates prefill graph inputs for `input_seq_len` new tokens attending over
// `kv_cache_len` already-cached prefix tokens (`kv_cache_len == 0` for cold
// prefill).
absl::StatusOr<Gemma4Inputs<>> CreatePrefillInputs(const Config& config,
                                                   int input_seq_len,
                                                   int kv_cache_len);

// Long-lived Gemma 4 Tensor API session speaking the AI Edge Bench JSON-lines
// protocol on `in` / `out`.
class AebGemma4Session {
 public:
  struct Options {
    std::string backend_name = "xnnpack";
    std::string dtype = "fp32";
    std::string model_variant = "E2B";
    int num_threads = 4;
    int ctx = 4096;
    std::string cache_dir;
    bool cache_prefix = false;
  };

  static absl::StatusOr<std::unique_ptr<AebGemma4Session>> Create(
      const Config& config, LoadedTensors loaded_tensors,
      std::optional<GemmaTokenizerSP> tokenizer,
      tflite::xnnpack::MMapWeightCacheProvider* weight_cache,
      const Options& options, int64_t load_ns);

  // Processes a single JSON request line and returns the JSON response line.
  // Sets `*should_quit = true` when `"op": "quit"` is received.
  std::string HandleJsonLine(absl::string_view line, bool* should_quit);

  // Runs the full JSON-lines server loop (`ready` event followed by `stdin`
  // request processing until `"quit"` or EOF).
  absl::Status Serve(std::istream& in, std::ostream& out);

  // Executes a single generation request.
  absl::StatusOr<GenerateResult> Generate(const GenerateRequest& req);

  // Tokenizes `text` if a SentencePiece tokenizer is loaded.
  absl::StatusOr<std::vector<int32_t>> Tokenize(const std::string& text,
                                                bool add_bos) const;

 private:
  struct PrefillInstance {
    int bucket_seq_len = 0;
    int kv_cache_len = 0;
    Gemma4Inputs<> inputs;
    Gemma4Outputs<> outputs;
    std::unique_ptr<NnpackRunner> runner;
    XnnpackRunner* xnn_runner = nullptr;
  };

  AebGemma4Session(const Config& config, LoadedTensors loaded_tensors,
                   std::optional<GemmaTokenizerSP> tokenizer,
                   tflite::xnnpack::MMapWeightCacheProvider* weight_cache,
                   const Options& options, int64_t load_ns,
                   Gemma4Inputs<> decode_inputs, Gemma4Outputs<> decode_outputs,
                   std::unique_ptr<NnpackRunner> decode_runner,
                   XnnpackRunner* decode_xnn_runner);

  int SelectPrefillBucket(int seq_len) const;

  absl::StatusOr<PrefillInstance*> GetOrCompilePrefill(int bucket_seq_len,
                                                       int kv_cache_len);

  absl::Status RunPrefill(
      PrefillInstance& inst, absl::Span<const int32_t> token_ids,
      int kv_cache_len, const std::vector<int>& sharing_patterns,
      const std::vector<std::vector<float>>* cached_keys,
      const std::vector<std::vector<float>>* cached_values);

  absl::Status PrepareRunnerWithCache(NnpackRunner& runner,
                                      XnnpackRunner* xnn_runner);

  Config config_;
  LoadedTensors loaded_tensors_;
  std::optional<GemmaTokenizerSP> tokenizer_;
  tflite::xnnpack::MMapWeightCacheProvider* weight_cache_ = nullptr;
  Options options_;
  int64_t load_ns_ = 0;

  Gemma4Inputs<> decode_inputs_;
  Gemma4Outputs<> decode_outputs_;
  std::unique_ptr<NnpackRunner> decode_runner_;
  XnnpackRunner* decode_xnn_runner_ = nullptr;

  absl::flat_hash_map<std::pair<int, int>, std::unique_ptr<PrefillInstance>>
      prefill_cache_;

  std::vector<int32_t> cached_prefix_ids_;
  std::vector<std::vector<float>> cached_prefix_keys_;
  std::vector<std::vector<float>> cached_prefix_values_;
};

}  // namespace litert::tensor::examples::gemma4

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_AEB_DRIVER_H_
