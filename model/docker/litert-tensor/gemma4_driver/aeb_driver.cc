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

#include "tensor/examples/gemma4/aeb_driver.h"

#include <algorithm>
#include <array>
#include <chrono>  // NOLINT
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/ynnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/tokenizer.h"
#include "tensor/examples/gemma3/util.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_graph.h"
#include "tensor/examples/gemma4/gemma4_runtime.h"
#include "tensor/examples/gemma4/helpers/rope.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"  // IWYU pragma: keep
#include "tensor/examples/ops/transformer/transformer_ops_ynnpack.h"  // IWYU pragma: keep
#include "tensor/examples/utils/minijson.h"
#include "tensor/runners/common_nnpack/runner.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/runners/ynnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tflite/delegates/xnnpack/weight_cache.h"

namespace litert::tensor::examples::gemma4 {
namespace {

int64_t NowMonotonicNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string EscapeJsonString(absl::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out.append("\\\"");
        break;
      case '\\':
        out.append("\\\\");
        break;
      case '\b':
        out.append("\\b");
        break;
      case '\f':
        out.append("\\f");
        break;
      case '\n':
        out.append("\\n");
        break;
      case '\r':
        out.append("\\r");
        break;
      case '\t':
        out.append("\\t");
        break;
      default:
        if (c < 0x20) {
          absl::StrAppend(&out, absl::StrFormat("\\u%04x", c));
        } else {
          out.push_back(static_cast<char>(c));
        }
        break;
    }
  }
  out.push_back('"');
  return out;
}

std::string FormatRequestId(const std::optional<int64_t>& id_int,
                            const std::optional<std::string>& id_str) {
  if (id_int.has_value()) {
    return absl::StrCat(*id_int);
  }
  if (id_str.has_value()) {
    return EscapeJsonString(*id_str);
  }
  return "null";
}

std::string MakeErrorResponse(const std::optional<int64_t>& id_int,
                              const std::optional<std::string>& id_str,
                              absl::string_view error_msg) {
  return absl::StrCat("{\"id\":", FormatRequestId(id_int, id_str),
                      ",\"ok\":false,\"error\":", EscapeJsonString(error_msg),
                      "}");
}

void FillPrefillAttentionMask(int input_seq_len, int kv_cache_len,
                              bool is_local, int sliding_window_size,
                              absl::Span<float> mask) {
  const int total_seq = kv_cache_len + input_seq_len;
  const float neg_inf = std::numeric_limits<float>::lowest();
  for (int i = 0; i < input_seq_len; ++i) {
    const int q_pos = kv_cache_len + i;
    const size_t row_offset = static_cast<size_t>(i) * total_seq;
    for (int j = 0; j < total_seq; ++j) {
      const bool is_causal_masked = (j > q_pos);
      const bool is_sliding_masked = is_local && (sliding_window_size > 0) &&
                                     (q_pos - j >= sliding_window_size);
      mask[row_offset + j] =
          (is_causal_masked || is_sliding_masked) ? neg_inf : 0.0f;
    }
  }
}

}  // namespace

int32_t SampleToken(absl::Span<const float> logits,
                    const SamplingParams& sampling, std::mt19937* rng) {
  if (logits.empty()) {
    return 0;
  }
  if (sampling.temperature <= 0.0f || rng == nullptr) {
    return static_cast<int32_t>(absl::c_max_element(logits) - logits.begin());
  }

  const int vocab_size = static_cast<int>(logits.size());
  const int k = (sampling.top_k > 0 && sampling.top_k < vocab_size)
                    ? sampling.top_k
                    : vocab_size;

  std::vector<std::pair<float, int32_t>> candidates(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    candidates[i] = {logits[i], static_cast<int32_t>(i)};
  }
  auto cmp = [](const std::pair<float, int32_t>& a,
                const std::pair<float, int32_t>& b) {
    if (a.first != b.first) {
      return a.first > b.first;
    }
    return a.second < b.second;
  };
  if (k < vocab_size) {
    std::partial_sort(candidates.begin(), candidates.begin() + k,
                      candidates.end(), cmp);
    candidates.resize(k);
  } else {
    std::sort(candidates.begin(), candidates.end(), cmp);
  }

  const double inv_temp = 1.0 / static_cast<double>(sampling.temperature);
  const double max_scaled =
      static_cast<double>(candidates[0].first) * inv_temp;
  std::vector<double> probs(candidates.size());
  double sum = 0.0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    probs[i] =
        std::exp(static_cast<double>(candidates[i].first) * inv_temp -
                 max_scaled);
    sum += probs[i];
  }
  if (sum <= 0.0) {
    return candidates[0].second;
  }
  for (double& p : probs) {
    p /= sum;
  }

  if (sampling.top_p > 0.0f && sampling.top_p < 1.0f) {
    double csum = 0.0;
    size_t keep = candidates.size();
    for (size_t i = 0; i < probs.size(); ++i) {
      csum += probs[i];
      if (csum >= static_cast<double>(sampling.top_p)) {
        keep = i + 1;
        break;
      }
    }
    candidates.resize(keep);
    probs.resize(keep);
  }

  std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
  const size_t chosen = dist(*rng);
  return candidates[chosen].second;
}

absl::StatusOr<Gemma4Inputs<>> CreatePrefillInputs(const Config& config,
                                                   int input_seq_len,
                                                   int kv_cache_len) {
  if (input_seq_len <= 0) {
    return absl::InvalidArgumentError("input_seq_len must be positive");
  }
  if (kv_cache_len < 0) {
    return absl::InvalidArgumentError("kv_cache_len must be non-negative");
  }
  if (kv_cache_len == 0 && input_seq_len > 1) {
    return CreateGemma4Inputs(config, input_seq_len, /*kv_cache_len=*/0);
  }

  const int batch_size = 1;
  const int total_seq = kv_cache_len + input_seq_len;
  Gemma4Inputs<> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input",
       .type = Type::kFP32,
       .shape = {batch_size, input_seq_len, config.embed_dim}});

  auto alloc_rope = [&](absl::string_view cos_name, absl::string_view sin_name,
                        int head_dim, float base_freq, float proportion) {
    Shape shape = {1, 1, input_seq_len, head_dim};
    auto cos_buf = OwningCpuBuffer::Allocate<Type::kFP32>(shape);
    auto sin_buf = OwningCpuBuffer::Allocate<Type::kFP32>(shape);
    RopeCosSin(kv_cache_len, input_seq_len, head_dim, base_freq, proportion,
               cos_buf->Span<float>(), sin_buf->Span<float>());
    TensorHandle cos_h({.name = std::string(cos_name),
                        .type = Type::kFP32,
                        .shape = shape,
                        .buffer = cos_buf});
    TensorHandle sin_h({.name = std::string(sin_name),
                        .type = Type::kFP32,
                        .shape = std::move(shape),
                        .buffer = sin_buf});
    return std::make_pair(std::move(cos_h), std::move(sin_h));
  };

  auto [g_cos, g_sin] = alloc_rope(
      "rope_global_cos", "rope_global_sin", config.global_key_size,
      config.global_base_frequency, config.global_rope_proportion);
  inputs.rope_global_cos = Tensor<>(std::move(g_cos));
  inputs.rope_global_sin = Tensor<>(std::move(g_sin));

  auto [l_cos, l_sin] = alloc_rope(
      "rope_local_cos", "rope_local_sin", config.head_dim,
      config.local_base_frequency, config.local_rope_proportion);
  inputs.rope_local_cos = Tensor<>(std::move(l_cos));
  inputs.rope_local_sin = Tensor<>(std::move(l_sin));

  auto alloc_mask = [&](absl::string_view name, bool is_local) {
    Shape shape = {1, 1, input_seq_len, total_seq};
    auto buf = OwningCpuBuffer::Allocate<Type::kFP32>(shape);
    FillPrefillAttentionMask(input_seq_len, kv_cache_len, is_local,
                             config.sliding_window_size, buf->Span<float>());
    return TensorHandle({.name = std::string(name),
                         .type = Type::kFP32,
                         .shape = std::move(shape),
                         .buffer = buf});
  };

  inputs.global_attention_mask =
      Tensor<>(alloc_mask("global_attention_mask", /*is_local=*/false));
  inputs.sliding_attention_mask =
      Tensor<>(alloc_mask("sliding_attention_mask", /*is_local=*/true));

  inputs.key_caches.reserve(config.num_layers);
  inputs.value_caches.reserve(config.num_layers);
  for (int i = 0; i < config.num_layers; ++i) {
    bool is_global = config.GetLayerType(i) == Config::LayerType::kGlobal;
    int head_dim = is_global ? config.global_key_size : config.head_dim;
    inputs.key_caches.emplace_back(TensorInit{
        .name = absl::StrCat("key_cache_", i),
        .type = Type::kFP32,
        .shape = {batch_size, config.num_kv_heads, kv_cache_len, head_dim}});
    inputs.value_caches.emplace_back(TensorInit{
        .name = absl::StrCat("value_cache_", i),
        .type = Type::kFP32,
        .shape = {batch_size, config.num_kv_heads, kv_cache_len, head_dim}});
  }

  inputs.per_layer_token_embeddings.reserve(config.num_layers);
  for (int l = 0; l < config.num_layers; ++l) {
    inputs.per_layer_token_embeddings.emplace_back(TensorInit{
        .name = absl::StrCat("per_layer_token_embedding_", l),
        .type = Type::kFP32,
        .shape = {batch_size, input_seq_len, config.per_layer_input_dim}});
  }

  return inputs;
}

AebGemma4Session::AebGemma4Session(
    const Config& config, LoadedTensors loaded_tensors,
    std::optional<GemmaTokenizerSP> tokenizer,
    tflite::xnnpack::MMapWeightCacheProvider* weight_cache,
    const Options& options, int64_t load_ns, Gemma4Inputs<> decode_inputs,
    Gemma4Outputs<> decode_outputs,
    std::unique_ptr<NnpackRunner> decode_runner,
    XnnpackRunner* decode_xnn_runner)
    : config_(config),
      loaded_tensors_(std::move(loaded_tensors)),
      tokenizer_(std::move(tokenizer)),
      weight_cache_(weight_cache),
      options_(options),
      load_ns_(load_ns),
      decode_inputs_(std::move(decode_inputs)),
      decode_outputs_(std::move(decode_outputs)),
      decode_runner_(std::move(decode_runner)),
      decode_xnn_runner_(decode_xnn_runner) {}

absl::Status AebGemma4Session::PrepareRunnerWithCache(
    NnpackRunner& runner, XnnpackRunner* xnn_runner) {
  if (weight_cache_ != nullptr && xnn_runner != nullptr) {
    xnn_runner->SetWeightsCache(&weight_cache_->GetCacheProvider());
    if (weight_cache_->CanStartBuildStep()) {
      if (!weight_cache_->StartBuildStep()) {
        return absl::InternalError(
            "Failed to start build step for XNNPACK weight cache.");
      }
      absl::Status status = runner.PrepareRuntime();
      if (!weight_cache_->StopBuildStep()) {
        ABSL_LOG(ERROR) << "Failed to stop build step for XNNPACK weight cache";
      }
      return status;
    }
  }
  return runner.PrepareRuntime();
}

absl::StatusOr<std::unique_ptr<AebGemma4Session>> AebGemma4Session::Create(
    const Config& config, LoadedTensors loaded_tensors,
    std::optional<GemmaTokenizerSP> tokenizer,
    tflite::xnnpack::MMapWeightCacheProvider* weight_cache,
    const Options& options, int64_t load_ns) {
  const int64_t t_compile0 = NowMonotonicNs();
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Gemma4Inputs<> decode_inputs,
      CreateGemma4Inputs(config, /*input_seq_len=*/1, /*kv_cache_len=*/16));

  Gemma4Outputs<> decode_outputs;
  std::unique_ptr<NnpackRunner> decode_runner;
  XnnpackRunner* decode_xnn_runner = nullptr;
  if (options.backend_name == "ynnpack") {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Gemma4Outputs<YnnpackMixinTag> decode_outputs_tagged,
        BuildGemma4Graph(ChangeMixinsTo<YnnpackMixinTag>(decode_inputs),
                         loaded_tensors.weights, config));
    decode_outputs = ChangeMixinsTo(decode_outputs_tagged);
    LRT_TENSOR_RETURN_IF_ERROR(decode_outputs.logits.GetStatus());
    LRT_TENSOR_ASSIGN_OR_RETURN(
        YnnpackRunner ynn_runner,
        YnnpackRunner::Create(decode_outputs.GetAllHandles()));
    decode_runner = std::make_unique<YnnpackRunner>(std::move(ynn_runner));
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Gemma4Outputs<XnnpackMixinTag> decode_outputs_tagged,
        BuildGemma4Graph(ChangeMixinsTo<XnnpackMixinTag>(decode_inputs),
                         loaded_tensors.weights, config));
    decode_outputs = ChangeMixinsTo(decode_outputs_tagged);
    LRT_TENSOR_RETURN_IF_ERROR(decode_outputs.logits.GetStatus());
    LRT_TENSOR_ASSIGN_OR_RETURN(
        XnnpackRunner xnn_runner,
        XnnpackRunner::Create(decode_outputs.GetAllHandles()));
    auto xnn_ptr = std::make_unique<XnnpackRunner>(std::move(xnn_runner));
    decode_xnn_runner = xnn_ptr.get();
    decode_runner = std::move(xnn_ptr);
  }
  LRT_TENSOR_RETURN_IF_ERROR(
      decode_runner->SetNumThreads(options.num_threads));

  auto session = std::unique_ptr<AebGemma4Session>(new AebGemma4Session(
      config, std::move(loaded_tensors), std::move(tokenizer), weight_cache,
      options, load_ns, std::move(decode_inputs), std::move(decode_outputs),
      std::move(decode_runner), decode_xnn_runner));
  LRT_TENSOR_RETURN_IF_ERROR(session->PrepareRunnerWithCache(
      *session->decode_runner_, session->decode_xnn_runner_));
  session->load_ns_ += (NowMonotonicNs() - t_compile0);
  return session;
}

int AebGemma4Session::SelectPrefillBucket(int seq_len) const {
  const int clamped = std::max(1, seq_len);
  if (config_.num_layers < 8) {
    return ((clamped + 7) / 8) * 8;
  }
  static constexpr int kBuckets[] = {16,  64,   128,  256, 512,
                                     896, 1024, 2048, 4096};
  for (int b : kBuckets) {
    if (clamped <= b) {
      return b;
    }
  }
  return ((clamped + 511) / 512) * 512;
}

absl::StatusOr<AebGemma4Session::PrefillInstance*>
AebGemma4Session::GetOrCompilePrefill(int bucket_seq_len, int kv_cache_len) {
  const std::pair<int, int> key = {bucket_seq_len, kv_cache_len};
  auto it = prefill_cache_.find(key);
  if (it != prefill_cache_.end()) {
    return it->second.get();
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      Gemma4Inputs<> prefill_inputs,
      CreatePrefillInputs(config_, bucket_seq_len, kv_cache_len));

  Gemma4Outputs<> prefill_outputs;
  std::unique_ptr<NnpackRunner> prefill_runner;
  XnnpackRunner* prefill_xnn_runner = nullptr;
  if (options_.backend_name == "ynnpack") {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Gemma4Outputs<YnnpackMixinTag> prefill_outputs_tagged,
        BuildGemma4Graph(ChangeMixinsTo<YnnpackMixinTag>(prefill_inputs),
                         loaded_tensors_.weights, config_));
    prefill_outputs = ChangeMixinsTo(prefill_outputs_tagged);
    LRT_TENSOR_RETURN_IF_ERROR(prefill_outputs.logits.GetStatus());
    LRT_TENSOR_ASSIGN_OR_RETURN(
        YnnpackRunner ynn_runner,
        YnnpackRunner::Create(prefill_outputs.GetAllHandles()));
    prefill_runner = std::make_unique<YnnpackRunner>(std::move(ynn_runner));
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Gemma4Outputs<XnnpackMixinTag> prefill_outputs_tagged,
        BuildGemma4Graph(ChangeMixinsTo<XnnpackMixinTag>(prefill_inputs),
                         loaded_tensors_.weights, config_));
    prefill_outputs = ChangeMixinsTo(prefill_outputs_tagged);
    LRT_TENSOR_RETURN_IF_ERROR(prefill_outputs.logits.GetStatus());
    LRT_TENSOR_ASSIGN_OR_RETURN(
        XnnpackRunner xnn_runner,
        XnnpackRunner::Create(prefill_outputs.GetAllHandles()));
    auto xnn_ptr = std::make_unique<XnnpackRunner>(std::move(xnn_runner));
    prefill_xnn_runner = xnn_ptr.get();
    prefill_runner = std::move(xnn_ptr);
  }
  LRT_TENSOR_RETURN_IF_ERROR(
      prefill_runner->SetNumThreads(options_.num_threads));
  LRT_TENSOR_RETURN_IF_ERROR(
      PrepareRunnerWithCache(*prefill_runner, prefill_xnn_runner));

  auto instance = std::make_unique<PrefillInstance>(PrefillInstance{
      bucket_seq_len, kv_cache_len, std::move(prefill_inputs),
      std::move(prefill_outputs), std::move(prefill_runner),
      prefill_xnn_runner});
  PrefillInstance* ptr = instance.get();
  prefill_cache_.emplace(key, std::move(instance));
  return ptr;
}

absl::Status AebGemma4Session::RunPrefill(
    PrefillInstance& inst, absl::Span<const int32_t> token_ids,
    int kv_cache_len, const std::vector<int>& sharing_patterns,
    const std::vector<std::vector<float>>* cached_keys,
    const std::vector<std::vector<float>>* cached_values) {
  const int bucket_seq_len = inst.bucket_seq_len;
  if (static_cast<int>(token_ids.size()) > bucket_seq_len) {
    return absl::InvalidArgumentError(
        "token_ids exceeds compiled prefill bucket_seq_len");
  }
  std::vector<int32_t> padded_tokens(bucket_seq_len, 0);
  std::copy(token_ids.begin(), token_ids.end(), padded_tokens.begin());

  std::vector<float> embedded_input(
      static_cast<size_t>(bucket_seq_len) * config_.embed_dim);
  std::vector<std::vector<float>> per_layer_tok_embs(
      config_.num_layers,
      std::vector<float>(static_cast<size_t>(bucket_seq_len) *
                         config_.per_layer_input_dim));
  LRT_TENSOR_RETURN_IF_ERROR(loaded_tensors_.token_embedding->Lookup(
      padded_tokens, absl::MakeSpan(embedded_input)));
  LRT_TENSOR_RETURN_IF_ERROR(loaded_tensors_.emb_per_layer_table->LookupPerLayer(
      padded_tokens, config_.num_layers, config_.per_layer_input_dim,
      absl::MakeSpan(per_layer_tok_embs)));

  LRT_TENSOR_RETURN_IF_ERROR(
      inst.runner->SetInput(inst.inputs.embedded_input, embedded_input));
  for (int l = 0; l < config_.num_layers; ++l) {
    LRT_TENSOR_RETURN_IF_ERROR(inst.runner->SetInput(
        inst.inputs.per_layer_token_embeddings[l], per_layer_tok_embs[l]));
  }

  if (kv_cache_len > 0) {
    if (cached_keys == nullptr || cached_values == nullptr) {
      return absl::InternalError(
          "cached_keys and cached_values are required when kv_cache_len > 0");
    }
    for (int i = 0; i < config_.num_layers; ++i) {
      if (sharing_patterns[i] != i) {
        continue;
      }
      LRT_TENSOR_RETURN_IF_ERROR(
          inst.runner->SetInput(inst.inputs.key_caches[i], (*cached_keys)[i]));
      LRT_TENSOR_RETURN_IF_ERROR(inst.runner->SetInput(
          inst.inputs.value_caches[i], (*cached_values)[i]));
    }
  }

  return inst.runner->Run();
}

absl::StatusOr<std::vector<int32_t>> AebGemma4Session::Tokenize(
    const std::string& text, bool add_bos) const {
  if (!tokenizer_.has_value()) {
    return absl::FailedPreconditionError(
        "No SentencePiece tokenizer loaded; pass prompt_ids or --tokenizer.");
  }
  return tokenizer_->Encode(text, add_bos);
}

absl::StatusOr<GenerateResult> AebGemma4Session::Generate(
    const GenerateRequest& req) {
  if (req.max_tokens <= 0) {
    return absl::InvalidArgumentError("max_tokens must be positive");
  }

  std::vector<int32_t> prompt_ids = req.prompt_ids;
  if (prompt_ids.empty()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(prompt_ids,
                                Tokenize(req.prompt, /*add_bos=*/true));
  }
  if (prompt_ids.empty()) {
    return absl::InvalidArgumentError("Prompt tokenized to 0 tokens");
  }

  const int seq_len = static_cast<int>(prompt_ids.size());
  const int max_tokens = req.max_tokens;
  const bool use_cache = (req.cache_prefix || options_.cache_prefix) &&
                         req.prefix_tokens > 0 && req.prefix_tokens < seq_len;
  const int prefix_len = use_cache ? req.prefix_tokens : 0;
  const int suffix_len = seq_len - prefix_len;

  std::optional<std::mt19937> rng;
  std::mt19937* rng_ptr = nullptr;
  if (req.sampling.temperature > 0.0f) {
    rng.emplace(req.sampling.seed);
    rng_ptr = &(*rng);
  }

  const std::vector<int> sharing_patterns = GetKvCacheSharingPatterns(config_);
  std::vector<std::vector<float>> host_key_caches(config_.num_layers);
  std::vector<std::vector<float>> host_value_caches(config_.num_layers);
  const int max_cache_len = seq_len + max_tokens;
  const int batch_size = 1;

  // Ensure the prefix (if any) and prefill runner are compiled before starting
  // the timed request window.
  if (use_cache) {
    std::vector<int32_t> prefix_ids(prompt_ids.begin(),
                                    prompt_ids.begin() + prefix_len);
    if (cached_prefix_ids_ != prefix_ids) {
      const int prefix_bucket = SelectPrefillBucket(prefix_len);
      LRT_TENSOR_ASSIGN_OR_RETURN(
          PrefillInstance * prefix_inst,
          GetOrCompilePrefill(prefix_bucket, /*kv_cache_len=*/0));
      LRT_TENSOR_RETURN_IF_ERROR(RunPrefill(
          *prefix_inst, prefix_ids, /*kv_cache_len=*/0, sharing_patterns,
          /*cached_keys=*/nullptr, /*cached_values=*/nullptr));
      cached_prefix_keys_.assign(config_.num_layers, {});
      cached_prefix_values_.assign(config_.num_layers, {});
      for (int i = 0; i < config_.num_layers; ++i) {
        if (sharing_patterns[i] != i) {
          continue;
        }
        bool is_global = config_.GetLayerType(i) == Config::LayerType::kGlobal;
        int head_dim = is_global ? config_.global_key_size : config_.head_dim;
        const size_t pref_h_elems =
            static_cast<size_t>(prefix_len) * head_dim;
        const size_t src_h_elems =
            static_cast<size_t>(prefix_inst->bucket_seq_len) * head_dim;
        cached_prefix_keys_[i].resize(
            static_cast<size_t>(config_.num_kv_heads) * pref_h_elems);
        cached_prefix_values_[i].resize(
            static_cast<size_t>(config_.num_kv_heads) * pref_h_elems);
        LRT_TENSOR_ASSIGN_OR_RETURN(
            auto k_span, prefix_inst->runner->ReadOutputAs<float>(
                             prefix_inst->outputs.key_caches[i]));
        LRT_TENSOR_ASSIGN_OR_RETURN(
            auto v_span, prefix_inst->runner->ReadOutputAs<float>(
                             prefix_inst->outputs.value_caches[i]));
        for (int h = 0; h < config_.num_kv_heads; ++h) {
          std::copy_n(k_span.data() + h * src_h_elems, pref_h_elems,
                      cached_prefix_keys_[i].data() + h * pref_h_elems);
          std::copy_n(v_span.data() + h * src_h_elems, pref_h_elems,
                      cached_prefix_values_[i].data() + h * pref_h_elems);
        }
      }
      cached_prefix_ids_ = std::move(prefix_ids);
    }
  }

  const int suffix_bucket = SelectPrefillBucket(suffix_len);
  LRT_TENSOR_ASSIGN_OR_RETURN(
      PrefillInstance * prefill_inst,
      GetOrCompilePrefill(suffix_bucket, prefix_len));

  // Timed request starts here.
  const int64_t t_req = NowMonotonicNs();
  PrefillTiming prefill_timing;
  DecodeTiming decode_timing;

  std::vector<int32_t> suffix_ids(prompt_ids.begin() + prefix_len,
                                  prompt_ids.end());
  {
    Timer::LapScope lap = prefill_timing.prefill.Lap();
    LRT_TENSOR_RETURN_IF_ERROR(RunPrefill(
        *prefill_inst, suffix_ids, /*kv_cache_len=*/prefix_len,
        sharing_patterns, use_cache ? &cached_prefix_keys_ : nullptr,
        use_cache ? &cached_prefix_values_ : nullptr));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto prefill_logits_locked,
      prefill_inst->runner->ReadOutputAs<float>(prefill_inst->outputs.logits));
  if (prefill_logits_locked.size() <
      static_cast<size_t>(suffix_len) * config_.vocab_size) {
    return absl::InternalError("Prefill logits buffer smaller than expected");
  }
  absl::Span<const float> last_prefill_logits(
      prefill_logits_locked.begin() + (suffix_len - 1) * config_.vocab_size,
      config_.vocab_size);
  int32_t current_token =
      SampleToken(last_prefill_logits, req.sampling, rng_ptr);
  const int64_t t_prefill_done = NowMonotonicNs();

  auto is_stop = [&](int32_t tok) {
    return IsStopToken(tok) || req.stop_ids.contains(tok);
  };

  GenerateResult result;
  result.prompt_ids = std::move(prompt_ids);
  result.n_prompt = seq_len;
  result.n_cached = prefix_len;
  result.n_prefilled = suffix_len;
  result.t_req_ns = t_req;
  result.t_prefill_done_ns = t_prefill_done;

  if (!req.ignore_eos && is_stop(current_token)) {
    result.stopped_on_eos = true;
    result.t_first_ns = 0;
    result.t_end_ns = NowMonotonicNs();
    result.n_gen = 0;
    result.prefill_ms = (t_prefill_done - t_req) / 1e6;
    return result;
  }

  result.gen_ids.push_back(current_token);
  result.token_ns.push_back(t_prefill_done);

  if (max_tokens > 1) {
    // Initialize decode KV caches from prefill (or prefix + suffix).
    for (int i = 0; i < config_.num_layers; ++i) {
      if (sharing_patterns[i] != i) {
        continue;
      }
      bool is_global = config_.GetLayerType(i) == Config::LayerType::kGlobal;
      int head_dim = is_global ? config_.global_key_size : config_.head_dim;
      const size_t max_elems =
          static_cast<size_t>(config_.num_kv_heads) * max_cache_len * head_dim;
      host_key_caches[i].assign(max_elems, 0.0f);
      host_value_caches[i].assign(max_elems, 0.0f);

      LRT_TENSOR_ASSIGN_OR_RETURN(
          auto suf_k, prefill_inst->runner->ReadOutputAs<float>(
                          prefill_inst->outputs.key_caches[i]));
      LRT_TENSOR_ASSIGN_OR_RETURN(
          auto suf_v, prefill_inst->runner->ReadOutputAs<float>(
                          prefill_inst->outputs.value_caches[i]));

      const size_t pref_h_elems = static_cast<size_t>(prefix_len) * head_dim;
      const size_t suf_h_elems = static_cast<size_t>(suffix_len) * head_dim;
      const size_t suf_src_h_elems =
          static_cast<size_t>(prefill_inst->bucket_seq_len) * head_dim;
      for (int h = 0; h < config_.num_kv_heads; ++h) {
        const size_t dst_h_offset =
            static_cast<size_t>(h) * seq_len * head_dim;
        if (use_cache && pref_h_elems > 0) {
          std::copy_n(cached_prefix_keys_[i].data() + h * pref_h_elems,
                      pref_h_elems, host_key_caches[i].data() + dst_h_offset);
          std::copy_n(cached_prefix_values_[i].data() + h * pref_h_elems,
                      pref_h_elems, host_value_caches[i].data() + dst_h_offset);
        }
        std::copy_n(suf_k.data() + h * suf_src_h_elems, suf_h_elems,
                    host_key_caches[i].data() + dst_h_offset + pref_h_elems);
        std::copy_n(suf_v.data() + h * suf_src_h_elems, suf_h_elems,
                    host_value_caches[i].data() + dst_h_offset + pref_h_elems);
      }

      std::array<int32_t, 4> shape = {batch_size, config_.num_kv_heads,
                                      seq_len, head_dim};
      const size_t cur_elems =
          static_cast<size_t>(config_.num_kv_heads) * seq_len * head_dim;
      LRT_TENSOR_RETURN_IF_ERROR(
          decode_runner_->ReshapeInput(decode_inputs_.key_caches[i], shape));
      LRT_TENSOR_RETURN_IF_ERROR(
          decode_runner_->ReshapeInput(decode_inputs_.value_caches[i], shape));
      absl::Span<const float> key_span =
          absl::MakeConstSpan(host_key_caches[i].data(), cur_elems);
      absl::Span<const float> val_span =
          absl::MakeConstSpan(host_value_caches[i].data(), cur_elems);
      LRT_TENSOR_RETURN_IF_ERROR(
          decode_runner_->SetInput(decode_inputs_.key_caches[i], key_span));
      LRT_TENSOR_RETURN_IF_ERROR(
          decode_runner_->SetInput(decode_inputs_.value_caches[i], val_span));
    }

    int cache_len = seq_len;
    std::vector<float> global_cos(config_.global_key_size);
    std::vector<float> global_sin(config_.global_key_size);
    std::vector<float> local_cos(config_.head_dim);
    std::vector<float> local_sin(config_.head_dim);

    for (int step = 1; step < max_tokens; ++step) {
      const int32_t seq_k = cache_len + 1;
      std::vector<float> sliding_mask(static_cast<size_t>(seq_k), 0.0f);
      std::vector<float> global_mask(static_cast<size_t>(seq_k), 0.0f);

      LRT_TENSOR_ASSIGN_OR_RETURN(
          auto token_embeddings,
          loaded_tensors_.token_embedding->Lookup(current_token));
      LRT_TENSOR_ASSIGN_OR_RETURN(
          auto token_per_layer_embs,
          loaded_tensors_.emb_per_layer_table->LookupPerLayer(
              current_token, config_.num_layers, config_.per_layer_input_dim));

      RopeCosSin(/*start=*/cache_len, /*seq_len=*/1, config_.global_key_size,
                 config_.global_base_frequency, config_.global_rope_proportion,
                 absl::MakeSpan(global_cos), absl::MakeSpan(global_sin));
      RopeCosSin(/*start=*/cache_len, /*seq_len=*/1, config_.head_dim,
                 config_.local_base_frequency, config_.local_rope_proportion,
                 absl::MakeSpan(local_cos), absl::MakeSpan(local_sin));

      if (config_.sliding_window_size > 0) {
        const int32_t min_allowed_pos =
            std::max<int32_t>(0, seq_k - config_.sliding_window_size);
        const float neg_inf = std::numeric_limits<float>::lowest();
        for (int32_t j = 0; j < min_allowed_pos; ++j) {
          sliding_mask[static_cast<size_t>(j)] = neg_inf;
        }
      }
      std::array<int32_t, 4> mask_shape = {1, 1, 1, seq_k};
      LRT_TENSOR_RETURN_IF_ERROR(decode_runner_->ReshapeInput(
          decode_inputs_.sliding_attention_mask, mask_shape));
      LRT_TENSOR_RETURN_IF_ERROR(decode_runner_->ReshapeInput(
          decode_inputs_.global_attention_mask, mask_shape));

      absl::Span<const float> token_embeddings_span = absl::MakeConstSpan(
          token_embeddings.data(), token_embeddings.size());
      LRT_TENSOR_RETURN_IF_ERROR(decode_runner_->SetInput(
          decode_inputs_.embedded_input, token_embeddings_span));
      LRT_TENSOR_RETURN_IF_ERROR(
          decode_runner_->SetInput(decode_inputs_.rope_global_cos, global_cos));
      LRT_TENSOR_RETURN_IF_ERROR(
          decode_runner_->SetInput(decode_inputs_.rope_global_sin, global_sin));
      LRT_TENSOR_RETURN_IF_ERROR(
          decode_runner_->SetInput(decode_inputs_.rope_local_cos, local_cos));
      LRT_TENSOR_RETURN_IF_ERROR(
          decode_runner_->SetInput(decode_inputs_.rope_local_sin, local_sin));
      for (int l = 0; l < config_.num_layers; ++l) {
        absl::Span<const float> ple_span = absl::MakeConstSpan(
            token_per_layer_embs[l].data(), token_per_layer_embs[l].size());
        LRT_TENSOR_RETURN_IF_ERROR(decode_runner_->SetInput(
            decode_inputs_.per_layer_token_embeddings[l], ple_span));
      }
      LRT_TENSOR_RETURN_IF_ERROR(decode_runner_->SetInput(
          decode_inputs_.sliding_attention_mask, sliding_mask));
      LRT_TENSOR_RETURN_IF_ERROR(decode_runner_->SetInput(
          decode_inputs_.global_attention_mask, global_mask));

      LRT_TENSOR_RETURN_IF_ERROR(decode_runner_->Run());
      LRT_TENSOR_ASSIGN_OR_RETURN(
          auto logits_locked,
          decode_runner_->ReadOutputAs<float>(decode_outputs_.logits));
      current_token = SampleToken(logits_locked, req.sampling, rng_ptr);
      const int64_t ts = NowMonotonicNs();

      if (!req.ignore_eos && is_stop(current_token)) {
        result.stopped_on_eos = true;
        break;
      }

      result.gen_ids.push_back(current_token);
      result.token_ns.push_back(ts);
      if (static_cast<int>(result.gen_ids.size()) >= max_tokens) {
        break;
      }

      LRT_TENSOR_RETURN_IF_ERROR(UpdateKvCache(
          *decode_runner_, decode_inputs_, decode_outputs_, config_, cache_len,
          batch_size, sharing_patterns, host_key_caches, host_value_caches,
          decode_timing));
      cache_len += 1;
    }
  }

  result.t_end_ns = NowMonotonicNs();
  result.t_first_ns = result.token_ns.empty() ? 0 : result.token_ns.front();
  result.n_gen = static_cast<int>(result.gen_ids.size());
  result.prefill_ms = (result.t_prefill_done_ns - result.t_req_ns) / 1e6;
  result.decode_ms = (result.t_end_ns - result.t_prefill_done_ns) / 1e6;

  if (req.return_text && tokenizer_.has_value()) {
    result.text = tokenizer_->Decode(result.gen_ids);
  }

  return result;
}

std::string AebGemma4Session::HandleJsonLine(absl::string_view line,
                                             bool* should_quit) {
  if (should_quit != nullptr) {
    *should_quit = false;
  }
  std::string line_str(line);
  const char* cursor = line_str.c_str();
  minijson::value root_val;
  minijson::error err = minijson::parse(cursor, root_val);
  if (err != minijson::no_error) {
    return MakeErrorResponse(std::nullopt, std::nullopt,
                             absl::StrCat("bad json: ", minijson::errstr(err)));
  }
  const minijson::object* obj = root_val.as<minijson::object>();
  if (obj == nullptr) {
    return MakeErrorResponse(std::nullopt, std::nullopt,
                             "request must be a JSON object");
  }

  std::optional<int64_t> id_int;
  std::optional<std::string> id_str;
  minijson::value id_val;
  if (obj->at("id", &id_val)) {
    if (const double* d = id_val.as<double>(); d != nullptr) {
      id_int = static_cast<int64_t>(*d);
    } else if (const std::string* s = id_val.as<std::string>(); s != nullptr) {
      id_str = *s;
    }
  }

  minijson::value op_val;
  if (!obj->at("op", &op_val) || op_val.as<std::string>() == nullptr) {
    return MakeErrorResponse(id_int, id_str, "missing string field 'op'");
  }
  const std::string& op = *op_val.as<std::string>();

  if (op == "quit") {
    if (should_quit != nullptr) {
      *should_quit = true;
    }
    return absl::StrCat("{\"id\":", FormatRequestId(id_int, id_str),
                        ",\"ok\":true}");
  }

  if (op == "tokenize") {
    minijson::value text_val;
    if (!obj->at("text", &text_val) || text_val.as<std::string>() == nullptr) {
      return MakeErrorResponse(id_int, id_str, "missing 'text' for tokenize");
    }
    bool add_bos = true;
    minijson::value add_bos_val;
    if (obj->at("add_bos", &add_bos_val) &&
        add_bos_val.as<bool>() != nullptr) {
      add_bos = *add_bos_val.as<bool>();
    }
    auto ids_or = Tokenize(*text_val.as<std::string>(), add_bos);
    if (!ids_or.ok()) {
      return MakeErrorResponse(id_int, id_str, ids_or.status().message());
    }
    std::ostringstream oss;
    oss << "{\"id\":" << FormatRequestId(id_int, id_str)
        << ",\"ok\":true,\"ids\":[";
    for (size_t i = 0; i < ids_or->size(); ++i) {
      if (i > 0) oss << ",";
      oss << (*ids_or)[i];
    }
    oss << "]}";
    return oss.str();
  }

  if (op == "generate") {
    GenerateRequest req;
    minijson::value v;
    if (obj->at("prompt", &v) && v.as<std::string>() != nullptr) {
      req.prompt = *v.as<std::string>();
    }
    if (obj->at("prompt_ids", &v) && v.as<minijson::array>() != nullptr) {
      for (const minijson::value& elem : *v.as<minijson::array>()) {
        if (const double* d = elem.as<double>(); d != nullptr) {
          req.prompt_ids.push_back(static_cast<int32_t>(*d));
        }
      }
    }
    if (obj->at("max_tokens", &v) && v.as<double>() != nullptr) {
      req.max_tokens = static_cast<int>(*v.as<double>());
    }
    if (obj->at("ignore_eos", &v) && v.as<bool>() != nullptr) {
      req.ignore_eos = *v.as<bool>();
    }
    if (obj->at("return_text", &v) && v.as<bool>() != nullptr) {
      req.return_text = *v.as<bool>();
    }
    if (obj->at("return_prompt_ids", &v) && v.as<bool>() != nullptr) {
      req.return_prompt_ids = *v.as<bool>();
    }
    if (obj->at("cache_prefix", &v) && v.as<bool>() != nullptr) {
      req.cache_prefix = *v.as<bool>();
    }
    if (obj->at("prefix_tokens", &v) && v.as<double>() != nullptr) {
      req.prefix_tokens = static_cast<int>(*v.as<double>());
    }
    if (obj->at("stop_ids", &v) && v.as<minijson::array>() != nullptr) {
      for (const minijson::value& elem : *v.as<minijson::array>()) {
        if (const double* d = elem.as<double>(); d != nullptr) {
          req.stop_ids.insert(static_cast<int32_t>(*d));
        }
      }
    }
    if (obj->at("sampling", &v) && v.as<minijson::object>() != nullptr) {
      const minijson::object* smpl = v.as<minijson::object>();
      minijson::value sv;
      if (smpl->at("temperature", &sv) && sv.as<double>() != nullptr) {
        req.sampling.temperature = static_cast<float>(*sv.as<double>());
      }
      if (smpl->at("top_k", &sv) && sv.as<double>() != nullptr) {
        req.sampling.top_k = static_cast<int>(*sv.as<double>());
      }
      if (smpl->at("top_p", &sv) && sv.as<double>() != nullptr) {
        req.sampling.top_p = static_cast<float>(*sv.as<double>());
      }
      if (smpl->at("seed", &sv) && sv.as<double>() != nullptr) {
        req.sampling.seed = static_cast<uint32_t>(*sv.as<double>());
      }
    }

    auto res_or = Generate(req);
    if (!res_or.ok()) {
      return MakeErrorResponse(id_int, id_str, res_or.status().message());
    }
    const GenerateResult& r = *res_or;
    std::ostringstream oss;
    oss << "{\"id\":" << FormatRequestId(id_int, id_str)
        << ",\"ok\":true"
        << ",\"n_prompt\":" << r.n_prompt
        << ",\"n_cached\":" << r.n_cached
        << ",\"n_prefilled\":" << r.n_prefilled
        << ",\"n_gen\":" << r.n_gen
        << ",\"stopped_on_eos\":" << (r.stopped_on_eos ? "true" : "false")
        << ",\"t_req_ns\":" << r.t_req_ns
        << ",\"t_prefill_done_ns\":" << r.t_prefill_done_ns
        << ",\"t_first_ns\":" << r.t_first_ns
        << ",\"t_end_ns\":" << r.t_end_ns
        << ",\"token_ns\":[";
    for (size_t i = 0; i < r.token_ns.size(); ++i) {
      if (i > 0) oss << ",";
      oss << r.token_ns[i];
    }
    oss << "],\"gen_ids\":[";
    for (size_t i = 0; i < r.gen_ids.size(); ++i) {
      if (i > 0) oss << ",";
      oss << r.gen_ids[i];
    }
    oss << "]";
    if (req.return_prompt_ids) {
      oss << ",\"prompt_ids\":[";
      for (size_t i = 0; i < r.prompt_ids.size(); ++i) {
        if (i > 0) oss << ",";
        oss << r.prompt_ids[i];
      }
      oss << "]";
    }
    if (req.return_text) {
      oss << ",\"text\":" << EscapeJsonString(r.text);
    }
    oss << ",\"native\":{\"prefill_ms\":" << r.prefill_ms
        << ",\"decode_ms\":" << r.decode_ms << "}}";
    return oss.str();
  }

  return MakeErrorResponse(id_int, id_str,
                           absl::StrCat("unknown op: ", op));
}

absl::Status AebGemma4Session::Serve(std::istream& in, std::ostream& out) {
  out << "{\"event\":\"ready\",\"framework\":\"litert-tensor\",\"load_ns\":"
      << load_ns_ << ",\"info\":{\"backend\":"
      << EscapeJsonString(options_.backend_name)
      << ",\"dtype\":" << EscapeJsonString(options_.dtype)
      << ",\"model_variant\":" << EscapeJsonString(options_.model_variant)
      << ",\"threads\":" << options_.num_threads
      << ",\"ctx\":" << options_.ctx
      << ",\"cache_dir\":" << EscapeJsonString(options_.cache_dir)
      << ",\"cache_prefix\":" << (options_.cache_prefix ? "true" : "false")
      << ",\"has_sp_tokenizer\":"
      << (tokenizer_.has_value() ? "true" : "false") << "}}\n"
      << std::flush;

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    bool should_quit = false;
    std::string resp = HandleJsonLine(line, &should_quit);
    out << resp << "\n" << std::flush;
    if (should_quit) {
      break;
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::tensor::examples::gemma4
