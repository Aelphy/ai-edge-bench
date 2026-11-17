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

#include "tensor/benchmarks/layer/layer_builders.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/ynnpack/arithmetic.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_spec.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/ops/transformer/transformer_block.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/examples/ops/transformer/transformer_ops_graph.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"
#include "tensor/examples/ops/transformer/transformer_ops_ynnpack.h"
#include "tensor/tensor.h"

namespace layer_ml_bench::litert {
namespace {

using ::litert::tensor::FFNActivation;
using ::litert::tensor::KvCache;
using ::litert::tensor::OwningCpuBuffer;
using ::litert::tensor::RopeCosSinParams;
using ::litert::tensor::SpanCpuBuffer;
using ::litert::tensor::TransformerBlockWeights;
using ::litert::tensor::Type;
using ::litert::tensor::XnnpackMixinTag;
using ::litert::tensor::YnnpackMixinTag;

// Returns OK if `spec` asks for single precision, the only precision the
// builders emit, and Unimplemented otherwise. A layer of any other precision
// would be measured under a label it does not execute.
absl::Status RequireFp32(const LayerSpec& spec) {
  if (spec.dtype == DataType::kFP32) {
    return absl::OkStatus();
  }
  return absl::UnimplementedError(
      absl::StrCat("Layer graphs execute in FP32 only; ",
                   DataTypeToString(spec.dtype), " is not supported."));
}

// Creates a dynamic external input tensor (.buffer is null so
// BuildXnnpackGraph assigns XNN_VALUE_FLAG_EXTERNAL_INPUT).
BackendTensor MakeInputTensor(const std::vector<int>& shape) {
  return BackendTensor({
      .type = Type::kFP32,
      .shape = shape,
  });
}

// Creates a static weight/constant tensor (.buffer is non-null so
// XNNPACK/YNNPACK packs constant weights once during runtime creation).
BackendTensor MakeWeightTensor(const std::vector<int>& shape,
                               const std::vector<float>& buffer) {
  size_t expected_elems = 1;
  for (int dim : shape) {
    if (dim <= 0) {
      return BackendTensor(
          ::litert::tensor::TensorHandle(absl::InvalidArgumentError(
              "MakeWeightTensor requires positive shape dimensions.")));
    }
    expected_elems *= static_cast<size_t>(dim);
  }
  if (buffer.size() != expected_elems) {
    return BackendTensor(::litert::tensor::TensorHandle(
        absl::InvalidArgumentError("MakeWeightTensor buffer size mismatch.")));
  }
  return BackendTensor({
      .type = Type::kFP32,
      .shape = shape,
      .buffer = std::make_shared<SpanCpuBuffer>(buffer),
  });
}

BackendTensor MakeScalarTensor(float value) {
  return BackendTensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = value,
  });
}

}  // namespace

absl::StatusOr<LitertLayerGraph> BuildRmsNorm(const LayerSpec& spec,
                                              const SeededLayerData& data) {
  absl::Status dtype_status = RequireFp32(spec);
  if (!dtype_status.ok()) {
    return dtype_status;
  }
  const int b = static_cast<int>(spec.batch_size);
  const int t = static_cast<int>(spec.EffectiveSeqLenQ());
  const int m = static_cast<int>(spec.ModelDim());

  BackendTensor input = MakeInputTensor({b, t, m});
  BackendTensor scale = MakeWeightTensor({m}, data.rms_scale);
  BackendTensor eps_tensor = MakeScalarTensor(spec.eps);

  BackendTensor output = ::litert::tensor::RmsNorm(input, scale, eps_tensor);
  if (!output.GetStatus().ok()) {
    return output.GetStatus();
  }
  return LitertLayerGraph{output, {{input, &data.input}}};
}

absl::StatusOr<LitertLayerGraph> BuildRoPE(const LayerSpec& spec,
                                           const SeededLayerData& data) {
  absl::Status dtype_status = RequireFp32(spec);
  if (!dtype_status.ok()) {
    return dtype_status;
  }
  const int b = static_cast<int>(spec.batch_size);
  const int h = static_cast<int>(spec.num_heads);
  const int t = static_cast<int>(spec.EffectiveSeqLenQ());
  const int d = static_cast<int>(spec.head_dim);
  const int q_dim = static_cast<int>(spec.QDim());

  BackendTensor x = MakeInputTensor({b, h, t, d});
  BackendTensor cos = MakeWeightTensor({1, 1, t, d}, data.rope_cos);
  BackendTensor sin = MakeWeightTensor({1, 1, t, d}, data.rope_sin);

  BackendTensor roped = ::litert::tensor::RoPE(x, cos, sin);
  if (!roped.GetStatus().ok()) {
    return roped.GetStatus();
  }
  BackendTensor output = ::litert::tensor::Reshape(roped, {b, t, q_dim});
  if (!output.GetStatus().ok()) {
    return output.GetStatus();
  }
  return LitertLayerGraph{output, {{x, &data.input}}};
}

absl::StatusOr<LitertLayerGraph> BuildGatedMlp(const LayerSpec& spec,
                                               const SeededLayerData& data) {
  absl::Status dtype_status = RequireFp32(spec);
  if (!dtype_status.ok()) {
    return dtype_status;
  }
  const int b = static_cast<int>(spec.batch_size);
  const int t = static_cast<int>(spec.EffectiveSeqLenQ());
  const int m = static_cast<int>(spec.ModelDim());
  const int f = static_cast<int>(spec.hidden_dim);

  BackendTensor input = MakeInputTensor({b, t, m});
  BackendTensor w_gate = MakeWeightTensor({f, m}, data.w_gate);
  BackendTensor w_up = MakeWeightTensor({f, m}, data.w_up);
  BackendTensor w_down = MakeWeightTensor({m, f}, data.w_down);

  ::litert::tensor::FFNActivation act =
      (spec.mlp_activation == MlpActivation::kSwiGLU)
          ? ::litert::tensor::FFNActivation::kSwish
          : ::litert::tensor::FFNActivation::kGeluApproximate;

  BackendTensor output =
      ::litert::tensor::FeedForward(input, w_gate, w_up, w_down, act);
  if (!output.GetStatus().ok()) {
    return output.GetStatus();
  }
  return LitertLayerGraph{output, {{input, &data.input}}};
}

absl::StatusOr<LitertLayerGraph> BuildStandardAttention(
    const LayerSpec& spec, const SeededLayerData& data) {
  absl::Status dtype_status = RequireFp32(spec);
  if (!dtype_status.ok()) {
    return dtype_status;
  }
  const int b = static_cast<int>(spec.batch_size);
  const int h = static_cast<int>(spec.num_heads);
  const int t = static_cast<int>(spec.EffectiveSeqLenQ());
  const int d = static_cast<int>(spec.head_dim);
  const int s = static_cast<int>(spec.EffectiveSeqLenKV());
  const int n_kv = static_cast<int>(spec.num_kv_heads);

  BackendTensor q = MakeInputTensor({b, h, t, d});
  BackendTensor k = MakeInputTensor({b, n_kv, s, d});
  BackendTensor v = MakeInputTensor({b, n_kv, s, d});

  const size_t kv_elems = static_cast<size_t>(b) * n_kv * s * d;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  BackendTensor attn_out = ::litert::tensor::ScaledDotProductAttention(
      q, k, v, BackendTensor(::litert::tensor::TensorHandle::Invalid()), scale);
  if (!attn_out.GetStatus().ok()) {
    return attn_out.GetStatus();
  }
  return LitertLayerGraph{
      attn_out, {{q, &data.q}, {k, &data.k, kv_elems}, {v, &data.v, kv_elems}}};
}

absl::StatusOr<LitertLayerGraph> BuildDecodeOptimizedAttention(
    const LayerSpec& spec, const SeededLayerData& data) {
  absl::Status dtype_status = RequireFp32(spec);
  if (!dtype_status.ok()) {
    return dtype_status;
  }
  if (spec.attention_mode != AttentionMode::kDecode) {
    return absl::InvalidArgumentError(
        "BuildDecodeOptimizedAttention requires AttentionMode::kDecode.");
  }
  const int b = static_cast<int>(spec.batch_size);
  const int h = static_cast<int>(spec.num_heads);
  const int d = static_cast<int>(spec.head_dim);
  const int s = static_cast<int>(spec.EffectiveSeqLenKV());
  const int n_kv = static_cast<int>(spec.num_kv_heads);

  BackendTensor q = MakeInputTensor({b, h, 1, d});
  BackendTensor k = MakeInputTensor({b, n_kv, s, d});
  BackendTensor v = MakeInputTensor({b, n_kv, s, d});

  BackendTensor q_in = q;
  BackendTensor k_in = k;
  BackendTensor v_in = v;
  const size_t kv_elems = static_cast<size_t>(b) * n_kv * s * d;

  if (n_kv < h) {
    const int num_groups = h / n_kv;
    k = ::litert::tensor::RepeatKVHeads(k, num_groups);
    v = ::litert::tensor::RepeatKVHeads(v, num_groups);
  }

  // scores_t = BatchMatMul(k, q, false, true)
  // [B, H, S, D] x [B, H, 1, D]^T -> [B, H, S, 1]
  BackendTensor scores_t =
      ::litert::tensor::BatchMatMul(k, q, /*adj_x=*/false, /*adj_y=*/true);
  if (!scores_t.GetStatus().ok()) {
    return scores_t.GetStatus();
  }

  // Reshape to [B, H, 1, S]
  BackendTensor scores = ::litert::tensor::Reshape(scores_t, {b, h, 1, s});
  if (!scores.GetStatus().ok()) {
    return scores.GetStatus();
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  BackendTensor scale_tensor = MakeScalarTensor(scale);
  BackendTensor scaled_scores = ::litert::tensor::Mul(scores, scale_tensor);
  if (!scaled_scores.GetStatus().ok()) {
    return scaled_scores.GetStatus();
  }

  BackendTensor probs = ::litert::tensor::Softmax(scaled_scores);
  if (!probs.GetStatus().ok()) {
    return probs.GetStatus();
  }

  // out = BatchMatMul(probs, v, false, false)
  // [B, H, 1, S] x [B, H, S, D] -> [B, H, 1, D]
  BackendTensor out =
      ::litert::tensor::BatchMatMul(probs, v, /*adj_x=*/false, /*adj_y=*/false);
  if (!out.GetStatus().ok()) {
    return out.GetStatus();
  }
  return LitertLayerGraph{
      out,
      {{q_in, &data.q}, {k_in, &data.k, kv_elems}, {v_in, &data.v, kv_elems}}};
}

absl::StatusOr<LitertLayerGraph> BuildTransformerBlock(
    const LayerSpec& spec, const SeededLayerData& data) {
  absl::Status dtype_status = RequireFp32(spec);
  if (!dtype_status.ok()) {
    return dtype_status;
  }
  const int b = static_cast<int>(spec.batch_size);
  const int t = static_cast<int>(spec.EffectiveSeqLenQ());
  const int h = static_cast<int>(spec.num_heads);
  const int d = static_cast<int>(spec.head_dim);
  const int m = static_cast<int>(spec.ModelDim());
  const int q_dim = static_cast<int>(spec.QDim());
  const int kv_dim = static_cast<int>(spec.KVDim());
  const int n_kv = static_cast<int>(spec.num_kv_heads);
  const int f = static_cast<int>(spec.hidden_dim);
  const int s = static_cast<int>(spec.EffectiveSeqLenKV());

  BackendTensor x = MakeInputTensor({b, t, m});

  TransformerBlockWeights<XnnpackMixinTag, YnnpackMixinTag> weights{
      .pre_attn_norm_weight = MakeWeightTensor({m}, data.rms_scale),
      .q_weight = MakeWeightTensor({q_dim, m}, data.w_q),
      .k_weight = MakeWeightTensor({kv_dim, m}, data.w_k),
      .v_weight = MakeWeightTensor({kv_dim, m}, data.w_v),
      .o_weight = MakeWeightTensor({m, q_dim}, data.w_o),
      .pre_ffn_norm_weight = MakeWeightTensor({m}, data.ffn_rms_scale),
      .ffn_gate_weight = MakeWeightTensor({f, m}, data.w_gate),
      .ffn_up_weight = MakeWeightTensor({f, m}, data.w_up),
      .ffn_down_weight = MakeWeightTensor({m, f}, data.w_down),
  };
  if (spec.qk_norm) {
    weights.q_norm_weight = MakeWeightTensor({d}, data.q_norm_scale);
    weights.k_norm_weight = MakeWeightTensor({d}, data.k_norm_scale);
  }
  if (spec.post_norms) {
    weights.post_attn_norm_weight =
        MakeWeightTensor({m}, data.post_attn_rms_scale);
    weights.post_ffn_norm_weight =
        MakeWeightTensor({m}, data.post_ffn_rms_scale);
  }

  RopeCosSinParams<XnnpackMixinTag, YnnpackMixinTag> rope{
      .cos = MakeWeightTensor({1, 1, t, d}, data.rope_cos),
      .sin = MakeWeightTensor({1, 1, t, d}, data.rope_sin),
  };

  std::optional<KvCache<XnnpackMixinTag, YnnpackMixinTag>> kv_cache;
  BackendTensor attention_mask(::litert::tensor::TensorHandle::Invalid());
  std::vector<ExternalInputBinding> external_inputs = {{x, &data.input}};

  if (spec.attention_mode == AttentionMode::kPrefill) {
    // Without a prior KV cache the block attends over its own T tokens, so the
    // scores are [B, H, T, T] and a mask spanning a longer S cannot apply.
    if (t != s) {
      return absl::InvalidArgumentError(
          "BuildTransformerBlock prefill requires seq_len_q == seq_len_kv.");
    }
    if (t > 1 || spec.sliding_window > 0) {
      std::vector<float> mask_vals(static_cast<size_t>(t) * t, 0.0f);
      const int win = static_cast<int>(spec.sliding_window);
      for (int i = 0; i < t; ++i) {
        for (int j = 0; j < t; ++j) {
          const bool causal_masked = (j > i);
          const bool window_masked = (win > 0) && ((i - j) >= win);
          if (causal_masked || window_masked) {
            mask_vals[static_cast<size_t>(i) * t + j] = -1e9f;
          }
        }
      }
      attention_mask = BackendTensor({
          .type = Type::kFP32,
          .shape = {1, 1, t, t},
          .buffer = OwningCpuBuffer::Copy<Type::kFP32>(mask_vals),
      });
    }
  } else if (s > 1) {
    // Decode mode (T=1) over prior KV cache of S - 1 tokens; TransformerBlock
    // projects token S, applies QK-norm + RoPE, and concatenates along axis 2.
    const int prior_s = s - 1;
    const size_t prior_elems = static_cast<size_t>(b) * n_kv * prior_s * d;
    BackendTensor k_cache = MakeInputTensor({b, n_kv, prior_s, d});
    BackendTensor v_cache = MakeInputTensor({b, n_kv, prior_s, d});
    kv_cache = KvCache<XnnpackMixinTag, YnnpackMixinTag>{
        .k = k_cache,
        .v = v_cache,
    };
    external_inputs.push_back({k_cache, &data.k, prior_elems});
    external_inputs.push_back({v_cache, &data.v, prior_elems});
  }

  const FFNActivation act = (spec.mlp_activation == MlpActivation::kSwiGLU)
                                ? FFNActivation::kSwish
                                : FFNActivation::kGeluApproximate;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));

  absl::StatusOr<::litert::tensor::TransformerBlockOutputs<XnnpackMixinTag,
                                                           YnnpackMixinTag>>
      block = ::litert::tensor::TransformerBlock(
          x, weights, rope, h, n_kv, d, std::move(kv_cache), attention_mask,
          spec.eps, act, scale);
  if (!block.ok()) {
    return block.status();
  }
  return LitertLayerGraph{std::move(block->output), std::move(external_inputs)};
}

absl::StatusOr<LitertLayerGraph> BuildLayer(const LayerSpec& spec,
                                            const SeededLayerData& data) {
  switch (spec.layer_type) {
    case LayerType::kRmsNorm:
      return BuildRmsNorm(spec, data);
    case LayerType::kRoPE:
      return BuildRoPE(spec, data);
    case LayerType::kGatedMlp:
      return BuildGatedMlp(spec, data);
    case LayerType::kStandardAttention:
      return BuildStandardAttention(spec, data);
    case LayerType::kDecodeOptimizedAttention:
      return BuildDecodeOptimizedAttention(spec, data);
    case LayerType::kTransformerBlock:
      return BuildTransformerBlock(spec, data);
  }
  return absl::InvalidArgumentError("Unsupported LayerType");
}

}  // namespace layer_ml_bench::litert
