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

#include "tensor/benchmarks/layer/layer_spec.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"

namespace layer_ml_bench {

std::string DataTypeToString(DataType dtype) {
  switch (dtype) {
    case DataType::kFP32:
      return "FP32";
    case DataType::kFP16:
      return "FP16";
    case DataType::kBF16:
      return "BF16";
    case DataType::kQD8:
      return "QD8";
    case DataType::kQB4W:
      return "QB4W";
  }
  return "Unknown";
}

std::string LayerTypeToString(LayerType type) {
  switch (type) {
    case LayerType::kRmsNorm:
      return "RmsNorm";
    case LayerType::kRoPE:
      return "RoPE";
    case LayerType::kGatedMlp:
      return "GatedMlp";
    case LayerType::kStandardAttention:
      return "StandardAttention";
    case LayerType::kDecodeOptimizedAttention:
      return "DecodeOptimizedAttention";
    case LayerType::kTransformerBlock:
      return "TransformerBlock";
  }
  return "Unknown";
}

std::string MlpActivationToString(MlpActivation act) {
  switch (act) {
    case MlpActivation::kSwiGLU:
      return "SwiGLU";
    case MlpActivation::kGeGLU:
      return "GeGLU";
  }
  return "Unknown";
}

std::string AttentionModeToString(AttentionMode mode) {
  switch (mode) {
    case AttentionMode::kPrefill:
      return "Prefill";
    case AttentionMode::kDecode:
      return "Decode";
  }
  return "Unknown";
}

int64_t LayerSpec::EffectiveSeqLenQ() const {
  if (attention_mode == AttentionMode::kDecode ||
      layer_type == LayerType::kDecodeOptimizedAttention) {
    return 1;
  }
  return seq_len_q;
}

int64_t LayerSpec::EffectiveSeqLenKV() const {
  if ((attention_mode == AttentionMode::kDecode ||
       layer_type == LayerType::kDecodeOptimizedAttention) &&
      sliding_window > 0 && sliding_window < seq_len_kv) {
    return sliding_window;
  }
  return seq_len_kv;
}

absl::Status LayerSpec::Validate() const {
  if (batch_size <= 0) {
    return absl::InvalidArgumentError("batch_size must be positive");
  }
  if (seq_len_kv <= 0) {
    return absl::InvalidArgumentError("seq_len_kv must be positive");
  }
  if (seq_len_q <= 0) {
    return absl::InvalidArgumentError("seq_len_q must be positive");
  }
  if (num_heads <= 0) {
    return absl::InvalidArgumentError("num_heads must be positive");
  }
  if (head_dim <= 0) {
    return absl::InvalidArgumentError("head_dim must be positive");
  }
  if (num_kv_heads <= 0) {
    return absl::InvalidArgumentError("num_kv_heads must be positive");
  }
  if (hidden_dim <= 0) {
    return absl::InvalidArgumentError("hidden_dim must be positive");
  }
  if (embed_dim < 0) {
    return absl::InvalidArgumentError("embed_dim must be non-negative");
  }
  if (sliding_window < 0) {
    return absl::InvalidArgumentError("sliding_window must be non-negative");
  }
  if (num_heads % num_kv_heads != 0) {
    return absl::InvalidArgumentError(
        "num_heads must be divisible by num_kv_heads");
  }
  if (head_dim % 2 != 0) {
    return absl::InvalidArgumentError(
        "head_dim must be even (required for RoPE)");
  }
  if (!(eps > 0.0f) || !std::isfinite(eps)) {
    return absl::InvalidArgumentError("eps must be positive and finite");
  }
  if (EffectiveSeqLenQ() > seq_len_kv) {
    return absl::InvalidArgumentError(
        "Effective seq_len_q cannot exceed seq_len_kv");
  }
  if (layer_type == LayerType::kDecodeOptimizedAttention &&
      attention_mode != AttentionMode::kDecode) {
    return absl::InvalidArgumentError(
        "kDecodeOptimizedAttention requires attention_mode to be kDecode");
  }
  if (layer_type == LayerType::kTransformerBlock &&
      attention_mode == AttentionMode::kPrefill && seq_len_q != seq_len_kv) {
    return absl::InvalidArgumentError(
        "kTransformerBlock prefill requires seq_len_q == seq_len_kv");
  }
  return absl::OkStatus();
}

int64_t LayerSpec::TotalFlops() const {
  const int64_t b = batch_size;
  const int64_t s = EffectiveSeqLenKV();
  const int64_t t = EffectiveSeqLenQ();
  const int64_t h = num_heads;
  const int64_t d = head_dim;
  const int64_t m = ModelDim();
  const int64_t q_dim = QDim();
  const int64_t kv_dim = KVDim();
  const int64_t f = hidden_dim;

  switch (layer_type) {
    case LayerType::kRmsNorm:
      return 4 * b * t * m;
    case LayerType::kRoPE:
      return 3 * b * t * q_dim;
    case LayerType::kGatedMlp:
      return 6 * b * t * m * f + 3 * b * t * f;
    case LayerType::kStandardAttention:
      return 4 * b * h * t * s * d + 3 * b * h * t * s;
    case LayerType::kDecodeOptimizedAttention:
      return 4 * b * h * 1 * s * d + 3 * b * h * 1 * s;
    case LayerType::kTransformerBlock: {
      const int64_t pre_attn_norm = 4 * b * t * m;
      const int64_t qkv_proj = 2 * b * t * m * (q_dim + 2 * kv_dim);
      const int64_t qk_norms = qk_norm ? 4 * b * t * (q_dim + kv_dim) : 0;
      const int64_t rope_qk = 3 * b * t * (q_dim + kv_dim);
      const int64_t attn_core = 4 * b * h * t * s * d + 3 * b * h * t * s;
      const int64_t out_proj = 2 * b * t * q_dim * m;
      const int64_t post_norms_flops = post_norms ? 8 * b * t * m : 0;
      const int64_t res1 = b * t * m;
      const int64_t pre_ffn_norm = 4 * b * t * m;
      const int64_t gated_mlp = 6 * b * t * m * f + 3 * b * t * f;
      const int64_t res2 = b * t * m;
      return pre_attn_norm + qkv_proj + qk_norms + rope_qk + attn_core +
             out_proj + post_norms_flops + res1 + pre_ffn_norm + gated_mlp +
             res2;
    }
  }
  return 0;
}

std::vector<int64_t> LayerSpec::OutputShape() const {
  if (layer_type == LayerType::kStandardAttention ||
      layer_type == LayerType::kDecodeOptimizedAttention) {
    return {batch_size, num_heads, EffectiveSeqLenQ(), head_dim};
  }
  return {batch_size, EffectiveSeqLenQ(), OutputDim()};
}

int64_t LayerSpec::TotalMemoryBytes() const {
  const int64_t b = batch_size;
  const int64_t s = EffectiveSeqLenKV();
  const int64_t t = EffectiveSeqLenQ();
  const int64_t h = num_heads;
  const int64_t d = head_dim;
  const int64_t m = ModelDim();
  const int64_t q_dim = QDim();
  const int64_t kv_dim = KVDim();
  const int64_t n_kv = num_kv_heads;
  const int64_t f = hidden_dim;

  double act_bytes = 4.0;
  double weight_bytes = 4.0;
  switch (dtype) {
    case DataType::kFP32:
      act_bytes = 4.0;
      weight_bytes = 4.0;
      break;
    case DataType::kFP16:
    case DataType::kBF16:
      act_bytes = 2.0;
      weight_bytes = 2.0;
      break;
    case DataType::kQD8:
      act_bytes = 4.0;
      weight_bytes = 1.0;
      break;
    case DataType::kQB4W:
      act_bytes = 4.0;
      weight_bytes = 0.5;
      break;
  }

  switch (layer_type) {
    case LayerType::kRmsNorm:
      return static_cast<int64_t>(act_bytes * (2 * b * t * m) + 4.0 * m);
    case LayerType::kRoPE:
      return static_cast<int64_t>(act_bytes * (2 * b * t * q_dim) +
                                  4.0 * (2 * t * d));
    case LayerType::kGatedMlp:
      return static_cast<int64_t>(act_bytes * (2 * b * t * m) +
                                  weight_bytes * (3 * m * f));
    case LayerType::kStandardAttention:
      return static_cast<int64_t>(act_bytes *
                                  (2 * b * h * t * d + 2 * b * n_kv * s * d));
    case LayerType::kDecodeOptimizedAttention:
      return static_cast<int64_t>(act_bytes *
                                  (2 * b * h * 1 * d + 2 * b * n_kv * s * d));
    case LayerType::kTransformerBlock: {
      const int64_t in_out_elems = 2 * b * t * m;
      const int64_t kv_cache_elems = 2 * b * n_kv * s * d;
      const int64_t rope_table_elems = 2 * t * d;
      const int64_t mask_elems = (attention_mode == AttentionMode::kPrefill &&
                                  (t > 1 || sliding_window > 0))
                                     ? (t * s)
                                     : 0;
      const int64_t proj_mlp_weight_elems =
          m * (q_dim + 2 * kv_dim) + q_dim * m + 3 * m * f;
      const int64_t norm_scale_elems =
          (post_norms ? 4 * m : 2 * m) + (qk_norm ? 2 * d : 0);
      return static_cast<int64_t>(
          act_bytes * (in_out_elems + kv_cache_elems) +
          4.0 * (rope_table_elems + mask_elems + norm_scale_elems) +
          weight_bytes * proj_mlp_weight_elems);
    }
  }
  return 0;
}

double LayerSpec::ArithmeticIntensity() const {
  const int64_t bytes = TotalMemoryBytes();
  if (bytes <= 0) {
    return 0.0;
  }
  return static_cast<double>(TotalFlops()) / static_cast<double>(bytes);
}

std::string LayerSpec::ToString() const {
  return absl::StrFormat("%s_%s_%s_B%d_S%d_T%d_H%d_D%d_Nkv%d_F%d_%s",
                         model_name, LayerTypeToString(layer_type),
                         AttentionModeToString(attention_mode), batch_size,
                         seq_len_kv, EffectiveSeqLenQ(), num_heads, head_dim,
                         num_kv_heads, hidden_dim, DataTypeToString(dtype));
}

}  // namespace layer_ml_bench
