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

#ifndef TENSOR_BENCHMARKS_LAYER_LAYER_SPEC_H_
#define TENSOR_BENCHMARKS_LAYER_LAYER_SPEC_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"

namespace layer_ml_bench {

// Supported benchmark precisions.
enum class DataType {
  kFP32,
  kFP16,
  kBF16,
  kQD8,
  kQB4W,
};

// Supported benchmark layer and composite subgraph types.
enum class LayerType {
  kRmsNorm,
  kRoPE,
  kGatedMlp,
  kStandardAttention,
  kDecodeOptimizedAttention,
  kTransformerBlock,
};

// MLP activation functions.
enum class MlpActivation {
  kSwiGLU,
  kGeGLU,
};

// Attention operational modes.
enum class AttentionMode {
  kPrefill,  // Query sequence length T = context length S
  kDecode,   // Query sequence length T = 1, context length S >= 1
};

// String conversion helpers.
std::string DataTypeToString(DataType dtype);
std::string LayerTypeToString(LayerType type);
std::string MlpActivationToString(MlpActivation act);
std::string AttentionModeToString(AttentionMode mode);

// Framework-agnostic declarative layer specification schema.
struct LayerSpec {
  LayerType layer_type = LayerType::kTransformerBlock;
  std::string model_name = "LLaMA3-8B";
  AttentionMode attention_mode = AttentionMode::kPrefill;
  MlpActivation mlp_activation = MlpActivation::kSwiGLU;

  // Tensor dimensions.
  int64_t batch_size = 1;      // B: Batch dimension
  int64_t seq_len_kv = 512;    // S: Context sequence length
  int64_t seq_len_q = 512;     // T: Query sequence length (1 for decode)
  int64_t num_heads = 32;      // H: Query attention heads
  int64_t head_dim = 128;      // D: Head dimension
  int64_t num_kv_heads = 8;    // N_kv: KV attention heads (Grouped-Query)
  int64_t hidden_dim = 14336;  // F: Intermediate MLP dimension
  int64_t embed_dim = 0;  // M: Model embedding dim (0 => num_heads * head_dim)
  int64_t sliding_window = 0;  // W: Local sliding window (0 => full causal)
  float eps = 1e-5f;
  bool qk_norm = false;     // Per-head Q/K RMSNorm (Gemma 3/4, Qwen 3/3.5)
  bool post_norms = false;  // Post-attn & post-FFN RMSNorms (Gemma 3/4)

  // Target precision.
  DataType dtype = DataType::kFP32;

  // Validates tensor dimensions and configuration invariants. Returns
  // InvalidArgument for non-positive dimensions, an odd head_dim, num_heads not
  // divisible by num_kv_heads, a non-positive or non-finite eps, an effective
  // query length exceeding seq_len_kv, kDecodeOptimizedAttention outside
  // kDecode, or a kTransformerBlock prefill whose seq_len_q differs from
  // seq_len_kv (the block derives K/V from its own T query tokens).
  absl::Status Validate() const;

  // Returns effective query sequence length T: exactly 1 in decode mode or
  // decode-optimized attention, otherwise seq_len_q.
  int64_t EffectiveSeqLenQ() const;

  // Returns effective KV horizon in decode mode (min(seq_len_kv,
  // sliding_window) when sliding_window > 0), or seq_len_kv in prefill mode.
  int64_t EffectiveSeqLenKV() const;

  // Returns model residual stream dimension M: embed_dim if explicitly set
  // (> 0), otherwise num_heads * head_dim.
  int64_t ModelDim() const {
    return embed_dim > 0 ? embed_dim : num_heads * head_dim;
  }

  // Returns total query projection dimension H * D.
  int64_t QDim() const { return num_heads * head_dim; }

  // Returns total key/value projection dimension N_kv * D.
  int64_t KVDim() const { return num_kv_heads * head_dim; }

  // Returns last dimension of the layer's flattened 3D output tensor: QDim()
  // for RoPE and standalone Attention layers, ModelDim() for RmsNorm, GatedMlp,
  // and TransformerBlock.
  int64_t OutputDim() const {
    if (layer_type == LayerType::kRoPE ||
        layer_type == LayerType::kStandardAttention ||
        layer_type == LayerType::kDecodeOptimizedAttention) {
      return QDim();
    }
    return ModelDim();
  }

  // Returns exact output tensor shape matching the graph's physical layout:
  // 4D `[B, H, T, D]` for standalone Attention layers (so
  // `OutputShape().back()` is `D` for per-head softmax KL checks), and 3D `[B,
  // T, OutputDim()]` for RmsNorm, RoPE, GatedMlp, and TransformerBlock.
  std::vector<int64_t> OutputShape() const;

  // Returns analytical floating-point operations for a single forward pass, or
  // 0 if layer_type is not a declared LayerType enumerator.
  int64_t TotalFlops() const;

  // Returns analytical memory bytes accessed for a single forward pass, or 0 if
  // layer_type is not a declared LayerType enumerator.
  int64_t TotalMemoryBytes() const;

  // Returns arithmetic intensity in FLOPs / byte for Roofline analysis.
  double ArithmeticIntensity() const;

  // Returns human- and agent-readable summary string.
  std::string ToString() const;
};

}  // namespace layer_ml_bench

#endif  // TENSOR_BENCHMARKS_LAYER_LAYER_SPEC_H_
