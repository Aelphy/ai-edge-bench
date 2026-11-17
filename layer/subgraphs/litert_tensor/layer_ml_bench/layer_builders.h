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

#ifndef TENSOR_BENCHMARKS_LAYER_LAYER_BUILDERS_H_
#define TENSOR_BENCHMARKS_LAYER_LAYER_BUILDERS_H_

#include <cstddef>
#include <vector>

#include "absl/status/statusor.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/ynnpack/arithmetic.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_spec.h"
#include "tensor/tensor.h"

namespace layer_ml_bench::litert {

using BackendTensor =
    ::litert::tensor::Tensor<::litert::tensor::XnnpackMixinTag,
                             ::litert::tensor::YnnpackMixinTag>;

// Represents a non-owning binding between a dynamic graph input tensor and its
// backing host buffer. The referenced `SeededLayerData` must outlive the
// returned `LitertLayerGraph` and any runner compiled from it. When
// `num_elements > 0`, binds the leading prefix of `host_buffer` (e.g., S - 1
// prior KV cache tokens before appending token S).
struct ExternalInputBinding {
  BackendTensor tensor;
  const std::vector<float>* host_buffer = nullptr;
  size_t num_elements = 0;
};

// Encapsulates a compiled LiteRT Tensor API operator graph ready for execution
// by XnnpackRunner or YnnpackRunner. Holds non-owning `SpanCpuBuffer` views
// and `host_buffer` pointers into the `SeededLayerData` passed to `Build*`;
// callers must keep `data` alive and unmutated for the lifetime of the graph
// and any runner built from it.
struct LitertLayerGraph {
  BackendTensor output;
  std::vector<ExternalInputBinding> external_inputs;
};

// The builders below emit single-precision graphs only: each returns
// Unimplemented unless `spec.dtype` is `DataType::kFP32`.

// Builds an isolated RMSNorm layer graph.
absl::StatusOr<LitertLayerGraph> BuildRmsNorm(const LayerSpec& spec,
                                              const SeededLayerData& data);

// Builds an isolated Rotary Position Embedding (RoPE) layer graph.
absl::StatusOr<LitertLayerGraph> BuildRoPE(const LayerSpec& spec,
                                           const SeededLayerData& data);

// Builds an isolated Gated MLP (SwiGLU or GeGLU) feed-forward layer graph.
absl::StatusOr<LitertLayerGraph> BuildGatedMlp(const LayerSpec& spec,
                                               const SeededLayerData& data);

// Builds a Standard Attention layer graph (supports prefill and decode modes).
absl::StatusOr<LitertLayerGraph> BuildStandardAttention(
    const LayerSpec& spec, const SeededLayerData& data);

// Builds a Decode-Optimized Attention layer graph (strictly T=1).
absl::StatusOr<LitertLayerGraph> BuildDecodeOptimizedAttention(
    const LayerSpec& spec, const SeededLayerData& data);

// Builds a full composite Transformer Layer block (Attention + MLP + RMSNorms +
// Residual connections). Prefill attends causally over the block's own T
// tokens and returns InvalidArgument unless seq_len_q == seq_len_kv; decode
// (T=1) appends one token to a prior KV cache of S - 1 tokens.
absl::StatusOr<LitertLayerGraph> BuildTransformerBlock(
    const LayerSpec& spec, const SeededLayerData& data);

// Dispatcher building the appropriate layer graph corresponding to
// spec.layer_type.
absl::StatusOr<LitertLayerGraph> BuildLayer(const LayerSpec& spec,
                                            const SeededLayerData& data);

}  // namespace layer_ml_bench::litert

#endif  // TENSOR_BENCHMARKS_LAYER_LAYER_BUILDERS_H_
