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

#ifndef TENSOR_BENCHMARKS_LAYER_LAYER_BACKEND_H_
#define TENSOR_BENCHMARKS_LAYER_LAYER_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "pthreadpool.h"
#include "tensor/benchmarks/layer/layer_spec.h"

namespace layer_ml_bench {

// Framework-agnostic contiguous host memory container for layer activations
// and weights.
struct SeededLayerData {
  std::vector<float> input;
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> rms_scale;
  std::vector<float> rope_cos;
  std::vector<float> rope_sin;
  std::vector<float> w_q;
  std::vector<float> w_k;
  std::vector<float> w_v;
  std::vector<float> w_o;
  std::vector<float> w_gate;
  std::vector<float> w_up;
  std::vector<float> w_down;
  std::vector<float> ffn_rms_scale;
  std::vector<float> q_norm_scale;
  std::vector<float> k_norm_scale;
  std::vector<float> post_attn_rms_scale;
  std::vector<float> post_ffn_rms_scale;
};

// Generates deterministic host float buffers from PRNG seed for the specified
// layer.
SeededLayerData CreateSeededLayerData(const LayerSpec& spec,
                                      uint32_t seed = 1234);

// Execution options passed to the backend during setup.
struct BackendOptions {
  size_t num_threads = 1;
  bool use_ynnpack = false;
};

// Abstract plugin interface for runtime backends (LiteRT Tensor API,
// ExecuTorch, Raw XNNPACK/YNNPACK C APIs, llama.cpp, etc.).
class LayerBackend {
 public:
  virtual ~LayerBackend() = default;

  // Human-readable backend name (e.g. "LitertTensorBackend",
  // "ExecutorchBackend").
  virtual std::string Name() const = 0;

  // Initializes the backend runtime, builds/compiles the operator graph, and
  // binds input/weight buffers from SeededLayerData.
  virtual absl::Status Setup(const LayerSpec& spec, const SeededLayerData& data,
                             const BackendOptions& options) = 0;

  // Rvalue overload allowing backends that persist host buffers to take
  // ownership without duplicating allocations during Setup().
  virtual absl::Status Setup(const LayerSpec& spec, SeededLayerData&& data,
                             const BackendOptions& options) {
    return Setup(spec, static_cast<const SeededLayerData&>(data), options);
  }

  // Executes a single forward pass of the prepared layer.
  virtual absl::Status Execute() = 0;

  // Returns the backend's underlying pthreadpool worker threadpool if
  // available, enabling multithreaded L2/LLC cache wiping. Returns nullptr if
  // single-threaded or if using a custom threadpool.
  virtual pthreadpool_t GetThreadpool() const { return nullptr; }

  // Returns flat contiguous FP32 output tensor data from the most recent
  // Execute().
  virtual absl::StatusOr<std::vector<float>> GetOutputData() const = 0;
};

}  // namespace layer_ml_bench

#endif  // TENSOR_BENCHMARKS_LAYER_LAYER_BACKEND_H_
