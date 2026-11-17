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

#ifndef TENSOR_BENCHMARKS_LAYER_LITERT_TENSOR_BACKEND_H_
#define TENSOR_BENCHMARKS_LAYER_LITERT_TENSOR_BACKEND_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "pthreadpool.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_builders.h"
#include "tensor/benchmarks/layer/layer_spec.h"
#include "tensor/runners/common_nnpack/runner.h"
#include "tensor/runners/xnnpack/runner.h"

namespace layer_ml_bench::litert {

// Selects the underlying LiteRT Tensor API execution engine.
enum class RuntimeBackend {
  kXnnpack,
  kYnnpack,
};

// Concrete runtime LayerBackend plugin for the LiteRT Tensor API and XNNPACK /
// YNNPACK execution engines.
class LitertTensorBackend : public LayerBackend {
 public:
  explicit LitertTensorBackend(
      RuntimeBackend runtime_backend = RuntimeBackend::kXnnpack)
      : runtime_backend_(runtime_backend) {}
  ~LitertTensorBackend() override = default;

  // Not copyable or movable: the compiled runner holds views into `data_` and
  // `xnn_runner_` aliases `runner_`.
  LitertTensorBackend(const LitertTensorBackend&) = delete;
  LitertTensorBackend& operator=(const LitertTensorBackend&) = delete;

  std::string Name() const override { return "LitertTensorBackend"; }

  // Validates the spec, ensures XNNPACK tracking initialization, constructs
  // the LiteRT tensor graph, prepares the runner, and binds threads. Replaces
  // any previously set up layer. Returns InvalidArgument when `spec` is
  // invalid or a `data` buffer is too small for it, and Unimplemented when
  // `spec.dtype` is not FP32, the only precision the layers execute in. After
  // any failure the backend holds no layer: Execute() and GetOutputData()
  // return FailedPrecondition until a later Setup() succeeds.
  absl::Status Setup(const LayerSpec& spec, const SeededLayerData& data,
                     const BackendOptions& options) override;
  absl::Status Setup(const LayerSpec& spec, SeededLayerData&& data,
                     const BackendOptions& options) override;

  // Invokes a single forward execution of the prepared graph.
  absl::Status Execute() override;

  // Returns the underlying worker threadpool from XnnpackRunner (or nullptr
  // when using YnnpackRunner).
  pthreadpool_t GetThreadpool() const override;

  // Reads the output tensor buffer into a flat contiguous FP32 vector.
  absl::StatusOr<std::vector<float>> GetOutputData() const override;

 private:
  LayerSpec spec_;
  SeededLayerData data_;
  std::optional<LitertLayerGraph> graph_;
  std::unique_ptr<::litert::tensor::NnpackRunner> runner_;
  ::litert::tensor::XnnpackRunner* xnn_runner_ = nullptr;
  RuntimeBackend runtime_backend_ = RuntimeBackend::kXnnpack;
};

}  // namespace layer_ml_bench::litert

#endif  // TENSOR_BENCHMARKS_LAYER_LITERT_TENSOR_BACKEND_H_
