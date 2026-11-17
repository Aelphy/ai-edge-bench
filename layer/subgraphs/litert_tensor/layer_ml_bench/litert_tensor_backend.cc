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

#include "tensor/benchmarks/layer/litert_tensor_backend.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "pthreadpool.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_builders.h"
#include "tensor/benchmarks/layer/layer_spec.h"
#include "tensor/benchmarks/layer/tracking_allocator.h"
#include "tensor/runners/common_nnpack/runner.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/runners/ynnpack/runner.h"

namespace layer_ml_bench::litert {

absl::Status LitertTensorBackend::Setup(const LayerSpec& spec,
                                        const SeededLayerData& data,
                                        const BackendOptions& options) {
  SeededLayerData copy = data;
  return Setup(spec, std::move(copy), options);
}

absl::Status LitertTensorBackend::Setup(const LayerSpec& spec,
                                        SeededLayerData&& data,
                                        const BackendOptions& options) {
  // Tear down any previously compiled runner and graph first: every failure
  // below must leave the backend without a layer, and SpanCpuBuffer views
  // owned by an earlier graph must never outlive the data_ they point into.
  runner_.reset();
  xnn_runner_ = nullptr;
  graph_.reset();

  absl::Status val_status = spec.Validate();
  if (!val_status.ok()) {
    return val_status;
  }

  const bool use_ynn =
      (runtime_backend_ == RuntimeBackend::kYnnpack) || options.use_ynnpack;
  if (!use_ynn) {
    // Ensure XNNPACK is initialized with global TrackingAllocator before runner
    // initialization.
    EnsureXnnpackInitializedWithTracking();
  }

  spec_ = spec;
  // Take ownership of host buffers so pointers remain valid for the entire
  // lifetime of runner_ without duplicating allocations inside Setup().
  data_ = std::move(data);

  auto reset_on_error = [this]() {
    runner_.reset();
    xnn_runner_ = nullptr;
    graph_.reset();
  };

  absl::StatusOr<LitertLayerGraph> graph = BuildLayer(spec_, data_);
  if (!graph.ok()) {
    reset_on_error();
    return graph.status();
  }
  graph_ = std::move(*graph);

  if (use_ynn) {
    absl::StatusOr<::litert::tensor::YnnpackRunner> runner =
        ::litert::tensor::YnnpackRunner::Create({graph_->output});
    if (!runner.ok()) {
      reset_on_error();
      return runner.status();
    }
    runner_ =
        std::make_unique<::litert::tensor::YnnpackRunner>(std::move(*runner));
    xnn_runner_ = nullptr;
  } else {
    absl::StatusOr<::litert::tensor::XnnpackRunner> runner =
        ::litert::tensor::XnnpackRunner::Create({graph_->output});
    if (!runner.ok()) {
      reset_on_error();
      return runner.status();
    }
    auto xnn_ptr =
        std::make_unique<::litert::tensor::XnnpackRunner>(std::move(*runner));
    xnn_runner_ = xnn_ptr.get();
    runner_ = std::move(xnn_ptr);
  }

  if (options.num_threads > 1) {
    absl::Status thread_status = runner_->SetNumThreads(options.num_threads);
    if (!thread_status.ok()) {
      reset_on_error();
      return thread_status;
    }
  }

  for (const ExternalInputBinding& binding : graph_->external_inputs) {
    if (binding.host_buffer == nullptr) {
      reset_on_error();
      return absl::InternalError(
          "ExternalInputBinding has null host_buffer pointer.");
    }
    const size_t count = binding.num_elements > 0 ? binding.num_elements
                                                  : binding.host_buffer->size();
    if (count > binding.host_buffer->size()) {
      reset_on_error();
      return absl::InvalidArgumentError(absl::StrFormat(
          "SeededLayerData buffer holds %zu elements but the layer binds %zu.",
          binding.host_buffer->size(), count));
    }
    absl::Span<const float> span =
        absl::MakeConstSpan(*binding.host_buffer).first(count);
    absl::Status input_status = runner_->SetInput(binding.tensor, span);
    if (!input_status.ok()) {
      reset_on_error();
      return input_status;
    }
  }

  absl::Status prep_status = runner_->PrepareRuntime();
  if (!prep_status.ok()) {
    reset_on_error();
    return prep_status;
  }
  return absl::OkStatus();
}

absl::Status LitertTensorBackend::Execute() {
  if (runner_ == nullptr) {
    return absl::FailedPreconditionError("LitertTensorBackend not setup yet.");
  }
  return runner_->Run();
}

pthreadpool_t LitertTensorBackend::GetThreadpool() const {
  if (xnn_runner_ == nullptr) {
    return nullptr;
  }
  return xnn_runner_->threadpool();
}

absl::StatusOr<std::vector<float>> LitertTensorBackend::GetOutputData() const {
  if (runner_ == nullptr || !graph_.has_value()) {
    return absl::FailedPreconditionError("LitertTensorBackend not setup yet.");
  }
  absl::StatusOr<::litert::tensor::LockedBufferSpan<const float>> output =
      runner_->ReadOutputAs<float>(graph_->output);
  if (!output.ok()) {
    return output.status();
  }
  return std::vector<float>(output->begin(), output->end());
}

}  // namespace layer_ml_bench::litert
