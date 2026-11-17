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

#include "tensor/benchmarks/layer/fake_layer_backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_spec.h"

namespace layer_ml_bench {
namespace {

// Returns the number of elements of a tensor of `shape`.
size_t ElementCount(const std::vector<int64_t>& shape) {
  size_t element_count = 1;
  for (int64_t dim : shape) {
    element_count *= static_cast<size_t>(dim);
  }
  return element_count;
}

}  // namespace

absl::Status FakeLayerBackend::Setup(const LayerSpec& spec,
                                     const SeededLayerData& data,
                                     const BackendOptions& options) {
  ++setup_call_count_;
  is_set_up_ = false;
  if (!setup_error_.ok()) {
    return setup_error_;
  }
  absl::Status spec_status = spec.Validate();
  if (!spec_status.ok()) {
    return spec_status;
  }
  spec_ = spec;
  data_ = data;
  options_ = options;
  is_set_up_ = true;
  if (on_setup_) {
    on_setup_();
  }
  return absl::OkStatus();
}

absl::Status FakeLayerBackend::Execute() {
  if (!is_set_up_) {
    return absl::FailedPreconditionError("FakeLayerBackend is not set up.");
  }
  if (!execute_error_.ok() &&
      execute_count_ >= successful_executions_before_error_) {
    return execute_error_;
  }
  ++execute_count_;
  if (on_execute_) {
    on_execute_();
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> FakeLayerBackend::GetOutputData() const {
  if (!is_set_up_) {
    return absl::FailedPreconditionError("FakeLayerBackend is not set up.");
  }
  if (!output_error_.ok()) {
    return output_error_;
  }
  return std::vector<float>(
      output_size_.value_or(ElementCount(spec_.OutputShape())),
      static_cast<float>(execute_count_));
}

}  // namespace layer_ml_bench
