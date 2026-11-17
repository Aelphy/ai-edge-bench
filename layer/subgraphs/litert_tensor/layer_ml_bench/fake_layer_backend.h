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

#ifndef TENSOR_BENCHMARKS_LAYER_FAKE_LAYER_BACKEND_H_
#define TENSOR_BENCHMARKS_LAYER_FAKE_LAYER_BACKEND_H_

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_spec.h"

namespace layer_ml_bench {

// In-memory LayerBackend for tests of code that drives a backend.
//
// Follows the LayerBackend lifecycle without running a real layer: Setup()
// validates the spec and retains its arguments, Execute() counts forward
// passes, and GetOutputData() returns one element per element of the bound
// spec's OutputShape(), each holding the number of forward passes completed so
// far. Execute() and GetOutputData() return FailedPrecondition unless the most
// recent Setup() succeeded.
//
// Every entry point can be made to fail with a chosen status, and callbacks
// let a test simulate the side effects of a real Setup() or Execute().
//
// Only the pure virtual methods of LayerBackend are overridden, so
// GetThreadpool() and the rvalue Setup() overload keep the LayerBackend
// defaults.
class FakeLayerBackend : public LayerBackend {
 public:
  FakeLayerBackend() = default;

  // Not copyable or movable: tests observe one instance through the pointer
  // they keep after handing ownership to the code under test.
  FakeLayerBackend(const FakeLayerBackend&) = delete;
  FakeLayerBackend& operator=(const FakeLayerBackend&) = delete;

  // Makes every Setup() return `status` without retaining its arguments.
  void FailSetupWith(absl::Status status) { setup_error_ = std::move(status); }

  // Makes Execute() return `status` once `successful_executions` forward
  // passes have completed.
  void FailExecuteWith(absl::Status status, int successful_executions = 0) {
    execute_error_ = std::move(status);
    successful_executions_before_error_ = successful_executions;
  }

  // Makes every GetOutputData() return `status`.
  void FailGetOutputDataWith(absl::Status status) {
    output_error_ = std::move(status);
  }

  // Overrides the element count of the output tensor.
  void set_output_size(size_t output_size) { output_size_ = output_size; }

  // Sets a callback invoked at the end of every successful Setup().
  void set_on_setup(std::function<void()> on_setup) {
    on_setup_ = std::move(on_setup);
  }

  // Sets a callback invoked at the end of every successful Execute().
  void set_on_execute(std::function<void()> on_execute) {
    on_execute_ = std::move(on_execute);
  }

  // LayerBackend implementation.
  std::string Name() const override { return "FakeLayerBackend"; }
  absl::Status Setup(const LayerSpec& spec, const SeededLayerData& data,
                     const BackendOptions& options) override;
  absl::Status Execute() override;
  absl::StatusOr<std::vector<float>> GetOutputData() const override;

  // Number of Setup() calls, whether or not they succeeded.
  int setup_call_count() const { return setup_call_count_; }

  // Number of forward passes completed, i.e. Execute() calls that succeeded.
  int execute_count() const { return execute_count_; }

  // Data and options retained by the most recent successful Setup().
  const SeededLayerData& data() const { return data_; }
  const BackendOptions& options() const { return options_; }

 private:
  absl::Status setup_error_;
  absl::Status execute_error_;
  int successful_executions_before_error_ = 0;
  absl::Status output_error_;
  std::optional<size_t> output_size_;
  std::function<void()> on_setup_;
  std::function<void()> on_execute_;

  LayerSpec spec_;
  SeededLayerData data_;
  BackendOptions options_;
  bool is_set_up_ = false;
  int setup_call_count_ = 0;
  int execute_count_ = 0;
};

}  // namespace layer_ml_bench

#endif  // TENSOR_BENCHMARKS_LAYER_FAKE_LAYER_BACKEND_H_
