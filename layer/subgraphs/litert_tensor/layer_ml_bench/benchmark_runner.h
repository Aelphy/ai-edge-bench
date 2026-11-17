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

#ifndef TENSOR_BENCHMARKS_LAYER_BENCHMARK_RUNNER_H_
#define TENSOR_BENCHMARKS_LAYER_BENCHMARK_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "pthreadpool.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_spec.h"
#include "tensor/benchmarks/layer/parity_validator.h"
#include "tensor/benchmarks/layer/tracking_allocator.h"

namespace layer_ml_bench {

// Configuration options controlling benchmark execution.
struct BenchmarkOptions {
  size_t num_threads = 1;
  bool wipe_caches = false;
  uint32_t seed = 1234;
  // Optional override for total cache-eviction buffer size in bytes (0 uses
  // the production default of max(256 MB, num_workers * 16 MB)). Intended as a
  // lightweight seam for unit tests.
  size_t wipe_buffer_bytes = 0;
  // Optional override for the process heap sampler behind the PeakMem_MB and
  // AllocDelta_MB telemetry (null uses the MemoryTracker default). Intended
  // as a lightweight seam for unit tests.
  MemoryTracker::HeapSampler heap_sampler;
  // Optional override for the allocator whose growth that telemetry reports
  // (null uses GetGlobalTrackingAllocator(), which XNNPACK allocates through).
  // Must outlive the runner. Intended as a lightweight seam for unit tests.
  TrackingAllocator* tracking_allocator = nullptr;
};

using BackendFactory = std::function<std::unique_ptr<LayerBackend>()>;

// Framework-agnostic benchmark runner bridging any LayerBackend to Google
// Benchmark State, managing cache-flushing, warm-up, and AI agent telemetry.
class LayerBenchmarkRunner {
 public:
  // Creates and initializes a runner with a concrete backend instance.
  static absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> Create(
      std::unique_ptr<LayerBackend> backend, const LayerSpec& spec,
      const BenchmarkOptions& options);

  // Creates and initializes a runner using a backend factory function.
  static absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> Create(
      BackendFactory backend_factory, const LayerSpec& spec,
      const BenchmarkOptions& options);

  ~LayerBenchmarkRunner();

  LayerBenchmarkRunner(const LayerBenchmarkRunner&) = delete;
  LayerBenchmarkRunner& operator=(const LayerBenchmarkRunner&) = delete;

  // Executes the timed benchmark loop across Google Benchmark State,
  // performing optional cache-wiping before each iteration and populating
  // machine-actionable Roofline and memory counters upon completion.
  absl::Status RunBenchmark(::benchmark::State& state);

  // Executes a single untimed forward pass for verification or testing.
  absl::Status ExecuteOnce();

  // Reads the output tensor from the backend as flat FP32 values.
  absl::StatusOr<std::vector<float>> GetOutputData() const;

  // Recorded setup and warm-up memory telemetry (outside timed loop).
  double PeakAllocatedMB() const;
  double AllocatedDeltaMB() const;

  const LayerSpec& spec() const;
  const LayerBackend& backend() const;

 private:
  LayerBenchmarkRunner(std::unique_ptr<LayerBackend> backend, LayerSpec spec,
                       BenchmarkOptions options);

  std::unique_ptr<LayerBackend> backend_;
  LayerSpec spec_;
  BenchmarkOptions options_;
  MemoryTracker memory_tracker_;
  std::shared_ptr<const std::vector<uint8_t>> wipe_buffer_;
  pthreadpool_t fallback_wipe_pool_ = nullptr;
};

// Output tensor dump and comparison requested for a benchmark case.
struct ParityOptions {
  // File receiving the output tensor (see DumpTensorToFile); empty disables
  // the dump.
  std::string dump_path;
  // Reference dump the output tensor is compared against; empty disables the
  // comparison.
  std::string compare_path;
  // Elementwise tolerances of the comparison (see CompareTensors).
  float rtol = 1e-4f;
  float atol = 1e-5f;
  // Distribution gates that replace the elementwise check when positive (see
  // EvaluateParityGate).
  double min_cosine = 0.0;
  double max_l2_rel_err = 0.0;
};

// Benchmarks the layer `spec` through `state` on a backend made by
// `backend_factory`, then dumps and compares the layer output as `parity`
// requests. A passing comparison reports the Parity_Passed, Parity_MaxAbsErr,
// Parity_MaxRelErr, Parity_CosineSim, Parity_L2RelErr, and Parity_Mismatches
// user counters; a failing one is described by its error message, because
// the benchmark library drops the counters of a failed case.
//
// The files named by `parity` each hold one output tensor, so `parity_guard`
// admits a single case, identified by `case_name`, to them; the benchmark
// library may invoke that case any number of times.
//
// Every failure (a refused case, a backend setup or execution error, an
// unreadable reference, an unwritable dump, or a failed parity gate) is
// reported through `state.SkipWithError`.
void RunLayerBenchmarkCase(::benchmark::State& state,
                           absl::string_view case_name, const LayerSpec& spec,
                           const BackendFactory& backend_factory,
                           const BenchmarkOptions& options,
                           const ParityOptions& parity,
                           SingleCaseParityGuard& parity_guard);

}  // namespace layer_ml_bench

#endif  // TENSOR_BENCHMARKS_LAYER_BENCHMARK_RUNNER_H_
