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

#include "tensor/benchmarks/layer/benchmark_runner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "benchmark/benchmark.h"
#include "pthreadpool.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_spec.h"
#include "tensor/benchmarks/layer/parity_validator.h"
#include "tensor/benchmarks/layer/tracking_allocator.h"

namespace layer_ml_bench {
namespace {

// Minimum total cache-eviction working set (256 MB) so large server LLCs
// (e.g., 105 MB Sapphire Rapids, 128 MB EPYC Milan) are completely displaced,
// plus at least 16 MB per worker thread so each core flushes its private L1/L2
// over an independent address shard.
constexpr size_t kMinTotalWipeBytes = 256 * 1024 * 1024;
constexpr size_t kPerWorkerWipeBytes = 16 * 1024 * 1024;

// Touch memory with cache line step (64 bytes) to evict cached data to RAM.
void TouchBufferToEvictCache(const void* ptr, size_t size, size_t step = 64) {
  const auto* u8_ptr = static_cast<const uint8_t*>(ptr);
  uint32_t sum = 0;
  while (size >= step) {
    sum += static_cast<uint32_t>(*u8_ptr);
    u8_ptr += step;
    size -= step;
  }
  ::benchmark::DoNotOptimize(sum);
}

size_t RequiredWipeBufferBytes(size_t num_workers, size_t override_bytes) {
  if (override_bytes > 0) {
    return override_bytes;
  }
  const size_t workers = std::max<size_t>(1, num_workers);
  return std::max(kMinTotalWipeBytes, workers * kPerWorkerWipeBytes);
}

std::shared_ptr<const std::vector<uint8_t>> GetOrCreateWipeBuffer(
    size_t num_workers, size_t override_bytes) {
  if (override_bytes > 0) {
    return std::make_shared<std::vector<uint8_t>>(override_bytes, 0xA5);
  }
  ABSL_CONST_INIT static absl::Mutex wipe_mu(absl::kConstInit);
  static auto* cached_buf = new std::shared_ptr<const std::vector<uint8_t>>();

  const size_t needed =
      RequiredWipeBufferBytes(num_workers, /*override_bytes=*/0);
  absl::MutexLock lock(&wipe_mu);
  if (*cached_buf == nullptr || (*cached_buf)->size() < needed) {
    *cached_buf = std::make_shared<std::vector<uint8_t>>(needed, 0xA5);
  }
  return *cached_buf;
}

struct ShardedWipeContext {
  const uint8_t* data = nullptr;
  size_t total_bytes = 0;
  size_t num_workers = 1;
};

void WipeCacheShard(const ShardedWipeContext& ctx, size_t worker_id) {
  if (ctx.data == nullptr || ctx.total_bytes == 0 ||
      worker_id >= ctx.num_workers) {
    return;
  }
  const size_t shard_bytes = (ctx.total_bytes / ctx.num_workers) & ~size_t{63};
  const size_t offset = worker_id * shard_bytes;
  if (offset < ctx.total_bytes) {
    const size_t len = (worker_id + 1 == ctx.num_workers)
                           ? (ctx.total_bytes - offset)
                           : shard_bytes;
    TouchBufferToEvictCache(ctx.data + offset, len);
  }
}

void SkipWithStatus(::benchmark::State& state, const absl::Status& status) {
  state.SkipWithError(std::string(status.message()).c_str());
}

// Writes the latest layer output of `runner` to `dump_path`.
absl::Status DumpOutput(const LayerBenchmarkRunner& runner,
                        const std::string& dump_path) {
  absl::StatusOr<std::vector<float>> output = runner.GetOutputData();
  if (!output.ok()) {
    return output.status();
  }
  return DumpTensorToFile(dump_path, *output, runner.spec().OutputShape());
}

// Compares the latest layer output of `runner` with the reference dump named
// by `parity`, reports the Parity_* counters on `state`, and returns the gate
// outcome. Fails if the reference or the layer output cannot be read.
absl::StatusOr<ParityGateResult> CompareOutput(
    const LayerBenchmarkRunner& runner, const ParityOptions& parity,
    ::benchmark::State& state) {
  absl::StatusOr<std::pair<std::vector<int64_t>, std::vector<float>>>
      reference = LoadTensorFromFile(parity.compare_path);
  if (!reference.ok()) {
    return reference.status();
  }
  absl::StatusOr<std::vector<float>> output = runner.GetOutputData();
  if (!output.ok()) {
    return output.status();
  }
  const std::vector<int64_t> output_shape = runner.spec().OutputShape();
  const ParityReport report =
      CompareWithReference(reference->first, reference->second, output_shape,
                           *output, parity.rtol, parity.atol);
  ParityGateResult gate =
      EvaluateParityGate(report, parity.min_cosine, parity.max_l2_rel_err);

  state.counters["Parity_Passed"] = gate.passed ? 1.0 : 0.0;
  state.counters["Parity_MaxAbsErr"] = report.max_absolute_error;
  state.counters["Parity_MaxRelErr"] = report.max_relative_error;
  state.counters["Parity_CosineSim"] = report.cosine_similarity;
  state.counters["Parity_L2RelErr"] = report.l2_relative_error;
  state.counters["Parity_Mismatches"] =
      static_cast<double>(report.mismatch_count);
  return gate;
}

}  // namespace

LayerBenchmarkRunner::LayerBenchmarkRunner(
    std::unique_ptr<LayerBackend> backend, LayerSpec spec,
    BenchmarkOptions options)
    : backend_(std::move(backend)),
      spec_(std::move(spec)),
      options_(std::move(options)),
      memory_tracker_(options_.tracking_allocator, options_.heap_sampler) {}

LayerBenchmarkRunner::~LayerBenchmarkRunner() {
  if (fallback_wipe_pool_ != nullptr) {
    pthreadpool_destroy(fallback_wipe_pool_);
    fallback_wipe_pool_ = nullptr;
  }
}

absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>>
LayerBenchmarkRunner::Create(std::unique_ptr<LayerBackend> backend,
                             const LayerSpec& spec,
                             const BenchmarkOptions& options) {
  if (backend == nullptr) {
    return absl::InvalidArgumentError("LayerBackend must not be null.");
  }
  absl::Status val_status = spec.Validate();
  if (!val_status.ok()) {
    return val_status;
  }

  EnsureXnnpackInitializedWithTracking();

  auto runner = std::unique_ptr<LayerBenchmarkRunner>(
      new LayerBenchmarkRunner(std::move(backend), spec, options));

  // Allocate deterministic host buffers and any cache-wipe buffer BEFORE
  // starting MemoryTracker so PeakMem_MB reflects backend weight packing and
  // runtime workspace rather than benchmark harness buffers.
  SeededLayerData data = CreateSeededLayerData(spec, options.seed);
  if (options.wipe_caches) {
    runner->wipe_buffer_ =
        GetOrCreateWipeBuffer(options.num_threads, options.wipe_buffer_bytes);
  }

  // Capture setup and warm-up memory allocation high-water mark.
  runner->memory_tracker_.Start();

  BackendOptions backend_options;
  backend_options.num_threads = options.num_threads;

  absl::Status setup_status =
      runner->backend_->Setup(spec, std::move(data), backend_options);
  if (!setup_status.ok()) {
    return setup_status;
  }

  // Untimed warm-up iteration.
  absl::Status warmup_status = runner->backend_->Execute();
  if (!warmup_status.ok()) {
    return warmup_status;
  }

  runner->memory_tracker_.Stop();

  // Backends that manage an internal threadpool (e.g., YnnpackRunner) return
  // nullptr from GetThreadpool(); provision a fallback pool of matching width
  // so --wipe_caches flushes private L2 caches across worker threads too.
  if (options.wipe_caches && runner->backend_->GetThreadpool() == nullptr &&
      options.num_threads > 1) {
    runner->fallback_wipe_pool_ = pthreadpool_create(options.num_threads);
    const size_t pool_threads =
        runner->fallback_wipe_pool_ != nullptr
            ? pthreadpool_get_threads_count(runner->fallback_wipe_pool_)
            : options.num_threads;
    runner->wipe_buffer_ = GetOrCreateWipeBuffer(
        std::max(options.num_threads, pool_threads), options.wipe_buffer_bytes);
  } else if (options.wipe_caches &&
             runner->backend_->GetThreadpool() != nullptr) {
    const size_t pool_threads =
        pthreadpool_get_threads_count(runner->backend_->GetThreadpool());
    runner->wipe_buffer_ = GetOrCreateWipeBuffer(
        std::max(options.num_threads, pool_threads), options.wipe_buffer_bytes);
  }

  return runner;
}

absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>>
LayerBenchmarkRunner::Create(BackendFactory backend_factory,
                             const LayerSpec& spec,
                             const BenchmarkOptions& options) {
  if (!backend_factory) {
    return absl::InvalidArgumentError("BackendFactory must not be null.");
  }
  std::unique_ptr<LayerBackend> backend = backend_factory();
  return Create(std::move(backend), spec, options);
}

absl::Status LayerBenchmarkRunner::RunBenchmark(::benchmark::State& state) {
  for (auto _ : state) {
    if (options_.wipe_caches) {
      state.PauseTiming();
      pthreadpool_t threadpool = backend_->GetThreadpool();
      if (threadpool == nullptr) {
        threadpool = fallback_wipe_pool_;
      }
      const size_t num_workers =
          threadpool != nullptr
              ? std::max<size_t>(1, pthreadpool_get_threads_count(threadpool))
              : 1;
      if (wipe_buffer_ == nullptr ||
          wipe_buffer_->size() < RequiredWipeBufferBytes(
                                     num_workers, options_.wipe_buffer_bytes)) {
        wipe_buffer_ =
            GetOrCreateWipeBuffer(num_workers, options_.wipe_buffer_bytes);
      }
      ShardedWipeContext ctx{
          .data = wipe_buffer_->data(),
          .total_bytes = wipe_buffer_->size(),
          .num_workers = num_workers,
      };
      if (threadpool != nullptr && num_workers > 1) {
        pthreadpool_parallelize_1d(
            threadpool,
            [](void* raw_ctx, size_t worker_id) {
              WipeCacheShard(*static_cast<const ShardedWipeContext*>(raw_ctx),
                             worker_id);
            },
            &ctx, num_workers, /*flags=*/0);
      } else {
        WipeCacheShard(ctx, /*worker_id=*/0);
      }
      state.ResumeTiming();
    }

    // Pure forward timing without profiling or allocation tracking overhead.
    absl::Status status = backend_->Execute();
    if (!status.ok()) {
      SkipWithStatus(state, status);
      return status;
    }
  }

  // Report machine-actionable Roofline and memory metrics for AI agents.
  const double total_flops = static_cast<double>(spec_.TotalFlops());
  const double total_bytes = static_cast<double>(spec_.TotalMemoryBytes());
  const double iterations = static_cast<double>(state.iterations());

  state.counters["GFLOP/s"] = ::benchmark::Counter(
      total_flops * iterations / 1e9, ::benchmark::Counter::kIsRate);
  state.counters["GB/s"] = ::benchmark::Counter(total_bytes * iterations / 1e9,
                                                ::benchmark::Counter::kIsRate);
  state.counters["ArithmeticIntensity"] = spec_.ArithmeticIntensity();
  state.counters["Theoretical_FLOPs"] = total_flops;
  state.counters["Theoretical_Bytes"] = total_bytes;
  state.counters["SeqLenQ"] = static_cast<double>(spec_.EffectiveSeqLenQ());
  state.counters["SeqLenKV"] = static_cast<double>(spec_.EffectiveSeqLenKV());
  state.counters["PeakMem_MB"] = memory_tracker_.PeakAllocatedMB();
  state.counters["AllocDelta_MB"] = memory_tracker_.AllocatedDeltaMB();

  return absl::OkStatus();
}

absl::Status LayerBenchmarkRunner::ExecuteOnce() { return backend_->Execute(); }

absl::StatusOr<std::vector<float>> LayerBenchmarkRunner::GetOutputData() const {
  return backend_->GetOutputData();
}

double LayerBenchmarkRunner::PeakAllocatedMB() const {
  return memory_tracker_.PeakAllocatedMB();
}

double LayerBenchmarkRunner::AllocatedDeltaMB() const {
  return memory_tracker_.AllocatedDeltaMB();
}

const LayerSpec& LayerBenchmarkRunner::spec() const { return spec_; }

const LayerBackend& LayerBenchmarkRunner::backend() const { return *backend_; }

void RunLayerBenchmarkCase(::benchmark::State& state,
                           absl::string_view case_name, const LayerSpec& spec,
                           const BackendFactory& backend_factory,
                           const BenchmarkOptions& options,
                           const ParityOptions& parity,
                           SingleCaseParityGuard& parity_guard) {
  if (!parity.dump_path.empty() || !parity.compare_path.empty()) {
    absl::Status claim = parity_guard.Claim(case_name);
    if (!claim.ok()) {
      state.SkipWithError(
          absl::StrCat("Parity dump and reference files hold one benchmark "
                       "case; narrow the benchmark filter to it: ",
                       claim.message())
              .c_str());
      return;
    }
  }

  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(backend_factory, spec, options);
  if (!runner.ok()) {
    SkipWithStatus(state, runner.status());
    return;
  }
  // RunBenchmark reports its own failure to `state`.
  if (!(*runner)->RunBenchmark(state).ok()) {
    return;
  }

  if (!parity.dump_path.empty()) {
    absl::Status dump_status = DumpOutput(**runner, parity.dump_path);
    if (!dump_status.ok()) {
      SkipWithStatus(state, dump_status);
      return;
    }
  }
  if (!parity.compare_path.empty()) {
    absl::StatusOr<ParityGateResult> gate =
        CompareOutput(**runner, parity, state);
    if (!gate.ok()) {
      SkipWithStatus(state, gate.status());
      return;
    }
    if (!gate->passed) {
      state.SkipWithError(gate->details.c_str());
    }
  }
}

}  // namespace layer_ml_bench
