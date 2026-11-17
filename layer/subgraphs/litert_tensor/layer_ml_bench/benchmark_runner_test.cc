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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "tensor/benchmarks/layer/fake_layer_backend.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_spec.h"
#include "tensor/benchmarks/layer/litert_tensor_backend.h"
#include "tensor/benchmarks/layer/parity_validator.h"
#include "tensor/benchmarks/layer/status_test_util.h"
#include "tensor/benchmarks/layer/tracking_allocator.h"

namespace layer_ml_bench {
namespace {

using ::layer_ml_bench::litert::LitertTensorBackend;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::IsSupersetOf;
using ::testing::Matcher;
using ::testing::Pair;
using ::testing::SizeIs;

constexpr size_t kMiB = 1024 * 1024;

// Cache-eviction buffer for tests that wipe caches; the production default of
// at least 256 MiB would dominate their memory use.
constexpr size_t kTestWipeBufferBytes = 64 * 1024;

// Small layer: B=1, S=T=8, H=4, D=16, N_kv=4, F=32, M=H*D=64.
LayerSpec CreateTestSpec(
    LayerType layer_type = LayerType::kRmsNorm,
    AttentionMode attention_mode = AttentionMode::kPrefill) {
  LayerSpec spec;
  spec.model_name = "TestModel";
  spec.layer_type = layer_type;
  spec.attention_mode = attention_mode;
  spec.mlp_activation = MlpActivation::kSwiGLU;
  spec.batch_size = 1;
  spec.seq_len_q = 8;
  spec.seq_len_kv = 8;
  spec.num_heads = 4;
  spec.num_kv_heads = 4;
  spec.head_dim = 16;
  spec.hidden_dim = 32;
  spec.eps = 1e-5f;
  return spec;
}

// Returns options whose heap sampler reports a heap that never grows, so that
// memory telemetry reflects only what the tracked allocator records.
BenchmarkOptions OptionsWithIdleHeap() {
  BenchmarkOptions options;
  options.heap_sampler = [] { return size_t{0}; };
  return options;
}

// Returns options that additionally measure `allocator` in place of the
// process-wide tracking allocator.
BenchmarkOptions OptionsMeasuring(TrackingAllocator& allocator) {
  BenchmarkOptions options = OptionsWithIdleHeap();
  options.tracking_allocator = &allocator;
  return options;
}

// Matches the benchmark user counter `name` when its value satisfies
// `value_matcher`.
Matcher<const benchmark::UserCounters::value_type&> CounterNamed(
    std::string name, Matcher<double> value_matcher) {
  return Pair(std::move(name),
              Field(&benchmark::Counter::value, std::move(value_matcher)));
}

TEST(LayerBenchmarkRunnerTest, CreateRejectsNullBackend) {
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::unique_ptr<LayerBackend>(),
                                   CreateTestSpec(), BenchmarkOptions{});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

TEST(LayerBenchmarkRunnerTest, CreateRejectsEmptyBackendFactory) {
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(BackendFactory(), CreateTestSpec(),
                                   BenchmarkOptions{});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

TEST(LayerBenchmarkRunnerTest, CreateRejectsFactoryReturningNullBackend) {
  BackendFactory null_backend_factory = []() -> std::unique_ptr<LayerBackend> {
    return nullptr;
  };

  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(null_backend_factory, CreateTestSpec(),
                                   BenchmarkOptions{});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

TEST(LayerBenchmarkRunnerTest, CreateRejectsInvalidSpec) {
  LayerSpec spec = CreateTestSpec();
  spec.head_dim = 15;  // Odd head dimensions cannot be rotated by RoPE.

  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::make_unique<FakeLayerBackend>(), spec,
                                   BenchmarkOptions{});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

TEST(LayerBenchmarkRunnerTest, CreatePropagatesSetupFailure) {
  auto backend = std::make_unique<FakeLayerBackend>();
  backend->FailSetupWith(absl::ResourceExhaustedError("setup failed"));

  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   BenchmarkOptions{});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kResourceExhausted)
      << runner.status();
}

TEST(LayerBenchmarkRunnerTest, CreatePropagatesWarmupFailure) {
  auto backend = std::make_unique<FakeLayerBackend>();
  backend->FailExecuteWith(absl::DataLossError("warm-up failed"));

  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   BenchmarkOptions{});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kDataLoss)
      << runner.status();
}

TEST(LayerBenchmarkRunnerTest, CreateSetsUpBackendWithRequestedThreadCount) {
  auto backend = std::make_unique<FakeLayerBackend>();
  FakeLayerBackend* fake = backend.get();
  BenchmarkOptions options;
  options.num_threads = 2;

  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   options);

  ASSERT_OK(runner.status());
  EXPECT_EQ(fake->options().num_threads, 2);
}

TEST(LayerBenchmarkRunnerTest, CreateBindsDataSeededWithOptionsSeed) {
  const LayerSpec spec = CreateTestSpec();
  auto backend = std::make_unique<FakeLayerBackend>();
  FakeLayerBackend* fake = backend.get();
  BenchmarkOptions options;
  options.seed = 999;

  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), spec, options);

  ASSERT_OK(runner.status());
  EXPECT_EQ(fake->data().input, CreateSeededLayerData(spec, 999).input);
}

TEST(LayerBenchmarkRunnerTest, CreateRunsOneWarmupForwardPass) {
  auto backend = std::make_unique<FakeLayerBackend>();
  FakeLayerBackend* fake = backend.get();

  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   BenchmarkOptions{});

  ASSERT_OK(runner.status());
  EXPECT_EQ(fake->execute_count(), 1);
}

TEST(LayerBenchmarkRunnerTest, CreateWithFactoryOwnsProducedBackend) {
  const LayerBackend* produced_backend = nullptr;
  BackendFactory backend_factory =
      [&produced_backend]() -> std::unique_ptr<LayerBackend> {
    auto backend = std::make_unique<FakeLayerBackend>();
    produced_backend = backend.get();
    return backend;
  };

  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(backend_factory, CreateTestSpec(),
                                   BenchmarkOptions{});

  ASSERT_OK(runner.status());
  EXPECT_EQ(&(*runner)->backend(), produced_backend);
}

TEST(LayerBenchmarkRunnerTest, SpecReturnsSpecGivenToCreate) {
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::make_unique<FakeLayerBackend>(),
                                   CreateTestSpec(LayerType::kGatedMlp),
                                   BenchmarkOptions{});

  ASSERT_OK(runner.status());
  EXPECT_EQ((*runner)->spec().layer_type, LayerType::kGatedMlp);
  EXPECT_EQ((*runner)->spec().hidden_dim, 32);
}

TEST(LayerBenchmarkRunnerTest, ExecuteOnceRunsOneMoreForwardPass) {
  auto backend = std::make_unique<FakeLayerBackend>();
  FakeLayerBackend* fake = backend.get();
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   BenchmarkOptions{});
  ASSERT_OK(runner.status());

  ASSERT_OK((*runner)->ExecuteOnce());

  // One warm-up pass during Create() plus the requested pass.
  EXPECT_EQ(fake->execute_count(), 2);
}

TEST(LayerBenchmarkRunnerTest, ExecuteOncePropagatesBackendFailure) {
  auto backend = std::make_unique<FakeLayerBackend>();
  // The warm-up pass during Create() succeeds; the next pass fails.
  backend->FailExecuteWith(absl::DataLossError("execute failed"),
                           /*successful_executions=*/1);
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   BenchmarkOptions{});
  ASSERT_OK(runner.status());

  EXPECT_EQ((*runner)->ExecuteOnce().code(), absl::StatusCode::kDataLoss);
}

TEST(LayerBenchmarkRunnerTest, GetOutputDataReturnsLatestBackendOutput) {
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::make_unique<FakeLayerBackend>(),
                                   CreateTestSpec(), BenchmarkOptions{});
  ASSERT_OK(runner.status());
  ASSERT_OK((*runner)->ExecuteOnce());

  absl::StatusOr<std::vector<float>> output = (*runner)->GetOutputData();

  // The fake output holds the forward-pass count (warm-up plus one) in each
  // of the B*T*M = 1*8*64 elements.
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, AllOf(SizeIs(512), Each(Eq(2.0f))));
}

TEST(LayerBenchmarkRunnerTest, GetOutputDataPropagatesBackendFailure) {
  auto backend = std::make_unique<FakeLayerBackend>();
  backend->FailGetOutputDataWith(absl::UnavailableError("output unavailable"));
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   BenchmarkOptions{});
  ASSERT_OK(runner.status());

  absl::StatusOr<std::vector<float>> output = (*runner)->GetOutputData();

  EXPECT_EQ(output.status().code(), absl::StatusCode::kUnavailable)
      << output.status();
}

TEST(LayerBenchmarkRunnerTest, MemoryTelemetryCoversSetupAndWarmup) {
  constexpr size_t kRetainedBytes = 4 * kMiB;
  constexpr size_t kTransientBytes = 2 * kMiB;
  TrackingAllocator allocator;
  auto backend = std::make_unique<FakeLayerBackend>();
  backend->set_on_setup(
      [&allocator] { allocator.RecordAlloc(kRetainedBytes); });
  backend->set_on_execute([&allocator] {
    allocator.RecordAlloc(kTransientBytes);
    allocator.RecordFree(kTransientBytes);
  });

  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   OptionsMeasuring(allocator));

  ASSERT_OK(runner.status());
  EXPECT_DOUBLE_EQ((*runner)->AllocatedDeltaMB(), 4.0);
  EXPECT_DOUBLE_EQ((*runner)->PeakAllocatedMB(), 6.0);
}

TEST(LayerBenchmarkRunnerTest, MemoryTelemetryExcludesPassesAfterWarmup) {
  constexpr size_t kBytesPerPass = 2 * kMiB;
  TrackingAllocator allocator;
  auto backend = std::make_unique<FakeLayerBackend>();
  backend->set_on_execute(
      [&allocator] { allocator.RecordAlloc(kBytesPerPass); });
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   OptionsMeasuring(allocator));
  ASSERT_OK(runner.status());

  ASSERT_OK((*runner)->ExecuteOnce());

  // Only the warm-up pass falls inside the measured window.
  EXPECT_DOUBLE_EQ((*runner)->AllocatedDeltaMB(), 2.0);
  EXPECT_DOUBLE_EQ((*runner)->PeakAllocatedMB(), 2.0);
}

TEST(LayerBenchmarkRunnerTest, MemoryTelemetryReportsHeapGrowthDuringSetup) {
  // The runner samples the heap once before setup and once after warm-up.
  size_t heap_bytes = 0;
  TrackingAllocator allocator;
  auto backend = std::make_unique<FakeLayerBackend>();
  backend->set_on_setup([&heap_bytes] { heap_bytes = 3 * kMiB; });
  BenchmarkOptions options;
  options.tracking_allocator = &allocator;
  options.heap_sampler = [&heap_bytes] { return heap_bytes; };

  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   options);

  ASSERT_OK(runner.status());
  EXPECT_DOUBLE_EQ((*runner)->AllocatedDeltaMB(), 3.0);
  EXPECT_DOUBLE_EQ((*runner)->PeakAllocatedMB(), 3.0);
}

TEST(LayerBenchmarkRunnerTest, MemoryTelemetryTracksLitertRuntimeAllocations) {
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::make_unique<LitertTensorBackend>(),
                                   CreateTestSpec(LayerType::kGatedMlp),
                                   OptionsWithIdleHeap());

  // With an idle heap, positive telemetry can only come from the runtime
  // that setup built through the process-wide tracking allocator and kept
  // alive.
  ASSERT_OK(runner.status());
  EXPECT_GT((*runner)->AllocatedDeltaMB(), 0.0);
  EXPECT_GT((*runner)->PeakAllocatedMB(), 0.0);
}

// Collects the runs reported by the benchmark library.
class RunCollector : public benchmark::BenchmarkReporter {
 public:
  bool ReportContext(const Context&) override { return true; }

  void ReportRuns(const std::vector<Run>& runs) override {
    runs_.insert(runs_.end(), runs.begin(), runs.end());
  }

  const std::vector<Run>& runs() const { return runs_; }

 private:
  std::vector<Run> runs_;
};

// Result of driving LayerBenchmarkRunner::RunBenchmark() through the benchmark
// library.
struct BenchmarkOutcome {
  absl::Status status;  // Returned by RunBenchmark().
  std::vector<benchmark::BenchmarkReporter::Run> runs;
};

// Registers `benchmark_function` as a benchmark, runs it for exactly
// `iterations` timed iterations through the benchmark library, unregisters it,
// and returns the reported runs.
std::vector<benchmark::BenchmarkReporter::Run> RunThroughBenchmarkLibrary(
    std::function<void(benchmark::State&)> benchmark_function, int iterations) {
  // The benchmark library keeps a pointer to the program name it is
  // initialized with, so the name needs static storage.
  static char program_name[] = "benchmark_runner_test";
  int argc = 1;
  char* argv[] = {program_name, nullptr};
  benchmark::Initialize(&argc, argv);

  benchmark::RegisterBenchmark("BM_LayerUnderTest",
                               std::move(benchmark_function))
      ->Iterations(iterations);
  RunCollector collector;
  benchmark::RunSpecifiedBenchmarks(&collector, "BM_LayerUnderTest");
  benchmark::ClearRegisteredBenchmarks();
  return collector.runs();
}

// Runs `runner` as a benchmark of exactly `iterations` timed iterations.
BenchmarkOutcome RunAsBenchmark(LayerBenchmarkRunner& runner, int iterations) {
  BenchmarkOutcome outcome;
  outcome.runs = RunThroughBenchmarkLibrary(
      [&runner, &outcome](benchmark::State& state) {
        outcome.status = runner.RunBenchmark(state);
      },
      iterations);
  return outcome;
}

TEST(RunBenchmarkTest, ReportsRooflineCountersForSpec) {
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::make_unique<FakeLayerBackend>(),
                                   CreateTestSpec(LayerType::kRmsNorm),
                                   BenchmarkOptions{});
  ASSERT_OK(runner.status());

  BenchmarkOutcome outcome = RunAsBenchmark(**runner, /*iterations=*/5);

  // RmsNorm with B=1, T=8, M=64: 4*T*M = 2048 FLOPs over 4 * (2*T*M + M) =
  // 4352 bytes.
  ASSERT_OK(outcome.status);
  ASSERT_THAT(outcome.runs, SizeIs(1));
  EXPECT_THAT(
      outcome.runs.front().counters,
      IsSupersetOf({
          CounterNamed("Theoretical_FLOPs", DoubleEq(2048.0)),
          CounterNamed("Theoretical_Bytes", DoubleEq(4352.0)),
          CounterNamed("ArithmeticIntensity", DoubleNear(0.470588235294, 1e-9)),
          CounterNamed("SeqLenQ", DoubleEq(8.0)),
          CounterNamed("SeqLenKV", DoubleEq(8.0)),
          CounterNamed("GFLOP/s", Gt(0.0)),
          CounterNamed("GB/s", Gt(0.0)),
      }));
}

TEST(RunBenchmarkTest, ReportsDecodeSequenceLengths) {
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(
          std::make_unique<FakeLayerBackend>(),
          CreateTestSpec(LayerType::kStandardAttention, AttentionMode::kDecode),
          BenchmarkOptions{});
  ASSERT_OK(runner.status());

  BenchmarkOutcome outcome = RunAsBenchmark(**runner, /*iterations=*/5);

  ASSERT_OK(outcome.status);
  ASSERT_THAT(outcome.runs, SizeIs(1));
  EXPECT_THAT(outcome.runs.front().counters,
              IsSupersetOf({
                  CounterNamed("SeqLenQ", DoubleEq(1.0)),
                  CounterNamed("SeqLenKV", DoubleEq(8.0)),
              }));
}

TEST(RunBenchmarkTest, ReportsSetupMemoryTelemetryAsCounters) {
  constexpr size_t kRetainedBytes = 4 * kMiB;
  constexpr size_t kTransientBytes = 2 * kMiB;
  TrackingAllocator allocator;
  auto backend = std::make_unique<FakeLayerBackend>();
  backend->set_on_setup(
      [&allocator] { allocator.RecordAlloc(kRetainedBytes); });
  backend->set_on_execute([&allocator] {
    allocator.RecordAlloc(kTransientBytes);
    allocator.RecordFree(kTransientBytes);
  });
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   OptionsMeasuring(allocator));
  ASSERT_OK(runner.status());

  BenchmarkOutcome outcome = RunAsBenchmark(**runner, /*iterations=*/5);

  ASSERT_OK(outcome.status);
  ASSERT_THAT(outcome.runs, SizeIs(1));
  EXPECT_THAT(outcome.runs.front().counters,
              IsSupersetOf({
                  CounterNamed("AllocDelta_MB", DoubleEq(4.0)),
                  CounterNamed("PeakMem_MB", DoubleEq(6.0)),
              }));
}

TEST(RunBenchmarkTest, RunsOneForwardPassPerIterationWithoutCacheWiping) {
  auto backend = std::make_unique<FakeLayerBackend>();
  FakeLayerBackend* fake = backend.get();
  BenchmarkOptions options;
  options.wipe_caches = false;
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   options);
  ASSERT_OK(runner.status());

  BenchmarkOutcome outcome = RunAsBenchmark(**runner, /*iterations=*/3);

  ASSERT_OK(outcome.status);
  ASSERT_THAT(outcome.runs, SizeIs(1));
  EXPECT_EQ(outcome.runs.front().skipped, benchmark::internal::NotSkipped);
  EXPECT_EQ(outcome.runs.front().iterations, 3);
  // One warm-up pass during Create() plus one pass per timed iteration.
  EXPECT_EQ(fake->execute_count(), 4);
}

TEST(RunBenchmarkTest, CompletesRunWhenWipingCachesOnCallingThread) {
  auto backend = std::make_unique<FakeLayerBackend>();
  FakeLayerBackend* fake = backend.get();
  BenchmarkOptions options;
  options.num_threads = 1;
  options.wipe_caches = true;
  options.wipe_buffer_bytes = kTestWipeBufferBytes;
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   options);
  ASSERT_OK(runner.status());

  BenchmarkOutcome outcome = RunAsBenchmark(**runner, /*iterations=*/3);

  ASSERT_OK(outcome.status);
  ASSERT_THAT(outcome.runs, SizeIs(1));
  EXPECT_EQ(outcome.runs.front().skipped, benchmark::internal::NotSkipped);
  EXPECT_EQ(outcome.runs.front().iterations, 3);
  EXPECT_EQ(fake->execute_count(), 4);
}

TEST(RunBenchmarkTest, CompletesRunWhenWipingCachesOnFallbackThreadpool) {
  // FakeLayerBackend exposes no threadpool, so a multithreaded runner must
  // provision its own workers to evict per-core caches.
  auto backend = std::make_unique<FakeLayerBackend>();
  FakeLayerBackend* fake = backend.get();
  BenchmarkOptions options;
  options.num_threads = 4;
  options.wipe_caches = true;
  options.wipe_buffer_bytes = kTestWipeBufferBytes;
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   options);
  ASSERT_OK(runner.status());

  BenchmarkOutcome outcome = RunAsBenchmark(**runner, /*iterations=*/3);

  ASSERT_OK(outcome.status);
  ASSERT_THAT(outcome.runs, SizeIs(1));
  EXPECT_EQ(outcome.runs.front().skipped, benchmark::internal::NotSkipped);
  EXPECT_EQ(outcome.runs.front().iterations, 3);
  EXPECT_EQ(fake->execute_count(), 4);
}

TEST(RunBenchmarkTest, CompletesRunWhenWipingCachesOnBackendThreadpool) {
  BenchmarkOptions options;
  options.num_threads = 2;
  options.wipe_caches = true;
  options.wipe_buffer_bytes = kTestWipeBufferBytes;
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(
          std::make_unique<LitertTensorBackend>(),
          CreateTestSpec(LayerType::kStandardAttention, AttentionMode::kDecode),
          options);
  ASSERT_OK(runner.status());

  BenchmarkOutcome outcome = RunAsBenchmark(**runner, /*iterations=*/5);

  ASSERT_OK(outcome.status);
  ASSERT_THAT(outcome.runs, SizeIs(1));
  EXPECT_EQ(outcome.runs.front().skipped, benchmark::internal::NotSkipped);
  EXPECT_EQ(outcome.runs.front().iterations, 5);
}

TEST(RunBenchmarkTest, SkipsRunAndReturnsBackendErrorWhenForwardPassFails) {
  auto backend = std::make_unique<FakeLayerBackend>();
  // The warm-up pass and the first timed iteration succeed; the second timed
  // iteration fails.
  backend->FailExecuteWith(absl::DataLossError("forward pass failed"),
                           /*successful_executions=*/2);
  absl::StatusOr<std::unique_ptr<LayerBenchmarkRunner>> runner =
      LayerBenchmarkRunner::Create(std::move(backend), CreateTestSpec(),
                                   BenchmarkOptions{});
  ASSERT_OK(runner.status());

  BenchmarkOutcome outcome = RunAsBenchmark(**runner, /*iterations=*/5);

  EXPECT_EQ(outcome.status.code(), absl::StatusCode::kDataLoss)
      << outcome.status;
  ASSERT_THAT(outcome.runs, SizeIs(1));
  EXPECT_EQ(outcome.runs.front().skipped,
            benchmark::internal::SkippedWithError);
  EXPECT_THAT(outcome.runs.front().skip_message,
              HasSubstr("forward pass failed"));
}

// Name of the benchmark case the RunLayerBenchmarkCase tests run.
constexpr absl::string_view kCaseName =
    "BM_RmsNorm/TestModel/B:1/S:8/Threads:1";

// Timed iterations of every RunLayerBenchmarkCase test. With the warm-up pass
// a fake backend then reports an output tensor of fours.
constexpr int kCaseIterations = 3;

// Returns the path of `file_name` inside the test's temporary directory.
std::string TempPath(absl::string_view file_name) {
  return absl::StrCat(testing::TempDir(), "/", file_name);
}

// Returns a factory of fake backends that run any layer.
BackendFactory FakeBackendFactory() {
  return []() -> std::unique_ptr<LayerBackend> {
    return std::make_unique<FakeLayerBackend>();
  };
}

// Runs the RmsNorm test layer, whose output shape is [B=1, T=8, M=64], as the
// benchmark case `case_name` on backends from `backend_factory`.
std::vector<benchmark::BenchmarkReporter::Run> RunCase(
    absl::string_view case_name, const ParityOptions& parity,
    SingleCaseParityGuard& parity_guard,
    const BackendFactory& backend_factory = FakeBackendFactory()) {
  const LayerSpec spec = CreateTestSpec(LayerType::kRmsNorm);
  return RunThroughBenchmarkLibrary(
      [&](benchmark::State& state) {
        RunLayerBenchmarkCase(state, case_name, spec, backend_factory,
                              BenchmarkOptions{}, parity, parity_guard);
      },
      kCaseIterations);
}

TEST(RunLayerBenchmarkCaseTest, CompletesCaseWithoutParityFiles) {
  SingleCaseParityGuard parity_guard;

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, ParityOptions{}, parity_guard);

  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::NotSkipped);
  EXPECT_EQ(runs.front().iterations, 3);
}

TEST(RunLayerBenchmarkCaseTest, DumpsLayerOutputWithSpecShape) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.dump_path = TempPath("case_output.lmlb");

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard);

  absl::StatusOr<std::pair<std::vector<int64_t>, std::vector<float>>> dump =
      LoadTensorFromFile(parity.dump_path);
  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::NotSkipped);
  ASSERT_OK(dump.status());
  EXPECT_THAT(*dump,
              Pair(ElementsAre(1, 8, 64), AllOf(SizeIs(512), Each(Eq(4.0f)))));
}

TEST(RunLayerBenchmarkCaseTest, ReportsParityCountersForMatchingReference) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.compare_path = TempPath("matching_reference.lmlb");
  ASSERT_OK(DumpTensorToFile(parity.compare_path, std::vector<float>(512, 4.0f),
                             {1, 8, 64}));

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard);

  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::NotSkipped);
  EXPECT_THAT(runs.front().counters,
              IsSupersetOf({
                  CounterNamed("Parity_Passed", DoubleEq(1.0)),
                  CounterNamed("Parity_Mismatches", DoubleEq(0.0)),
                  CounterNamed("Parity_MaxAbsErr", DoubleEq(0.0)),
                  CounterNamed("Parity_MaxRelErr", DoubleEq(0.0)),
                  CounterNamed("Parity_CosineSim", DoubleEq(1.0)),
                  CounterNamed("Parity_L2RelErr", DoubleEq(0.0)),
              }));
}

TEST(RunLayerBenchmarkCaseTest, OutputDifferingFromReferenceSkipsWithError) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.compare_path = TempPath("differing_reference.lmlb");
  ASSERT_OK(DumpTensorToFile(parity.compare_path, std::vector<float>(512, 5.0f),
                             {1, 8, 64}));

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard);

  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::SkippedWithError);
  EXPECT_THAT(runs.front().skip_message,
              HasSubstr("512/512 elements mismatched"));
}

TEST(RunLayerBenchmarkCaseTest, RelativeToleranceAdmitsOutputWithinIt) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.compare_path = TempPath("relative_tolerance_reference.lmlb");
  parity.rtol = 0.1f;
  ASSERT_OK(DumpTensorToFile(parity.compare_path, std::vector<float>(512, 4.2f),
                             {1, 8, 64}));

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard);

  // Each four is 0.2 below its reference of 4.2, within 0.1 * 4.2 = 0.42.
  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::NotSkipped);
  EXPECT_THAT(runs.front().counters,
              Contains(CounterNamed("Parity_Mismatches", DoubleEq(0.0))));
}

TEST(RunLayerBenchmarkCaseTest, AbsoluteToleranceAdmitsOutputWithinIt) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.compare_path = TempPath("absolute_tolerance_reference.lmlb");
  parity.rtol = 0.0f;
  parity.atol = 0.5f;
  ASSERT_OK(DumpTensorToFile(parity.compare_path, std::vector<float>(512, 4.2f),
                             {1, 8, 64}));

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard);

  // Each four is 0.2 below its reference of 4.2, within 0.5.
  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::NotSkipped);
  EXPECT_THAT(runs.front().counters,
              Contains(CounterNamed("Parity_Mismatches", DoubleEq(0.0))));
}

TEST(RunLayerBenchmarkCaseTest, DistributionGatesAdmitOutputWithinThem) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.compare_path = TempPath("drifted_reference.lmlb");
  parity.min_cosine = 0.99;
  parity.max_l2_rel_err = 0.05;
  ASSERT_OK(DumpTensorToFile(parity.compare_path, std::vector<float>(512, 4.2f),
                             {1, 8, 64}));

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard);

  // An output of fours is parallel to a reference of 4.2s and 0.2 / 4.2 =
  // 0.047619 away from it in relative L2 norm, so it passes both gates even
  // though no element is within the elementwise tolerances.
  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::NotSkipped);
  EXPECT_THAT(runs.front().counters,
              IsSupersetOf({
                  CounterNamed("Parity_Passed", DoubleEq(1.0)),
                  CounterNamed("Parity_Mismatches", DoubleEq(512.0)),
                  CounterNamed("Parity_CosineSim", DoubleNear(1.0, 1e-9)),
                  CounterNamed("Parity_L2RelErr", DoubleNear(0.047619, 1e-6)),
              }));
}

TEST(RunLayerBenchmarkCaseTest, OutputBeyondRelativeL2GateSkipsWithError) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.compare_path = TempPath("distant_reference.lmlb");
  parity.max_l2_rel_err = 0.01;
  ASSERT_OK(DumpTensorToFile(parity.compare_path, std::vector<float>(512, 4.2f),
                             {1, 8, 64}));

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard);

  // The output is 0.2 / 4.2 = 0.047619 away from the reference.
  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::SkippedWithError);
  EXPECT_THAT(runs.front().skip_message,
              HasSubstr("l2_rel_err=0.047619 (max=0.01)"));
}

TEST(RunLayerBenchmarkCaseTest, OutputBelowCosineGateSkipsWithError) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.compare_path = TempPath("skewed_reference.lmlb");
  parity.min_cosine = 0.9;
  std::vector<float> reference(256, 8.0f);
  reference.resize(512, 0.0f);
  ASSERT_OK(DumpTensorToFile(parity.compare_path, reference, {1, 8, 64}));

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard);

  // An output of fours meets a reference of 256 eights and 256 zeros at 45
  // degrees: cosine similarity 1 / sqrt(2).
  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::SkippedWithError);
  EXPECT_THAT(runs.front().skip_message,
              HasSubstr("cosine_sim=0.70710678 (min=0.90000000)"));
}

TEST(RunLayerBenchmarkCaseTest, ReferenceOfDifferentShapeSkipsWithError) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.compare_path = TempPath("reshaped_reference.lmlb");
  ASSERT_OK(DumpTensorToFile(parity.compare_path, std::vector<float>(512, 4.0f),
                             {1, 512}));

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard);

  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::SkippedWithError);
  EXPECT_THAT(runs.front().skip_message,
              HasSubstr("reference [1,512] vs candidate [1,8,64]"));
}

TEST(RunLayerBenchmarkCaseTest, MissingReferenceSkipsWithError) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.compare_path = TempPath("absent_reference.lmlb");

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard);

  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::SkippedWithError);
}

TEST(RunLayerBenchmarkCaseTest, UnwritableDumpPathSkipsWithError) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.dump_path = TempPath("absent_directory/case_output.lmlb");

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard);

  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::SkippedWithError);
}

TEST(RunLayerBenchmarkCaseTest, SameCaseMayRunAgainWithItsParityFiles) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.dump_path = TempPath("repeated_case_output.lmlb");
  RunCase(kCaseName, parity, parity_guard);

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard);

  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::NotSkipped);
}

TEST(RunLayerBenchmarkCaseTest, SecondCaseIsRefusedTheParityFiles) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.dump_path = TempPath("contested_case_output.lmlb");
  RunCase(kCaseName, parity, parity_guard);

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase("BM_RmsNorm/TestModel/B:1/S:8/Threads:4", parity, parity_guard);

  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::SkippedWithError);
  EXPECT_THAT(runs.front().skip_message, HasSubstr(kCaseName));
}

TEST(RunLayerBenchmarkCaseTest, BackendSetupFailureSkipsWithError) {
  SingleCaseParityGuard parity_guard;
  BackendFactory failing_factory = []() -> std::unique_ptr<LayerBackend> {
    auto backend = std::make_unique<FakeLayerBackend>();
    backend->FailSetupWith(absl::ResourceExhaustedError("setup failed"));
    return backend;
  };

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, ParityOptions{}, parity_guard, failing_factory);

  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::SkippedWithError);
  EXPECT_THAT(runs.front().skip_message, HasSubstr("setup failed"));
}

TEST(RunLayerBenchmarkCaseTest, FailedForwardPassLeavesNoDump) {
  SingleCaseParityGuard parity_guard;
  ParityOptions parity;
  parity.dump_path = TempPath("failed_case_output.lmlb");
  BackendFactory failing_factory = []() -> std::unique_ptr<LayerBackend> {
    auto backend = std::make_unique<FakeLayerBackend>();
    // The warm-up pass and the first timed iteration succeed.
    backend->FailExecuteWith(absl::DataLossError("forward pass failed"),
                             /*successful_executions=*/2);
    return backend;
  };

  std::vector<benchmark::BenchmarkReporter::Run> runs =
      RunCase(kCaseName, parity, parity_guard, failing_factory);

  ASSERT_THAT(runs, SizeIs(1));
  EXPECT_EQ(runs.front().skipped, benchmark::internal::SkippedWithError);
  EXPECT_EQ(LoadTensorFromFile(parity.dump_path).status().code(),
            absl::StatusCode::kNotFound);
}

}  // namespace
}  // namespace layer_ml_bench
