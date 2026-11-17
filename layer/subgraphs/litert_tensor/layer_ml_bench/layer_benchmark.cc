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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "tensor/benchmarks/layer/benchmark_runner.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/litert_tensor_backend.h"
#include "tensor/benchmarks/layer/parity_validator.h"
#include "tensor/benchmarks/layer/suite_loader.h"

ABSL_FLAG(bool, wipe_caches, true,
          "Wipe L2/LLC caches before each timed iteration.");
ABSL_FLAG(uint32_t, seed, 1234,
          "Deterministic PRNG seed for synthetic inputs and weights.");
ABSL_FLAG(std::string, backend, "",
          "Execution backend: xnnpack or ynnpack (auto-detected from argv[0] "
          "if empty).");
ABSL_FLAG(std::string, suite_file, "",
          "Optional path to a declarative JSON layer suite file defining "
          "presets, batch_sizes, seq_lengths, thread_counts, and families.");
ABSL_FLAG(std::string, dump_parity_tensors, "",
          "Optional file path to dump output tensor (in LMLB binary format) "
          "for numerical parity verification.");
ABSL_FLAG(std::string, compare_parity_tensors, "",
          "Optional file path to a reference LMLB binary tensor dump to verify "
          "numerical parity against.");
ABSL_FLAG(float, parity_rtol, 1e-4f,
          "Relative tolerance for --compare_parity_tensors.");
ABSL_FLAG(float, parity_atol, 1e-5f,
          "Absolute tolerance for --compare_parity_tensors.");
ABSL_FLAG(double, parity_min_cosine, 0.0,
          "Optional minimum cosine similarity gate when > 0.0.");
ABSL_FLAG(double, parity_max_l2_rel_err, 0.0,
          "Optional maximum relative L2 error gate when > 0.0.");

namespace layer_ml_bench {
namespace {

// Registers one benchmark per case of the suite in `suite_file` (the built-in
// suite when empty), each running on `runtime_backend` with the options the
// command-line flags select.
absl::Status RegisterLayerBenchmarks(absl::string_view suite_file,
                                     litert::RuntimeBackend runtime_backend) {
  absl::StatusOr<SuiteSpec> suite = BuildSuiteSpec(suite_file);
  if (!suite.ok()) {
    return suite.status();
  }

  ParityOptions parity;
  parity.dump_path = absl::GetFlag(FLAGS_dump_parity_tensors);
  parity.compare_path = absl::GetFlag(FLAGS_compare_parity_tensors);
  parity.rtol = absl::GetFlag(FLAGS_parity_rtol);
  parity.atol = absl::GetFlag(FLAGS_parity_atol);
  parity.min_cosine = absl::GetFlag(FLAGS_parity_min_cosine);
  parity.max_l2_rel_err = absl::GetFlag(FLAGS_parity_max_l2_rel_err);

  BackendFactory backend_factory =
      [runtime_backend]() -> std::unique_ptr<LayerBackend> {
    return std::make_unique<litert::LitertTensorBackend>(runtime_backend);
  };
  // Shared by every registered case and kept alive by their closures.
  std::shared_ptr<SingleCaseParityGuard> parity_guard =
      std::make_shared<SingleCaseParityGuard>();

  for (BenchmarkCase& bench_case : ExpandSuite(*suite)) {
    BenchmarkOptions options;
    options.num_threads = bench_case.num_threads;
    options.wipe_caches = absl::GetFlag(FLAGS_wipe_caches);
    options.seed = absl::GetFlag(FLAGS_seed);

    // Copied before `bench_case` moves into the closure below.
    const std::string name = bench_case.name;
    ::benchmark::RegisterBenchmark(
        name,
        [bench_case = std::move(bench_case), backend_factory, options, parity,
         parity_guard](::benchmark::State& state) {
          RunLayerBenchmarkCase(state, bench_case.name, bench_case.spec,
                                backend_factory, options, parity,
                                *parity_guard);
        })
        ->UseRealTime()
        ->Unit(::benchmark::kMillisecond);
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace layer_ml_bench

int main(int argc, char** argv) {
  const std::string argv0 = (argc > 0 && argv[0] != nullptr) ? argv[0] : "";
  ::benchmark::Initialize(&argc, argv);
  absl::ParseCommandLine(argc, argv);

  absl::StatusOr<layer_ml_bench::litert::RuntimeBackend> runtime_backend =
      layer_ml_bench::ParseRuntimeBackend(absl::GetFlag(FLAGS_backend), argv0);
  if (!runtime_backend.ok()) {
    ABSL_LOG(ERROR) << runtime_backend.status().message();
    return EXIT_FAILURE;
  }

  absl::Status status = layer_ml_bench::RegisterLayerBenchmarks(
      absl::GetFlag(FLAGS_suite_file), *runtime_backend);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to register layer benchmarks: " << status;
    return EXIT_FAILURE;
  }
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return EXIT_SUCCESS;
}
