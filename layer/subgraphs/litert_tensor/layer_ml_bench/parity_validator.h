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

#ifndef TENSOR_BENCHMARKS_LAYER_PARITY_VALIDATOR_H_
#define TENSOR_BENCHMARKS_LAYER_PARITY_VALIDATOR_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_spec.h"

namespace layer_ml_bench {

// Magic header for binary parity dumps: 0x4C4D4C42 ("LMLB" - Layer ML Bench).
inline constexpr uint32_t kParityDumpMagic = 0x4C4D4C42;

// Numerical parity validation report for AI agents and human engineers.
struct ParityReport {
  bool passed = false;
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  double cosine_similarity = 1.0;
  double l2_relative_error = 0.0;
  size_t total_elements = 0;
  size_t mismatch_count = 0;
  std::string details;
};

// Outcome of applying subgraph-level parity quality gates to a ParityReport.
struct ParityGateResult {
  bool passed = false;
  std::string details;
};

// Compares two float buffers element-by-element using standard NumPy/PyTorch
// allclose tolerance:
//   |ref[i] - cand[i]| <= atol + rtol * |ref[i]|
// Any NaN or Inf in either buffer, invalid tolerance, or size mismatch sets
// passed = false, cosine_similarity = NaN, and l2_relative_error = +inf.
ParityReport CompareTensors(absl::Span<const float> reference,
                            absl::Span<const float> candidate,
                            float rtol = 1e-4f, float atol = 1e-5f);

// Compares a candidate tensor against a reference tensor of known shape.
// Tensors of equal shape are compared with CompareTensors. Tensors of
// different shapes never pass: the report then carries the non-finite metrics
// of a structural failure and names both shapes in `details`.
ParityReport CompareWithReference(absl::Span<const int64_t> reference_shape,
                                  absl::Span<const float> reference,
                                  absl::Span<const int64_t> candidate_shape,
                                  absl::Span<const float> candidate,
                                  float rtol = 1e-4f, float atol = 1e-5f);

// Evaluates whether `report` satisfies exact elementwise tolerances (when both
// `min_cosine <= 0` and `max_l2_rel_err <= 0`) or subgraph quality gates
// (`cosine_similarity >= min_cosine` and `l2_relative_error <=
// max_l2_rel_err`). Structural or non-finite failures (NaN cosine similarity or
// non-finite L2 error) always fail regardless of relaxed gate thresholds.
ParityGateResult EvaluateParityGate(const ParityReport& report,
                                    double min_cosine = 0.0,
                                    double max_l2_rel_err = 0.0);

// Dumps tensor buffer to file in binary format:
//   - 4-byte uint32 magic: 0x4C4D4C42 ('LMLB')
//   - 4-byte uint32 rank R (number of dimensions in `shape`)
//   - R * 8-byte int64_t dimension values (`shape[0] ... shape[R-1]`)
//   - N * 4-byte IEEE-754 float payload (`tensor_data`), where N = prod(shape).
//
// Python AI Agent Reader Reference:
// ```python
// import struct
// import numpy as np
//
// def read_parity_dump(path: str) -> np.ndarray:
//     with open(path, "rb") as f:
//         magic, rank = struct.unpack("<II", f.read(8))
//         assert magic == 0x4C4D4C42, f"Bad magic: {hex(magic)}"
//         shape = struct.unpack(f"<{rank}q", f.read(rank * 8))
//         data = np.frombuffer(f.read(), dtype=np.float32)
//         return data.reshape(shape)
// ```
absl::Status DumpTensorToFile(absl::string_view path,
                              absl::Span<const float> tensor_data,
                              absl::Span<const int64_t> shape);

// Reads a dumped tensor from file, validating the magic header, rank,
// dimension sizes (rejecting non-positive/overflowing dims and truncated
// or extra trailing bytes), and returning {shape, float_data}.
absl::StatusOr<std::pair<std::vector<int64_t>, std::vector<float>>>
LoadTensorFromFile(absl::string_view path);

// Runs a LayerBackend with deterministic seeded data (`CreateSeededLayerData`)
// for a single execution and dumps its output tensor to `output_file` with
// shape `spec.OutputShape()` (`[B, H, T, D]` for standalone Attention layers,
// `[B, T, OutputDim()]` otherwise).
absl::Status ExecuteAndDumpParity(LayerBackend& backend, const LayerSpec& spec,
                                  uint32_t seed, absl::string_view output_file,
                                  const BackendOptions& options = {});

// Arbitrates which benchmark case may use a single parity dump or reference
// file. One file describes exactly one case's output tensor, so the first
// case to claim it becomes the owner and every other case is refused. The
// owner may claim again any number of times, because benchmark frameworks
// re-invoke one case for iteration-count calibration and repetitions.
// Thread-safe.
class SingleCaseParityGuard {
 public:
  SingleCaseParityGuard() = default;

  // Not copyable or movable.
  SingleCaseParityGuard(const SingleCaseParityGuard&) = delete;
  SingleCaseParityGuard& operator=(const SingleCaseParityGuard&) = delete;

  // Returns OK if `case_name` is the first claimant or already the owner,
  // and FailedPrecondition naming the owner otherwise.
  absl::Status Claim(absl::string_view case_name);

 private:
  absl::Mutex mutex_;
  bool claimed_ ABSL_GUARDED_BY(mutex_) = false;
  std::string owner_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace layer_ml_bench

#endif  // TENSOR_BENCHMARKS_LAYER_PARITY_VALIDATOR_H_
