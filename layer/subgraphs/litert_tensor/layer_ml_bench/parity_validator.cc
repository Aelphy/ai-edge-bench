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

#include "tensor/benchmarks/layer/parity_validator.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_spec.h"

namespace layer_ml_bench {

ParityReport CompareTensors(absl::Span<const float> reference,
                            absl::Span<const float> candidate, float rtol,
                            float atol) {
  ParityReport report;
  report.total_elements = reference.size();

  if (rtol < 0.0f || atol < 0.0f || !std::isfinite(rtol) ||
      !std::isfinite(atol)) {
    report.passed = false;
    report.max_absolute_error = std::numeric_limits<double>::infinity();
    report.max_relative_error = std::numeric_limits<double>::infinity();
    report.cosine_similarity = std::numeric_limits<double>::quiet_NaN();
    report.l2_relative_error = std::numeric_limits<double>::infinity();
    report.mismatch_count =
        std::max<size_t>(1, std::max(reference.size(), candidate.size()));
    report.details = absl::StrFormat(
        "Invalid tolerance parameters: rtol=%g, atol=%g (must be non-negative "
        "and finite)",
        rtol, atol);
    return report;
  }

  if (reference.size() != candidate.size()) {
    report.passed = false;
    report.max_absolute_error = std::numeric_limits<double>::infinity();
    report.max_relative_error = std::numeric_limits<double>::infinity();
    report.cosine_similarity = std::numeric_limits<double>::quiet_NaN();
    report.l2_relative_error = std::numeric_limits<double>::infinity();
    report.mismatch_count =
        std::max<size_t>(1, std::max(reference.size(), candidate.size()));
    report.details = absl::StrFormat(
        "Tensor size mismatch: reference size %zu vs candidate size %zu",
        reference.size(), candidate.size());
    return report;
  }

  if (reference.empty()) {
    report.passed = true;
    report.details = "Both tensors are empty; parity check passed.";
    return report;
  }

  report.passed = true;
  report.mismatch_count = 0;
  report.max_absolute_error = 0.0;
  report.max_relative_error = 0.0;
  report.cosine_similarity = 1.0;
  report.l2_relative_error = 0.0;

  double dot_rc = 0.0;
  double norm_r_sq = 0.0;
  double norm_c_sq = 0.0;
  double diff_sq = 0.0;
  bool has_non_finite = false;

  std::vector<std::string> sample_mismatches;
  constexpr size_t kMaxSampleMismatches = 5;

  for (size_t i = 0; i < reference.size(); ++i) {
    const float r = reference[i];
    const float c = candidate[i];

    const bool r_finite = std::isfinite(r);
    const bool c_finite = std::isfinite(c);

    if (!r_finite || !c_finite) {
      has_non_finite = true;
      report.passed = false;
      report.mismatch_count++;
      report.max_absolute_error = std::numeric_limits<double>::infinity();
      report.max_relative_error = std::numeric_limits<double>::infinity();
      if (sample_mismatches.size() < kMaxSampleMismatches) {
        sample_mismatches.push_back(absl::StrFormat(
            "idx %zu: non-finite value (ref=%f, cand=%f)", i, r, c));
      }
      continue;
    }

    const double rd = static_cast<double>(r);
    const double cd = static_cast<double>(c);
    const double diff = rd - cd;
    dot_rc += rd * cd;
    norm_r_sq += rd * rd;
    norm_c_sq += cd * cd;
    diff_sq += diff * diff;

    const double abs_err = std::abs(diff);
    const double ref_mag = std::abs(rd);
    const double rel_err = (ref_mag > 1e-12) ? (abs_err / ref_mag) : 0.0;

    if (abs_err > report.max_absolute_error) {
      report.max_absolute_error = abs_err;
    }
    if (rel_err > report.max_relative_error) {
      report.max_relative_error = rel_err;
    }

    const double allowed_tol =
        static_cast<double>(atol) + static_cast<double>(rtol) * ref_mag;
    if (abs_err > allowed_tol) {
      report.passed = false;
      report.mismatch_count++;
      if (sample_mismatches.size() < kMaxSampleMismatches) {
        sample_mismatches.push_back(absl::StrFormat(
            "idx %zu: ref=%g, cand=%g, abs_err=%g > tol=%g (rel_err=%g)", i, r,
            c, abs_err, allowed_tol, rel_err));
      }
    }
  }

  if (has_non_finite) {
    report.cosine_similarity = std::numeric_limits<double>::quiet_NaN();
    report.l2_relative_error = std::numeric_limits<double>::infinity();
  } else {
    const double norm_r = std::sqrt(norm_r_sq);
    const double norm_c = std::sqrt(norm_c_sq);
    const double diff_norm = std::sqrt(diff_sq);
    if (norm_r == 0.0 && norm_c == 0.0) {
      report.cosine_similarity = 1.0;
    } else if (norm_r == 0.0 || norm_c == 0.0) {
      report.cosine_similarity = 0.0;
    } else {
      report.cosine_similarity =
          std::clamp(dot_rc / (norm_r * norm_c), -1.0, 1.0);
    }
    if (norm_r == 0.0) {
      report.l2_relative_error =
          (diff_norm == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
    } else {
      report.l2_relative_error = diff_norm / norm_r;
    }
  }

  if (report.passed) {
    report.details = absl::StrFormat(
        "Parity PASSED: %zu elements compared. Max abs error: %g (atol=%g), "
        "Max rel error: %g (rtol=%g), cosine_sim: %.8f, l2_rel_err: %g.",
        report.total_elements, report.max_absolute_error, atol,
        report.max_relative_error, rtol, report.cosine_similarity,
        report.l2_relative_error);
  } else {
    std::string sample_str;
    for (const std::string& sample : sample_mismatches) {
      if (!sample_str.empty()) sample_str.append("; ");
      sample_str.append(sample);
    }
    report.details = absl::StrFormat(
        "Parity FAILED: %zu/%zu elements mismatched. Max abs error: %g "
        "(atol=%g), Max rel error: %g (rtol=%g), cosine_sim: %.8f, "
        "l2_rel_err: %g. Sample mismatches: [%s]",
        report.mismatch_count, report.total_elements, report.max_absolute_error,
        atol, report.max_relative_error, rtol, report.cosine_similarity,
        report.l2_relative_error, sample_str);
  }

  return report;
}

ParityReport CompareWithReference(absl::Span<const int64_t> reference_shape,
                                  absl::Span<const float> reference,
                                  absl::Span<const int64_t> candidate_shape,
                                  absl::Span<const float> candidate, float rtol,
                                  float atol) {
  if (reference_shape == candidate_shape) {
    return CompareTensors(reference, candidate, rtol, atol);
  }
  ParityReport report;
  report.passed = false;
  report.total_elements = reference.size();
  report.mismatch_count =
      std::max<size_t>(1, std::max(reference.size(), candidate.size()));
  report.max_absolute_error = std::numeric_limits<double>::infinity();
  report.max_relative_error = std::numeric_limits<double>::infinity();
  report.cosine_similarity = std::numeric_limits<double>::quiet_NaN();
  report.l2_relative_error = std::numeric_limits<double>::infinity();
  report.details = absl::StrFormat(
      "Tensor shape mismatch: reference [%s] vs candidate [%s]",
      absl::StrJoin(reference_shape, ","), absl::StrJoin(candidate_shape, ","));
  return report;
}

ParityGateResult EvaluateParityGate(const ParityReport& report,
                                    double min_cosine, double max_l2_rel_err) {
  if (min_cosine <= 0.0 && max_l2_rel_err <= 0.0) {
    return ParityGateResult{report.passed, report.details};
  }
  const bool metrics_finite = std::isfinite(report.cosine_similarity) &&
                              std::isfinite(report.l2_relative_error);
  const bool cos_ok =
      (min_cosine <= 0.0) || (report.cosine_similarity >= min_cosine);
  const bool l2_ok =
      (max_l2_rel_err <= 0.0) || (report.l2_relative_error <= max_l2_rel_err);
  const bool gate_passed = metrics_finite && cos_ok && l2_ok;
  if (gate_passed) {
    return ParityGateResult{true, report.details};
  }
  return ParityGateResult{
      false, absl::StrFormat(
                 "Subgraph quality gate FAILED: cosine_sim=%.8f (min=%.8f), "
                 "l2_rel_err=%g (max=%g). %s",
                 report.cosine_similarity, min_cosine, report.l2_relative_error,
                 max_l2_rel_err, report.details)};
}

absl::Status DumpTensorToFile(absl::string_view path,
                              absl::Span<const float> tensor_data,
                              absl::Span<const int64_t> shape) {
  if (shape.empty()) {
    return absl::InvalidArgumentError("Tensor shape rank must be at least 1");
  }
  if (shape.size() > 16) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Shape rank %zu exceeds maximum supported rank of 16", shape.size()));
  }

  int64_t num_elements = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Shape dimension [%zu] = %d must be positive", i, shape[i]));
    }
    if (num_elements > std::numeric_limits<int64_t>::max() / shape[i]) {
      return absl::InvalidArgumentError(
          "Shape total element count overflows int64");
    }
    num_elements *= shape[i];
  }

  if (static_cast<size_t>(num_elements) != tensor_data.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Tensor data size %zu does not match shape product %d",
                        tensor_data.size(), num_elements));
  }

  std::string path_str(path);
  std::ofstream out(path_str, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Failed to open file for writing: %s", path));
  }

  const uint32_t magic = kParityDumpMagic;
  const uint32_t rank = static_cast<uint32_t>(shape.size());

  out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
  out.write(reinterpret_cast<const char*>(shape.data()),
            shape.size() * sizeof(int64_t));
  if (!tensor_data.empty()) {
    out.write(reinterpret_cast<const char*>(tensor_data.data()),
              tensor_data.size() * sizeof(float));
  }

  out.close();
  if (out.fail()) {
    return absl::InternalError(
        absl::StrFormat("Failed while writing parity dump to %s", path));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::pair<std::vector<int64_t>, std::vector<float>>>
LoadTensorFromFile(absl::string_view path) {
  std::string path_str(path);
  std::ifstream in(path_str, std::ios::binary);
  if (!in.is_open()) {
    return absl::NotFoundError(
        absl::StrFormat("Failed to open parity dump file: %s", path));
  }

  in.seekg(0, std::ios::end);
  const std::streampos file_size_pos = in.tellg();
  in.seekg(0, std::ios::beg);

  if (file_size_pos < 0) {
    return absl::InternalError(
        absl::StrFormat("Failed to determine file size: %s", path));
  }
  const size_t file_size = static_cast<size_t>(file_size_pos);

  constexpr size_t kHeaderSize = sizeof(uint32_t) * 2;
  if (file_size < kHeaderSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "File %s is too short for parity dump header (size %zu bytes)", path,
        file_size));
  }

  uint32_t magic = 0;
  uint32_t rank = 0;
  in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char*>(&rank), sizeof(rank));
  if (!in.good()) {
    return absl::InternalError(
        absl::StrFormat("Failed to read header from %s", path));
  }

  if (magic != kParityDumpMagic) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid parity dump magic 0x%08X (expected 0x%08X) in %s", magic,
        kParityDumpMagic, path));
  }

  if (rank == 0 || rank > 16) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid rank %u in parity dump %s (must be 1..16)", rank, path));
  }

  const size_t shape_bytes = static_cast<size_t>(rank) * sizeof(int64_t);
  if (file_size < kHeaderSize + shape_bytes) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "File %s truncated: expected at least %zu bytes for shape, got %zu",
        path, kHeaderSize + shape_bytes, file_size));
  }

  std::vector<int64_t> shape(rank);
  in.read(reinterpret_cast<char*>(shape.data()), shape_bytes);
  if (!in.good()) {
    return absl::InternalError(
        absl::StrFormat("Failed reading shape from %s", path));
  }

  int64_t num_elements = 1;
  for (uint32_t i = 0; i < rank; ++i) {
    if (shape[i] <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid non-positive dimension [%u] = %d in %s", i, shape[i], path));
    }
    if (num_elements > std::numeric_limits<int64_t>::max() / shape[i]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor shape element count overflows int64 in %s", path));
    }
    num_elements *= shape[i];
  }

  // Bounding the element count by the bytes left after the header and shape
  // keeps both the payload size and the total size below from wrapping.
  const size_t metadata_bytes = kHeaderSize + shape_bytes;
  if (static_cast<uint64_t>(num_elements) >
      (std::numeric_limits<size_t>::max() - metadata_bytes) / sizeof(float)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Parity dump byte count overflows size_t in %s", path));
  }
  const size_t payload_bytes =
      static_cast<size_t>(num_elements) * sizeof(float);

  const size_t expected_total_bytes = metadata_bytes + payload_bytes;
  if (file_size != expected_total_bytes) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Parity dump file size mismatch in %s: expected %zu bytes, got %zu",
        path, expected_total_bytes, file_size));
  }

  // The exact-size check above rules out both a short payload and trailing
  // bytes, and every dimension is positive, so the payload is non-empty.
  std::vector<float> data(static_cast<size_t>(num_elements));
  in.read(reinterpret_cast<char*>(data.data()), payload_bytes);
  if (!in.good()) {
    return absl::InternalError(
        absl::StrFormat("Failed reading float tensor data from %s", path));
  }

  return std::make_pair(std::move(shape), std::move(data));
}

absl::Status ExecuteAndDumpParity(LayerBackend& backend, const LayerSpec& spec,
                                  uint32_t seed, absl::string_view output_file,
                                  const BackendOptions& options) {
  absl::Status validation_status = spec.Validate();
  if (!validation_status.ok()) {
    return validation_status;
  }

  SeededLayerData data = CreateSeededLayerData(spec, seed);
  absl::Status setup_status = backend.Setup(spec, std::move(data), options);
  if (!setup_status.ok()) {
    return setup_status;
  }

  absl::Status exec_status = backend.Execute();
  if (!exec_status.ok()) {
    return exec_status;
  }

  absl::StatusOr<std::vector<float>> output = backend.GetOutputData();
  if (!output.ok()) {
    return output.status();
  }

  return DumpTensorToFile(output_file, *output, spec.OutputShape());
}

absl::Status SingleCaseParityGuard::Claim(absl::string_view case_name) {
  absl::MutexLock lock(&mutex_);
  if (!claimed_) {
    claimed_ = true;
    owner_ = std::string(case_name);
    return absl::OkStatus();
  }
  if (owner_ == case_name) {
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(absl::StrFormat(
      "parity file is already owned by benchmark case '%s'; refusing '%s'",
      owner_, case_name));
}

}  // namespace layer_ml_bench
