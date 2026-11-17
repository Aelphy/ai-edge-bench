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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensor/benchmarks/layer/fake_layer_backend.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_spec.h"
#include "tensor/benchmarks/layer/litert_tensor_backend.h"
#include "tensor/benchmarks/layer/parity_validator.h"
#include "tensor/benchmarks/layer/status_test_util.h"
#include "tensor/benchmarks/layer/test_matchers.h"

namespace layer_ml_bench {
namespace {

using ::layer_ml_bench::litert::LitertTensorBackend;
using ::testing::AllOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsNan;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::Values;

// Shape and elements of a tensor read back from a parity dump.
using LoadedTensor = std::pair<std::vector<int64_t>, std::vector<float>>;

constexpr uint32_t kSeed = 42;
constexpr uint32_t kOtherSeed = 99;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

// Names of two distinct benchmark cases.
constexpr absl::string_view kOwnerCase = "BM_RmsNorm/TestModel";
constexpr absl::string_view kOtherCase = "BM_RoPE/TestModel";

// Little-endian bytes of a rank-1 parity dump holding {1.0f, -2.0f}.
constexpr uint8_t kTwoElementDumpBytes[] = {
    0x42, 0x4C, 0x4D, 0x4C,                          // Magic 0x4C4D4C42.
    0x01, 0x00, 0x00, 0x00,                          // Rank 1.
    0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // Dimension 2.
    0x00, 0x00, 0x80, 0x3F,                          // 1.0f.
    0x00, 0x00, 0x00, 0xC0,                          // -2.0f.
};

// Small layer: B=1, S=T=8, H=4, D=16, N_kv=2, F=64, M=H*D=64.
LayerSpec CreateTestSpec(LayerType layer_type, AttentionMode attention_mode =
                                                   AttentionMode::kPrefill) {
  LayerSpec spec;
  spec.model_name = "TestModel";
  spec.layer_type = layer_type;
  spec.attention_mode = attention_mode;
  spec.mlp_activation = MlpActivation::kSwiGLU;
  spec.batch_size = 1;
  spec.seq_len_kv = 8;
  spec.seq_len_q = 8;
  spec.num_heads = 4;
  spec.head_dim = 16;
  spec.num_kv_heads = 2;
  spec.hidden_dim = 64;
  spec.eps = 1e-5f;
  spec.dtype = DataType::kFP32;
  return spec;
}

// Returns the path of `file_name` inside the test's temporary directory.
std::string TempPath(absl::string_view file_name) {
  return absl::StrCat(testing::TempDir(), "/", file_name);
}

// Fields of a parity dump file. They are serialized verbatim, so a test can
// describe a file that violates the format.
struct RawDump {
  uint32_t magic = kParityDumpMagic;
  uint32_t rank = 1;
  std::vector<int64_t> dims = {1};
  std::vector<float> payload = {1.0f};
  std::string trailing_bytes;
};

// Returns the bytes of `dump` in file order: magic, rank, dimensions, payload
// and trailing bytes.
std::string Serialize(const RawDump& dump) {
  std::string bytes;
  bytes.append(reinterpret_cast<const char*>(&dump.magic), sizeof(dump.magic));
  bytes.append(reinterpret_cast<const char*>(&dump.rank), sizeof(dump.rank));
  bytes.append(reinterpret_cast<const char*>(dump.dims.data()),
               dump.dims.size() * sizeof(int64_t));
  bytes.append(reinterpret_cast<const char*>(dump.payload.data()),
               dump.payload.size() * sizeof(float));
  bytes.append(dump.trailing_bytes);
  return bytes;
}

// Writes `contents` to `path`, replacing any existing file.
absl::Status WriteFile(const std::string& path, absl::string_view contents) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  out.close();
  if (out.fail()) {
    return absl::InternalError(absl::StrCat("Failed to write ", path));
  }
  return absl::OkStatus();
}

// Returns the bytes of the file at `path`.
absl::StatusOr<std::vector<uint8_t>> ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return absl::NotFoundError(absl::StrCat("Failed to open ", path));
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in),
                              std::istreambuf_iterator<char>());
}

TEST(CompareTensorsTest, IdenticalTensorsPassWithZeroError) {
  const std::vector<float> values = {1.0f, -2.5f, 3.1415f, 0.0f, -0.001f};

  ParityReport report = CompareTensors(values, values);

  EXPECT_TRUE(report.passed);
  EXPECT_EQ(report.total_elements, 5);
  EXPECT_EQ(report.mismatch_count, 0);
  EXPECT_DOUBLE_EQ(report.max_absolute_error, 0.0);
  EXPECT_DOUBLE_EQ(report.max_relative_error, 0.0);
  EXPECT_DOUBLE_EQ(report.cosine_similarity, 1.0);
  EXPECT_DOUBLE_EQ(report.l2_relative_error, 0.0);
}

TEST(CompareTensorsTest, EmptyTensorsPass) {
  ParityReport report = CompareTensors({}, {});

  EXPECT_TRUE(report.passed);
  EXPECT_EQ(report.total_elements, 0);
  EXPECT_EQ(report.mismatch_count, 0);
}

struct SimilarityCase {
  std::string name;
  std::vector<float> reference;
  std::vector<float> candidate;
  bool expected_passed;
  double expected_cosine;
  double expected_l2_relative_error;
};

class CompareTensorsSimilarityTest
    : public testing::TestWithParam<SimilarityCase> {};

TEST_P(CompareTensorsSimilarityTest, ReportsCosineAndRelativeL2Error) {
  const SimilarityCase& tc = GetParam();

  ParityReport report = CompareTensors(tc.reference, tc.candidate);

  EXPECT_EQ(report.passed, tc.expected_passed);
  EXPECT_DOUBLE_EQ(report.cosine_similarity, tc.expected_cosine);
  EXPECT_DOUBLE_EQ(report.l2_relative_error, tc.expected_l2_relative_error);
}

INSTANTIATE_TEST_SUITE_P(
    VectorPairs, CompareTensorsSimilarityTest,
    Values(
        // ||r - c|| / ||r|| = 5 / 3.
        SimilarityCase{.name = "Orthogonal",
                       .reference = {3.0f, 0.0f},
                       .candidate = {0.0f, 4.0f},
                       .expected_passed = false,
                       .expected_cosine = 0.0,
                       .expected_l2_relative_error = 5.0 / 3.0},
        // c = 1.5 r, so ||r - c|| / ||r|| = 0.5.
        SimilarityCase{.name = "PositivelyScaled",
                       .reference = {2.0f, -4.0f},
                       .candidate = {3.0f, -6.0f},
                       .expected_passed = false,
                       .expected_cosine = 1.0,
                       .expected_l2_relative_error = 0.5},
        // c = -r, so ||r - c|| / ||r|| = 2.
        SimilarityCase{.name = "Negated",
                       .reference = {1.0f, -2.0f, 2.0f},
                       .candidate = {-1.0f, 2.0f, -2.0f},
                       .expected_passed = false,
                       .expected_cosine = -1.0,
                       .expected_l2_relative_error = 2.0},
        SimilarityCase{.name = "BothZero",
                       .reference = {0.0f, 0.0f},
                       .candidate = {0.0f, 0.0f},
                       .expected_passed = true,
                       .expected_cosine = 1.0,
                       .expected_l2_relative_error = 0.0},
        SimilarityCase{.name = "ZeroReference",
                       .reference = {0.0f, 0.0f},
                       .candidate = {1.0f, -1.0f},
                       .expected_passed = false,
                       .expected_cosine = 0.0,
                       .expected_l2_relative_error = kInfinity},
        SimilarityCase{.name = "ZeroCandidate",
                       .reference = {3.0f, 4.0f},
                       .candidate = {0.0f, 0.0f},
                       .expected_passed = false,
                       .expected_cosine = 0.0,
                       .expected_l2_relative_error = 1.0}),
    [](const testing::TestParamInfo<SimilarityCase>& info) {
      return info.param.name;
    });

struct ToleranceCase {
  std::string name;
  std::vector<float> reference;
  std::vector<float> candidate;
  float rtol;
  float atol;
  bool expected_passed;
};

class CompareTensorsToleranceTest
    : public testing::TestWithParam<ToleranceCase> {};

TEST_P(CompareTensorsToleranceTest, AppliesAbsolutePlusRelativeTolerance) {
  const ToleranceCase& tc = GetParam();

  EXPECT_EQ(CompareTensors(tc.reference, tc.candidate, tc.rtol, tc.atol).passed,
            tc.expected_passed);
}

INSTANTIATE_TEST_SUITE_P(
    ToleranceTerms, CompareTensorsToleranceTest,
    Values(
        // Allowed errors 1.1e-4, 1.01e-3 and 1.001e-2 exceed the actual
        // errors 5e-5, 5e-4 and 5e-3.
        ToleranceCase{.name = "WithinCombinedTolerance",
                      .reference = {1.0f, 10.0f, 100.0f},
                      .candidate = {1.00005f, 10.0005f, 100.005f},
                      .rtol = 1e-4f,
                      .atol = 1e-5f,
                      .expected_passed = true},
        ToleranceCase{.name = "AbsoluteToleranceCoversZeroReference",
                      .reference = {0.0f},
                      .candidate = {5e-6f},
                      .rtol = 0.0f,
                      .atol = 1e-5f,
                      .expected_passed = true},
        ToleranceCase{.name = "AbsoluteToleranceExceededAtZeroReference",
                      .reference = {0.0f},
                      .candidate = {2e-5f},
                      .rtol = 0.0f,
                      .atol = 1e-5f,
                      .expected_passed = false},
        // Allowed error 1e-4 * 1000 = 0.1.
        ToleranceCase{.name = "RelativeToleranceScalesWithReference",
                      .reference = {1000.0f},
                      .candidate = {1000.05f},
                      .rtol = 1e-4f,
                      .atol = 0.0f,
                      .expected_passed = true},
        ToleranceCase{.name = "RelativeToleranceExceeded",
                      .reference = {1000.0f},
                      .candidate = {1000.5f},
                      .rtol = 1e-4f,
                      .atol = 0.0f,
                      .expected_passed = false}),
    [](const testing::TestParamInfo<ToleranceCase>& info) {
      return info.param.name;
    });

TEST(CompareTensorsTest, ElementBeyondToleranceIsReportedAsMismatch) {
  const std::vector<float> reference = {1.0f, 2.0f, 3.0f};
  const std::vector<float> candidate = {1.0f, 2.05f, 3.0f};

  ParityReport report =
      CompareTensors(reference, candidate, /*rtol=*/1e-4f, /*atol=*/1e-5f);

  EXPECT_FALSE(report.passed);
  EXPECT_EQ(report.total_elements, 3);
  EXPECT_EQ(report.mismatch_count, 1);
  EXPECT_NEAR(report.max_absolute_error, 0.05, 1e-6);
  EXPECT_NEAR(report.max_relative_error, 0.025, 1e-6);
  EXPECT_THAT(report.details, HasSubstr("idx 1"));
}

TEST(CompareTensorsTest, SizeMismatchFailsWithNonFiniteMetrics) {
  const std::vector<float> reference = {1.0f, 2.0f};
  const std::vector<float> candidate = {1.0f, 2.0f, 3.0f};

  ParityReport report = CompareTensors(reference, candidate);

  EXPECT_FALSE(report.passed);
  EXPECT_EQ(report.mismatch_count, 3);
  EXPECT_EQ(report.max_absolute_error, kInfinity);
  EXPECT_THAT(report.cosine_similarity, IsNan());
  EXPECT_EQ(report.l2_relative_error, kInfinity);
}

TEST(CompareTensorsTest, NonFiniteElementsFailWithNonFiniteMetrics) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  const std::vector<float> reference = {1.0f, nan, 3.0f, 4.0f};
  const std::vector<float> candidate = {1.0f, 2.0f, infinity, 4.0f};

  ParityReport report = CompareTensors(reference, candidate);

  EXPECT_FALSE(report.passed);
  EXPECT_EQ(report.mismatch_count, 2);
  EXPECT_EQ(report.max_absolute_error, kInfinity);
  EXPECT_THAT(report.cosine_similarity, IsNan());
  EXPECT_EQ(report.l2_relative_error, kInfinity);
}

TEST(CompareWithReferenceTest, EqualShapesAndValuesPass) {
  const std::vector<float> values = {1.0f, -2.0f, 3.0f, 4.0f};

  ParityReport report = CompareWithReference({2, 2}, values, {2, 2}, values);

  EXPECT_TRUE(report.passed);
  EXPECT_EQ(report.total_elements, 4);
  EXPECT_EQ(report.mismatch_count, 0);
}

TEST(CompareWithReferenceTest, EqualShapesAreComparedElementwise) {
  const std::vector<float> reference = {1.0f, 2.0f};
  const std::vector<float> candidate = {1.0f, 2.5f};

  ParityReport report =
      CompareWithReference({1, 2}, reference, {1, 2}, candidate);

  EXPECT_FALSE(report.passed);
  EXPECT_EQ(report.mismatch_count, 1);
  EXPECT_DOUBLE_EQ(report.max_absolute_error, 0.5);
}

TEST(CompareWithReferenceTest, DifferentShapesFailEvenWithEqualValues) {
  const std::vector<float> values = {1.0f, -2.0f, 3.0f, 4.0f};

  ParityReport report = CompareWithReference({2, 2}, values, {1, 4}, values);

  EXPECT_FALSE(report.passed);
  EXPECT_EQ(report.total_elements, 4);
  EXPECT_EQ(report.mismatch_count, 4);
  EXPECT_EQ(report.max_absolute_error, kInfinity);
  EXPECT_EQ(report.max_relative_error, kInfinity);
  EXPECT_THAT(report.cosine_similarity, IsNan());
  EXPECT_EQ(report.l2_relative_error, kInfinity);
  EXPECT_EQ(report.details,
            "Tensor shape mismatch: reference [2,2] vs candidate [1,4]");
}

TEST(CompareWithReferenceTest, ShapeMismatchCountsTheLargerTensor) {
  const std::vector<float> reference = {1.0f, 2.0f};
  const std::vector<float> candidate = {1.0f, 2.0f, 3.0f};

  ParityReport report = CompareWithReference({2}, reference, {3}, candidate);

  EXPECT_EQ(report.total_elements, 2);
  EXPECT_EQ(report.mismatch_count, 3);
}

struct InvalidToleranceCase {
  std::string name;
  float rtol;
  float atol;
};

class CompareTensorsInvalidToleranceTest
    : public testing::TestWithParam<InvalidToleranceCase> {};

TEST_P(CompareTensorsInvalidToleranceTest, FailsIdenticalTensors) {
  const std::vector<float> values = {1.0f, 2.0f};

  ParityReport report =
      CompareTensors(values, values, GetParam().rtol, GetParam().atol);

  EXPECT_FALSE(report.passed);
  EXPECT_EQ(report.mismatch_count, 2);
  EXPECT_THAT(report.cosine_similarity, IsNan());
  EXPECT_EQ(report.l2_relative_error, kInfinity);
}

INSTANTIATE_TEST_SUITE_P(
    NegativeOrNonFinite, CompareTensorsInvalidToleranceTest,
    Values(
        InvalidToleranceCase{
            .name = "NegativeRtol", .rtol = -1e-4f, .atol = 1e-5f},
        InvalidToleranceCase{
            .name = "NegativeAtol", .rtol = 1e-4f, .atol = -1e-5f},
        InvalidToleranceCase{.name = "NanRtol",
                             .rtol = std::numeric_limits<float>::quiet_NaN(),
                             .atol = 1e-5f},
        InvalidToleranceCase{.name = "NanAtol",
                             .rtol = 1e-4f,
                             .atol = std::numeric_limits<float>::quiet_NaN()},
        InvalidToleranceCase{.name = "InfiniteRtol",
                             .rtol = std::numeric_limits<float>::infinity(),
                             .atol = 1e-5f},
        InvalidToleranceCase{.name = "InfiniteAtol",
                             .rtol = 1e-4f,
                             .atol = std::numeric_limits<float>::infinity()}),
    [](const testing::TestParamInfo<InvalidToleranceCase>& info) {
      return info.param.name;
    });

struct ParityGateCase {
  std::string name;
  bool report_passed;
  double cosine_similarity;
  double l2_relative_error;
  double min_cosine;
  double max_l2_rel_err;
  bool expected_passed;
};

class ParityGateThresholdTest : public testing::TestWithParam<ParityGateCase> {
};

TEST_P(ParityGateThresholdTest, AppliesThresholdsToReport) {
  const ParityGateCase& tc = GetParam();
  ParityReport report;
  report.passed = tc.report_passed;
  report.cosine_similarity = tc.cosine_similarity;
  report.l2_relative_error = tc.l2_relative_error;

  EXPECT_EQ(EvaluateParityGate(report, tc.min_cosine, tc.max_l2_rel_err).passed,
            tc.expected_passed);
}

// A threshold that is zero or negative disables its gate; with both disabled
// the elementwise verdict of the report decides.
INSTANTIATE_TEST_SUITE_P(
    Thresholds, ParityGateThresholdTest,
    Values(ParityGateCase{.name = "DisabledGatesKeepPassingVerdict",
                          .report_passed = true,
                          .cosine_similarity = 1.0,
                          .l2_relative_error = 0.0,
                          .min_cosine = 0.0,
                          .max_l2_rel_err = 0.0,
                          .expected_passed = true},
           ParityGateCase{.name = "DisabledGatesKeepFailingVerdict",
                          .report_passed = false,
                          .cosine_similarity = 0.99995,
                          .l2_relative_error = 0.001,
                          .min_cosine = 0.0,
                          .max_l2_rel_err = 0.0,
                          .expected_passed = false},
           ParityGateCase{.name = "NegativeThresholdsDisableGates",
                          .report_passed = true,
                          .cosine_similarity = 0.5,
                          .l2_relative_error = 9.0,
                          .min_cosine = -1.0,
                          .max_l2_rel_err = -1.0,
                          .expected_passed = true},
           ParityGateCase{.name = "BothMetricsWithinThresholds",
                          .report_passed = false,
                          .cosine_similarity = 0.99995,
                          .l2_relative_error = 0.001,
                          .min_cosine = 0.9999,
                          .max_l2_rel_err = 0.01,
                          .expected_passed = true},
           ParityGateCase{.name = "CosineBelowMinimum",
                          .report_passed = false,
                          .cosine_similarity = 0.9998,
                          .l2_relative_error = 0.001,
                          .min_cosine = 0.9999,
                          .max_l2_rel_err = 0.01,
                          .expected_passed = false},
           ParityGateCase{.name = "L2AboveMaximum",
                          .report_passed = false,
                          .cosine_similarity = 0.99995,
                          .l2_relative_error = 0.02,
                          .min_cosine = 0.9999,
                          .max_l2_rel_err = 0.01,
                          .expected_passed = false},
           ParityGateCase{.name = "CosineEqualToMinimum",
                          .report_passed = false,
                          .cosine_similarity = 0.9999,
                          .l2_relative_error = 0.001,
                          .min_cosine = 0.9999,
                          .max_l2_rel_err = 0.01,
                          .expected_passed = true},
           ParityGateCase{.name = "L2EqualToMaximum",
                          .report_passed = false,
                          .cosine_similarity = 0.99995,
                          .l2_relative_error = 0.01,
                          .min_cosine = 0.9999,
                          .max_l2_rel_err = 0.01,
                          .expected_passed = true},
           ParityGateCase{.name = "CosineOnlyGateIgnoresL2",
                          .report_passed = false,
                          .cosine_similarity = 0.99995,
                          .l2_relative_error = 5.0,
                          .min_cosine = 0.9999,
                          .max_l2_rel_err = 0.0,
                          .expected_passed = true},
           ParityGateCase{.name = "L2OnlyGateIgnoresCosine",
                          .report_passed = false,
                          .cosine_similarity = 0.5,
                          .l2_relative_error = 0.001,
                          .min_cosine = 0.0,
                          .max_l2_rel_err = 0.01,
                          .expected_passed = true},
           ParityGateCase{.name = "NanCosineFailsL2OnlyGate",
                          .report_passed = false,
                          .cosine_similarity = kNan,
                          .l2_relative_error = 0.001,
                          .min_cosine = 0.0,
                          .max_l2_rel_err = 0.01,
                          .expected_passed = false},
           ParityGateCase{.name = "InfiniteL2FailsCosineOnlyGate",
                          .report_passed = false,
                          .cosine_similarity = 0.99995,
                          .l2_relative_error = kInfinity,
                          .min_cosine = 0.9999,
                          .max_l2_rel_err = 0.0,
                          .expected_passed = false}),
    [](const testing::TestParamInfo<ParityGateCase>& info) {
      return info.param.name;
    });

TEST(EvaluateParityGateTest, PassingGateReturnsReportDetails) {
  ParityReport report;
  report.passed = true;
  report.details = "elementwise summary";

  EXPECT_EQ(EvaluateParityGate(report).details, "elementwise summary");
}

TEST(EvaluateParityGateTest, FailingThresholdGateKeepsReportDetails) {
  ParityReport report;
  report.cosine_similarity = 0.5;
  report.details = "elementwise summary";

  EXPECT_THAT(EvaluateParityGate(report, /*min_cosine=*/0.9).details,
              HasSubstr("elementwise summary"));
}

TEST(EvaluateParityGateTest, SmallDriftFailsElementwiseGate) {
  const std::vector<float> reference = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> drifted = {1.001f, 1.999f, 3.001f, 3.999f};
  ParityReport report =
      CompareTensors(reference, drifted, /*rtol=*/1e-4f, /*atol=*/1e-5f);

  EXPECT_FALSE(EvaluateParityGate(report).passed);
}

TEST(EvaluateParityGateTest, SmallDriftPassesThresholdGate) {
  const std::vector<float> reference = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> drifted = {1.001f, 1.999f, 3.001f, 3.999f};
  ParityReport report =
      CompareTensors(reference, drifted, /*rtol=*/1e-4f, /*atol=*/1e-5f);

  EXPECT_TRUE(EvaluateParityGate(report, /*min_cosine=*/0.9999,
                                 /*max_l2_rel_err=*/0.01)
                  .passed);
}

TEST(EvaluateParityGateTest, SizeMismatchFailsThresholdGate) {
  const std::vector<float> reference = {1.0f, 2.0f};
  const std::vector<float> candidate = {1.0f, 2.0f, 3.0f};
  ParityReport report = CompareTensors(reference, candidate);

  EXPECT_FALSE(
      EvaluateParityGate(report, /*min_cosine=*/0.99, /*max_l2_rel_err=*/0.1)
          .passed);
}

TEST(EvaluateParityGateTest, NonFiniteElementFailsThresholdGate) {
  const std::vector<float> reference = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> candidate = {
      1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f};
  ParityReport report = CompareTensors(reference, candidate);

  EXPECT_FALSE(
      EvaluateParityGate(report, /*min_cosine=*/0.9, /*max_l2_rel_err=*/0.5)
          .passed);
}

TEST(DumpTensorToFileTest, WritesDocumentedBinaryLayout) {
  const std::string path = TempPath("layout_written.dump");
  const std::vector<float> values = {1.0f, -2.0f};

  ASSERT_OK(DumpTensorToFile(path, values, {2}));

  absl::StatusOr<std::vector<uint8_t>> bytes = ReadFile(path);
  ASSERT_OK(bytes.status());
  EXPECT_THAT(*bytes, ElementsAreArray(kTwoElementDumpBytes));
}

TEST(DumpTensorToFileTest, RoundTripsMultiDimensionalTensor) {
  const std::string path = TempPath("round_trip.dump");
  const std::vector<float> values = {-3.0f, -2.5f, -2.0f, -1.5f, -1.0f, -0.5f};

  ASSERT_OK(DumpTensorToFile(path, values, {2, 3}));

  absl::StatusOr<LoadedTensor> loaded = LoadTensorFromFile(path);
  ASSERT_OK(loaded.status());
  EXPECT_THAT(*loaded, Pair(ElementsAre(2, 3), ElementsAreArray(values)));
}

struct InvalidDumpArgumentsCase {
  std::string name;
  std::vector<float> values;
  std::vector<int64_t> shape;
};

class InvalidDumpArgumentsTest
    : public testing::TestWithParam<InvalidDumpArgumentsCase> {};

TEST_P(InvalidDumpArgumentsTest, DumpReturnsInvalidArgument) {
  const std::string path =
      TempPath(absl::StrCat("invalid_arguments_", GetParam().name, ".dump"));

  absl::Status status =
      DumpTensorToFile(path, GetParam().values, GetParam().shape);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
}

// Every shape other than the mismatch cases has as many elements as `values`,
// so that only the named rule is violated.
INSTANTIATE_TEST_SUITE_P(
    ShapeRules, InvalidDumpArgumentsTest,
    Values(InvalidDumpArgumentsCase{"RankZero", {1.0f}, {}},
           InvalidDumpArgumentsCase{
               "RankAboveMaximum", {1.0f}, std::vector<int64_t>(17, 1)},
           InvalidDumpArgumentsCase{"ZeroDimension", {}, {0}},
           InvalidDumpArgumentsCase{
               "NegativeDimensions", {1.0f, 2.0f, 3.0f}, {-3, -1}},
           InvalidDumpArgumentsCase{
               "ElementCountMismatch", {1.0f, 2.0f, 3.0f}, {2, 2}},
           InvalidDumpArgumentsCase{"ElementCountOverflowsInt64",
                                    {1.0f, 2.0f, 3.0f},
                                    {std::numeric_limits<int64_t>::max(), 2}}),
    [](const testing::TestParamInfo<InvalidDumpArgumentsCase>& info) {
      return info.param.name;
    });

TEST(DumpTensorToFileTest, UnwritablePathReturnsInternal) {
  const std::vector<float> values = {1.0f};

  absl::Status status =
      DumpTensorToFile(TempPath("missing_directory/out.dump"), values, {1});

  EXPECT_EQ(status.code(), absl::StatusCode::kInternal) << status;
}

TEST(LoadTensorFromFileTest, ParsesDocumentedBinaryLayout) {
  const std::string path = TempPath("layout_parsed.dump");
  ASSERT_OK(WriteFile(path, std::string(std::begin(kTwoElementDumpBytes),
                                        std::end(kTwoElementDumpBytes))));

  absl::StatusOr<LoadedTensor> loaded = LoadTensorFromFile(path);

  ASSERT_OK(loaded.status());
  EXPECT_THAT(*loaded, Pair(ElementsAre(2), ElementsAre(1.0f, -2.0f)));
}

TEST(LoadTensorFromFileTest, MissingFileReturnsNotFound) {
  absl::StatusOr<LoadedTensor> loaded =
      LoadTensorFromFile(TempPath("does_not_exist.dump"));

  EXPECT_EQ(loaded.status().code(), absl::StatusCode::kNotFound)
      << loaded.status();
}

TEST(LoadTensorFromFileTest, DumpTooLargeForSizeTIsReportedAsOverflow) {
  // 2^62 - 1 elements take 2^64 - 4 payload bytes, which fit in a 64-bit
  // size_t on their own but not together with the 24 bytes of header and
  // shape.
  const std::string path = TempPath("dump_size_overflow.dump");
  ASSERT_OK(WriteFile(
      path,
      Serialize({.rank = 2, .dims = {2147483647, 2147483649}, .payload = {}})));

  absl::StatusOr<LoadedTensor> loaded = LoadTensorFromFile(path);

  EXPECT_EQ(loaded.status().code(), absl::StatusCode::kInvalidArgument)
      << loaded.status();
  EXPECT_THAT(std::string(loaded.status().message()),
              HasSubstr("overflows size_t"));
}

struct InvalidDumpFileCase {
  std::string name;
  std::string contents;
};

class InvalidDumpFileTest : public testing::TestWithParam<InvalidDumpFileCase> {
};

TEST_P(InvalidDumpFileTest, LoadReturnsInvalidArgument) {
  const std::string path =
      TempPath(absl::StrCat("invalid_file_", GetParam().name, ".dump"));
  ASSERT_OK(WriteFile(path, GetParam().contents));

  absl::StatusOr<LoadedTensor> loaded = LoadTensorFromFile(path);

  EXPECT_EQ(loaded.status().code(), absl::StatusCode::kInvalidArgument)
      << loaded.status();
}

// Each file departs from a valid one-element dump in a single respect.
INSTANTIATE_TEST_SUITE_P(
    FormatViolations, InvalidDumpFileTest,
    Values(InvalidDumpFileCase{"TooShortForHeader", std::string(4, '\0')},
           InvalidDumpFileCase{"BadMagic", Serialize({.magic = 0xDEADBEEF})},
           InvalidDumpFileCase{
               "RankZero", Serialize({.rank = 0, .dims = {}, .payload = {}})},
           InvalidDumpFileCase{
               "RankAboveMaximum",
               Serialize({.rank = 17, .dims = std::vector<int64_t>(17, 1)})},
           InvalidDumpFileCase{
               "TruncatedShape",
               Serialize({.rank = 2, .dims = {1}, .payload = {}})},
           InvalidDumpFileCase{"ZeroDimension",
                               Serialize({.dims = {0}, .payload = {}})},
           InvalidDumpFileCase{"NegativeDimension",
                               Serialize({.dims = {-4}, .payload = {}})},
           InvalidDumpFileCase{
               "ElementCountOverflowsInt64",
               Serialize({.rank = 2,
                          .dims = {std::numeric_limits<int64_t>::max(), 2},
                          .payload = {}})},
           InvalidDumpFileCase{
               "PayloadBytesOverflowSizeT",
               Serialize({.dims = {int64_t{1} << 62}, .payload = {}})},
           InvalidDumpFileCase{"TruncatedPayload", Serialize({.dims = {10}})},
           InvalidDumpFileCase{"TrailingBytes",
                               Serialize({.trailing_bytes = "X"})}),
    [](const testing::TestParamInfo<InvalidDumpFileCase>& info) {
      return info.param.name;
    });

TEST(ExecuteAndDumpParityTest, DumpsBackendOutputWithSpecOutputShape) {
  const std::string path = TempPath("fake_backend_output.dump");
  FakeLayerBackend backend;

  ASSERT_OK(ExecuteAndDumpParity(backend, CreateTestSpec(LayerType::kRmsNorm),
                                 kSeed, path));

  // The fake output holds the forward-pass count in each of the B*T*M =
  // 1*8*64 elements, so 1.0f also shows that the layer ran exactly once.
  absl::StatusOr<LoadedTensor> loaded = LoadTensorFromFile(path);
  ASSERT_OK(loaded.status());
  EXPECT_THAT(*loaded,
              Pair(ElementsAre(1, 8, 64), AllOf(SizeIs(512), Each(Eq(1.0f)))));
}

TEST(ExecuteAndDumpParityTest, BindsDataSeededWithRequestedSeed) {
  const LayerSpec spec = CreateTestSpec(LayerType::kRmsNorm);
  FakeLayerBackend backend;

  ASSERT_OK(
      ExecuteAndDumpParity(backend, spec, kSeed, TempPath("seeded_data.dump")));

  EXPECT_EQ(backend.data().input, CreateSeededLayerData(spec, kSeed).input);
  EXPECT_NE(backend.data().input,
            CreateSeededLayerData(spec, kOtherSeed).input);
}

TEST(ExecuteAndDumpParityTest, ForwardsBackendOptions) {
  FakeLayerBackend backend;

  ASSERT_OK(ExecuteAndDumpParity(
      backend, CreateTestSpec(LayerType::kRmsNorm), kSeed,
      TempPath("forwarded_options.dump"),
      BackendOptions{.num_threads = 3, .use_ynnpack = true}));

  EXPECT_EQ(backend.options().num_threads, 3);
  EXPECT_TRUE(backend.options().use_ynnpack);
}

TEST(ExecuteAndDumpParityTest, InvalidSpecReturnsInvalidArgumentBeforeSetup) {
  LayerSpec spec = CreateTestSpec(LayerType::kRmsNorm);
  spec.batch_size = 0;
  FakeLayerBackend backend;

  absl::Status status =
      ExecuteAndDumpParity(backend, spec, kSeed, TempPath("invalid_spec.dump"));

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_EQ(backend.setup_call_count(), 0);
}

TEST(ExecuteAndDumpParityTest, PropagatesSetupFailure) {
  FakeLayerBackend backend;
  backend.FailSetupWith(absl::ResourceExhaustedError("setup failed"));

  absl::Status status =
      ExecuteAndDumpParity(backend, CreateTestSpec(LayerType::kRmsNorm), kSeed,
                           TempPath("setup_failure.dump"));

  EXPECT_EQ(status.code(), absl::StatusCode::kResourceExhausted) << status;
}

TEST(ExecuteAndDumpParityTest, PropagatesExecuteFailure) {
  FakeLayerBackend backend;
  backend.FailExecuteWith(absl::DataLossError("execute failed"));

  absl::Status status =
      ExecuteAndDumpParity(backend, CreateTestSpec(LayerType::kRmsNorm), kSeed,
                           TempPath("execute_failure.dump"));

  EXPECT_EQ(status.code(), absl::StatusCode::kDataLoss) << status;
}

TEST(ExecuteAndDumpParityTest, PropagatesOutputReadFailure) {
  FakeLayerBackend backend;
  backend.FailGetOutputDataWith(absl::UnavailableError("output unavailable"));

  absl::Status status =
      ExecuteAndDumpParity(backend, CreateTestSpec(LayerType::kRmsNorm), kSeed,
                           TempPath("output_failure.dump"));

  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable) << status;
}

TEST(ExecuteAndDumpParityTest, OutputSmallerThanSpecShapeIsInvalidArgument) {
  FakeLayerBackend backend;
  backend.set_output_size(3);

  absl::Status status =
      ExecuteAndDumpParity(backend, CreateTestSpec(LayerType::kRmsNorm), kSeed,
                           TempPath("output_size_mismatch.dump"));

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
}

TEST(ExecuteAndDumpParityTest, UnwritablePathReturnsInternal) {
  FakeLayerBackend backend;

  absl::Status status =
      ExecuteAndDumpParity(backend, CreateTestSpec(LayerType::kRmsNorm), kSeed,
                           TempPath("missing_directory/out.dump"));

  EXPECT_EQ(status.code(), absl::StatusCode::kInternal) << status;
}

struct LitertDumpCase {
  std::string name;
  LayerType layer_type;
  AttentionMode attention_mode;
  std::vector<int64_t> expected_shape;
};

class LitertBackendDumpTest : public testing::TestWithParam<LitertDumpCase> {};

TEST_P(LitertBackendDumpTest, DumpsFiniteOutputWithLayerShape) {
  const LitertDumpCase& tc = GetParam();
  const std::string path = TempPath(absl::StrCat("litert_", tc.name, ".dump"));
  LayerSpec spec = CreateTestSpec(tc.layer_type, tc.attention_mode);
  spec.embed_dim = 48;
  LitertTensorBackend backend;

  ASSERT_OK(ExecuteAndDumpParity(backend, spec, kSeed, path));

  absl::StatusOr<LoadedTensor> loaded = LoadTensorFromFile(path);
  ASSERT_OK(loaded.status());
  EXPECT_THAT(*loaded,
              Pair(ElementsAreArray(tc.expected_shape), Each(IsFinite())));
}

// B=1, T=8 (1 in decode), H=4, D=16, H*D=64, and M=48 decoupled from H*D.
INSTANTIATE_TEST_SUITE_P(
    DecoupledEmbedDim, LitertBackendDumpTest,
    Values(LitertDumpCase{"RmsNorm",
                          LayerType::kRmsNorm,
                          AttentionMode::kPrefill,
                          {1, 8, 48}},
           LitertDumpCase{
               "RoPE", LayerType::kRoPE, AttentionMode::kPrefill, {1, 8, 64}},
           LitertDumpCase{"StandardAttentionPrefill",
                          LayerType::kStandardAttention,
                          AttentionMode::kPrefill,
                          {1, 4, 8, 16}},
           LitertDumpCase{"DecodeOptimizedAttention",
                          LayerType::kDecodeOptimizedAttention,
                          AttentionMode::kDecode,
                          {1, 4, 1, 16}}),
    [](const testing::TestParamInfo<LitertDumpCase>& info) {
      return info.param.name;
    });

TEST(ExecuteAndDumpParityTest, SameSeedReproducesLitertOutputBitForBit) {
  const LayerSpec spec = CreateTestSpec(LayerType::kGatedMlp);
  const std::string first_path = TempPath("gated_mlp_first.dump");
  const std::string second_path = TempPath("gated_mlp_second.dump");
  LitertTensorBackend first_backend;
  LitertTensorBackend second_backend;

  ASSERT_OK(ExecuteAndDumpParity(first_backend, spec, kSeed, first_path));
  ASSERT_OK(ExecuteAndDumpParity(second_backend, spec, kSeed, second_path));

  absl::StatusOr<LoadedTensor> first = LoadTensorFromFile(first_path);
  absl::StatusOr<LoadedTensor> second = LoadTensorFromFile(second_path);
  ASSERT_OK(first.status());
  ASSERT_OK(second.status());
  EXPECT_EQ(first->second, second->second);
}

TEST(ExecuteAndDumpParityTest, DifferentSeedsChangeLitertOutput) {
  const LayerSpec spec = CreateTestSpec(LayerType::kGatedMlp);
  const std::string first_path = TempPath("gated_mlp_seed.dump");
  const std::string second_path = TempPath("gated_mlp_other_seed.dump");
  LitertTensorBackend first_backend;
  LitertTensorBackend second_backend;

  ASSERT_OK(ExecuteAndDumpParity(first_backend, spec, kSeed, first_path));
  ASSERT_OK(
      ExecuteAndDumpParity(second_backend, spec, kOtherSeed, second_path));

  absl::StatusOr<LoadedTensor> first = LoadTensorFromFile(first_path);
  absl::StatusOr<LoadedTensor> second = LoadTensorFromFile(second_path);
  ASSERT_OK(first.status());
  ASSERT_OK(second.status());
  EXPECT_FALSE(CompareTensors(first->second, second->second).passed);
}

TEST(ExecuteAndDumpParityTest, DumpsOfEquivalentAttentionLayersPassComparison) {
  const std::string standard_path = TempPath("standard_attention.dump");
  const std::string optimized_path = TempPath("optimized_attention.dump");
  LitertTensorBackend standard_backend;
  LitertTensorBackend optimized_backend;

  ASSERT_OK(ExecuteAndDumpParity(
      standard_backend,
      CreateTestSpec(LayerType::kStandardAttention, AttentionMode::kDecode),
      kSeed, standard_path));
  ASSERT_OK(
      ExecuteAndDumpParity(optimized_backend,
                           CreateTestSpec(LayerType::kDecodeOptimizedAttention,
                                          AttentionMode::kDecode),
                           kSeed, optimized_path));

  absl::StatusOr<LoadedTensor> standard = LoadTensorFromFile(standard_path);
  absl::StatusOr<LoadedTensor> optimized = LoadTensorFromFile(optimized_path);
  ASSERT_OK(standard.status());
  ASSERT_OK(optimized.status());
  ParityReport report =
      CompareTensors(standard->second, optimized->second, /*rtol=*/1e-4f,
                     /*atol=*/1e-5f);
  EXPECT_TRUE(report.passed) << report.details;
}

TEST(SingleCaseParityGuardTest, FirstClaimSucceeds) {
  SingleCaseParityGuard guard;

  EXPECT_EQ(guard.Claim(kOwnerCase), absl::OkStatus());
}

TEST(SingleCaseParityGuardTest, OwnerMayClaimRepeatedly) {
  SingleCaseParityGuard guard;
  ASSERT_OK(guard.Claim(kOwnerCase));

  EXPECT_EQ(guard.Claim(kOwnerCase), absl::OkStatus());
}

TEST(SingleCaseParityGuardTest, DifferentCaseIsRefused) {
  SingleCaseParityGuard guard;
  ASSERT_OK(guard.Claim(kOwnerCase));

  EXPECT_EQ(guard.Claim(kOtherCase).code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(SingleCaseParityGuardTest, OwnerKeepsFileAfterRefusingAnotherCase) {
  SingleCaseParityGuard guard;
  ASSERT_OK(guard.Claim(kOwnerCase));
  guard.Claim(kOtherCase).IgnoreError();

  EXPECT_EQ(guard.Claim(kOwnerCase), absl::OkStatus());
}

TEST(SingleCaseParityGuardTest, EmptyNameOwnsLikeAnyOtherName) {
  SingleCaseParityGuard guard;
  ASSERT_OK(guard.Claim(""));

  EXPECT_EQ(guard.Claim(""), absl::OkStatus());
  EXPECT_EQ(guard.Claim(kOtherCase).code(),
            absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace layer_ml_bench
