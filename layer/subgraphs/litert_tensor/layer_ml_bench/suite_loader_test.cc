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

#include "tensor/benchmarks/layer/suite_loader.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensor/benchmarks/layer/layer_spec.h"
#include "tensor/benchmarks/layer/litert_tensor_backend.h"
#include "tensor/benchmarks/layer/status_test_util.h"

namespace layer_ml_bench {
namespace {

using ::layer_ml_bench::litert::RuntimeBackend;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::Optional;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::testing::Values;

// Returns the path of `file_name` inside the test's temporary directory.
std::string TempPath(absl::string_view file_name) {
  return absl::StrCat(testing::TempDir(), "/", file_name);
}

// Returns suite JSON whose single inline preset "P" holds the four required
// dimensions followed by `member`, one additional `"key": value` member.
std::string SuiteWithPresetMember(absl::string_view member) {
  return absl::StrCat(
      R"({"presets": [{"name": "P", "num_heads": 4, "head_dim": 64, )",
      R"("hidden_dim": 128, "num_kv_heads": 2, )", member, "}]}");
}

TEST(BuildSuiteSpecTest, EmptyPathReturnsBuiltInSuite) {
  absl::StatusOr<SuiteSpec> suite = BuildSuiteSpec("");

  ASSERT_OK(suite.status());
  EXPECT_THAT(suite->families, SizeIs(9));
  EXPECT_THAT(suite->presets,
              Contains(Field(&ModelPreset::name, "Gemma4-E2B")));
  EXPECT_THAT(suite->batch_sizes, ElementsAre(1, 4));
  EXPECT_THAT(suite->seq_lengths, ElementsAre(128, 512, 1024, 2048));
  EXPECT_THAT(suite->thread_counts, ElementsAre(1, 4, 8, 16, 32));
}

TEST(BuildSuiteSpecTest, MissingFileReturnsNotFound) {
  absl::StatusOr<SuiteSpec> suite =
      BuildSuiteSpec(TempPath("does_not_exist.json"));

  EXPECT_EQ(suite.status().code(), absl::StatusCode::kNotFound)
      << suite.status();
}

TEST(BuildSuiteSpecTest, LoadsSuiteFromFile) {
  const std::string path = TempPath("valid_suite.json");
  {
    std::ofstream out(path);
    out << R"({"presets": ["TinyCI"], "thread_counts": [2]})";
  }

  absl::StatusOr<SuiteSpec> suite = BuildSuiteSpec(path);

  ASSERT_OK(suite.status());
  EXPECT_THAT(suite->presets, ElementsAre(Field(&ModelPreset::name, "TinyCI")));
  EXPECT_THAT(suite->thread_counts, ElementsAre(2));
}

TEST(BuildSuiteSpecTest, MalformedFileReturnsInvalidArgument) {
  const std::string path = TempPath("malformed_suite.json");
  {
    std::ofstream out(path);
    out << "{bad";
  }

  absl::StatusOr<SuiteSpec> suite = BuildSuiteSpec(path);

  EXPECT_EQ(suite.status().code(), absl::StatusCode::kInvalidArgument)
      << suite.status();
}

TEST(ParseSuiteJsonTest, EmptyObjectKeepsBuiltInSuite) {
  absl::StatusOr<SuiteSpec> suite = ParseSuiteJson("{}");

  ASSERT_OK(suite.status());
  EXPECT_THAT(suite->families, SizeIs(9));
  EXPECT_THAT(suite->presets,
              Contains(Field(&ModelPreset::name, "Gemma4-E2B")));
  EXPECT_THAT(suite->batch_sizes, ElementsAre(1, 4));
  EXPECT_THAT(suite->seq_lengths, ElementsAre(128, 512, 1024, 2048));
  EXPECT_THAT(suite->thread_counts, ElementsAre(1, 4, 8, 16, 32));
}

TEST(ParseSuiteJsonTest, SweepAxesReplaceDefaults) {
  absl::StatusOr<SuiteSpec> suite =
      ParseSuiteJson(R"({"batch_sizes": [1, 2], "seq_lengths": [512], )"
                     R"("thread_counts": [4, 8]})");

  ASSERT_OK(suite.status());
  EXPECT_THAT(suite->batch_sizes, ElementsAre(1, 2));
  EXPECT_THAT(suite->seq_lengths, ElementsAre(512));
  EXPECT_THAT(suite->thread_counts, ElementsAre(4, 8));
}

TEST(ParseSuiteJsonTest, AcceptsLargestSupportedInteger) {
  // 2^31 - 1, the largest value a layer dimension can hold.
  absl::StatusOr<SuiteSpec> suite =
      ParseSuiteJson(R"({"seq_lengths": [2147483647]})");

  ASSERT_OK(suite.status());
  EXPECT_THAT(suite->seq_lengths, ElementsAre(2147483647));
}

TEST(ParseSuiteJsonTest, FamiliesSelectBuiltInFamiliesByName) {
  absl::StatusOr<SuiteSpec> suite = ParseSuiteJson(
      R"({"families": ["BM_TransformerBlock_Decode", "BM_RmsNorm"]})");

  ASSERT_OK(suite.status());
  EXPECT_THAT(
      suite->families,
      UnorderedElementsAre(
          AllOf(Field(&FamilyConfig::family_name, "BM_RmsNorm"),
                Field(&FamilyConfig::layer_type, LayerType::kRmsNorm),
                Field(&FamilyConfig::attention_mode, AttentionMode::kPrefill)),
          AllOf(Field(&FamilyConfig::family_name, "BM_TransformerBlock_Decode"),
                Field(&FamilyConfig::layer_type, LayerType::kTransformerBlock),
                Field(&FamilyConfig::attention_mode, AttentionMode::kDecode))));
}

TEST(ParseSuiteJsonTest, PresetNameSelectsBuiltInPreset) {
  absl::StatusOr<SuiteSpec> suite =
      ParseSuiteJson(R"({"presets": ["TinyCI"]})");

  ASSERT_OK(suite.status());
  EXPECT_THAT(suite->presets,
              ElementsAre(AllOf(Field(&ModelPreset::name, "TinyCI"),
                                Field(&ModelPreset::num_heads, 4),
                                Field(&ModelPreset::head_dim, 64))));
}

TEST(ParseSuiteJsonTest, InlinePresetSetsEveryField) {
  constexpr absl::string_view kJson = R"json({
    "presets": [{
      "name": "CustomBlock",
      "num_heads": 8,
      "head_dim": 128,
      "hidden_dim": 4096,
      "num_kv_heads": 2,
      "embed_dim": 1024,
      "eps": 1e-6,
      "mlp_activation": "gelu_pytorch_tanh",
      "qk_norm": true,
      "post_norms": true,
      "sliding_window": 256
    }]
  })json";

  absl::StatusOr<SuiteSpec> suite = ParseSuiteJson(kJson, "inline.json");

  ASSERT_OK(suite.status());
  ASSERT_THAT(suite->presets, SizeIs(1));
  const ModelPreset& preset = suite->presets.front();
  EXPECT_EQ(preset.name, "CustomBlock");
  EXPECT_EQ(preset.num_heads, 8);
  EXPECT_EQ(preset.head_dim, 128);
  EXPECT_EQ(preset.hidden_dim, 4096);
  EXPECT_EQ(preset.num_kv_heads, 2);
  EXPECT_EQ(preset.embed_dim, 1024);
  EXPECT_FLOAT_EQ(preset.eps, 1e-6f);
  EXPECT_THAT(preset.block_mlp_activation, Optional(MlpActivation::kGeGLU));
  EXPECT_TRUE(preset.qk_norm);
  EXPECT_TRUE(preset.post_norms);
  EXPECT_EQ(preset.sliding_window, 256);
}

TEST(ParseSuiteJsonTest, InlinePresetOptionalFieldsHaveDefaults) {
  absl::StatusOr<SuiteSpec> suite = ParseSuiteJson(
      R"({"presets": [{"name": "P", "num_heads": 4, "head_dim": 64, )"
      R"("hidden_dim": 128, "num_kv_heads": 2}]})");

  ASSERT_OK(suite.status());
  ASSERT_THAT(suite->presets, SizeIs(1));
  const ModelPreset& preset = suite->presets.front();
  EXPECT_EQ(preset.embed_dim, 0);
  EXPECT_FLOAT_EQ(preset.eps, 1e-5f);
  EXPECT_EQ(preset.block_mlp_activation, std::nullopt);
  EXPECT_FALSE(preset.qk_norm);
  EXPECT_FALSE(preset.post_norms);
  EXPECT_EQ(preset.sliding_window, 0);
}

TEST(ParseSuiteJsonTest, PresetsKeepListedOrder) {
  absl::StatusOr<SuiteSpec> suite = ParseSuiteJson(
      R"({"presets": [{"name": "P", "num_heads": 4, "head_dim": 64, )"
      R"("hidden_dim": 128, "num_kv_heads": 2}, "TinyCI"]})");

  ASSERT_OK(suite.status());
  EXPECT_THAT(suite->presets, ElementsAre(Field(&ModelPreset::name, "P"),
                                          Field(&ModelPreset::name, "TinyCI")));
}

TEST(ParseSuiteJsonTest, NullSlidingWindowMeansNoWindow) {
  absl::StatusOr<SuiteSpec> suite =
      ParseSuiteJson(SuiteWithPresetMember(R"("sliding_window": null)"));

  ASSERT_OK(suite.status());
  EXPECT_THAT(suite->presets,
              ElementsAre(Field(&ModelPreset::sliding_window, 0)));
}

struct ActivationCase {
  std::string name;
  std::string json_value;
  MlpActivation expected_activation;
};

class PresetActivationTest : public testing::TestWithParam<ActivationCase> {};

TEST_P(PresetActivationTest, MapsNameToMlpActivation) {
  absl::StatusOr<SuiteSpec> suite = ParseSuiteJson(SuiteWithPresetMember(
      absl::StrCat(R"("mlp_activation": ")", GetParam().json_value, "\"")));

  ASSERT_OK(suite.status());
  EXPECT_THAT(suite->presets,
              ElementsAre(Field(&ModelPreset::block_mlp_activation,
                                Optional(GetParam().expected_activation))));
}

INSTANTIATE_TEST_SUITE_P(
    SupportedNames, PresetActivationTest,
    Values(ActivationCase{"Geglu", "geglu", MlpActivation::kGeGLU},
           ActivationCase{"UppercaseGelu", "GELU", MlpActivation::kGeGLU},
           ActivationCase{"GeluPytorchTanh", "gelu_pytorch_tanh",
                          MlpActivation::kGeGLU},
           ActivationCase{"Swiglu", "swiglu", MlpActivation::kSwiGLU},
           ActivationCase{"MixedCaseSilu", "SiLU", MlpActivation::kSwiGLU}),
    [](const testing::TestParamInfo<ActivationCase>& info) {
      return info.param.name;
    });

struct InvalidSuiteJsonCase {
  std::string name;
  std::string json;
};

class InvalidSuiteJsonTest
    : public testing::TestWithParam<InvalidSuiteJsonCase> {};

TEST_P(InvalidSuiteJsonTest, ParseReturnsInvalidArgument) {
  absl::StatusOr<SuiteSpec> suite = ParseSuiteJson(GetParam().json);

  EXPECT_EQ(suite.status().code(), absl::StatusCode::kInvalidArgument)
      << suite.status();
}

std::string InvalidSuiteJsonCaseName(
    const testing::TestParamInfo<InvalidSuiteJsonCase>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(Document, InvalidSuiteJsonTest,
                         Values(InvalidSuiteJsonCase{"MalformedJson", "{bad"},
                                InvalidSuiteJsonCase{"NonObjectRoot",
                                                     "[1, 2]"}),
                         InvalidSuiteJsonCaseName);

// 2147483648 is 2^31 and 4294967296 is 2^32; both exceed the largest supported
// integer 2^31 - 1.
INSTANTIATE_TEST_SUITE_P(
    SweepAxes, InvalidSuiteJsonTest,
    Values(InvalidSuiteJsonCase{"EmptyBatchSizes", R"({"batch_sizes": []})"},
           InvalidSuiteJsonCase{"BatchSizesNotAnArray",
                                R"({"batch_sizes": 4})"},
           InvalidSuiteJsonCase{"FractionalBatchSize",
                                R"({"batch_sizes": [1.5]})"},
           InvalidSuiteJsonCase{"BatchSizeAboveInt32",
                                R"({"batch_sizes": [4294967296]})"},
           InvalidSuiteJsonCase{"ZeroSeqLength", R"({"seq_lengths": [0]})"},
           InvalidSuiteJsonCase{"SeqLengthJustAboveInt32",
                                R"({"seq_lengths": [2147483648]})"},
           InvalidSuiteJsonCase{"NonNumericSeqLength",
                                R"({"seq_lengths": ["512"]})"},
           InvalidSuiteJsonCase{"NegativeThreadCount",
                                R"({"thread_counts": [-2]})"}),
    InvalidSuiteJsonCaseName);

INSTANTIATE_TEST_SUITE_P(
    Families, InvalidSuiteJsonTest,
    Values(InvalidSuiteJsonCase{"EmptyFamilies", R"({"families": []})"},
           InvalidSuiteJsonCase{"FamiliesNotAnArray",
                                R"({"families": "BM_RmsNorm"})"},
           InvalidSuiteJsonCase{"EmptyFamilyName", R"({"families": [""]})"},
           InvalidSuiteJsonCase{"NonStringFamily", R"({"families": [7]})"},
           InvalidSuiteJsonCase{"UnknownFamily",
                                R"({"families": ["BM_Nope"]})"},
           InvalidSuiteJsonCase{"UnknownFamilyAmongKnown",
                                R"({"families": ["BM_RmsNorm", "BM_Nope"]})"}),
    InvalidSuiteJsonCaseName);

INSTANTIATE_TEST_SUITE_P(
    PresetList, InvalidSuiteJsonTest,
    Values(
        InvalidSuiteJsonCase{"EmptyPresets", R"({"presets": []})"},
        InvalidSuiteJsonCase{"PresetsNotAnArray", R"({"presets": "TinyCI"})"},
        InvalidSuiteJsonCase{"PresetNeitherNameNorObject",
                             R"({"presets": [123]})"},
        InvalidSuiteJsonCase{"UnknownBuiltInPreset",
                             R"({"presets": ["Nope"]})"},
        InvalidSuiteJsonCase{"BuiltInPresetListedTwice",
                             R"({"presets": ["TinyCI", "TinyCI"]})"},
        InvalidSuiteJsonCase{"InlinePresetsWithSameName",
                             R"({"presets": [)"
                             R"({"name": "P", "num_heads": 4, "head_dim": 64, )"
                             R"("hidden_dim": 128, "num_kv_heads": 2}, )"
                             R"({"name": "P", "num_heads": 8, "head_dim": 32, )"
                             R"("hidden_dim": 256, "num_kv_heads": 4}]})"},
        InvalidSuiteJsonCase{
            "InlinePresetNamedLikeListedBuiltIn",
            R"({"presets": ["TinyCI", )"
            R"({"name": "TinyCI", "num_heads": 4, "head_dim": 64, )"
            R"("hidden_dim": 128, "num_kv_heads": 2}]})"}),
    InvalidSuiteJsonCaseName);

// Each preset is valid apart from the named member.
INSTANTIATE_TEST_SUITE_P(
    PresetRequiredFields, InvalidSuiteJsonTest,
    Values(
        InvalidSuiteJsonCase{"MissingName",
                             R"({"presets": [{"num_heads": 4, "head_dim": 64, )"
                             R"("hidden_dim": 128, "num_kv_heads": 2}]})"},
        InvalidSuiteJsonCase{
            "EmptyName",
            R"({"presets": [{"name": "", "num_heads": 4, "head_dim": 64, )"
            R"("hidden_dim": 128, "num_kv_heads": 2}]})"},
        InvalidSuiteJsonCase{
            "NonStringName",
            R"({"presets": [{"name": 7, "num_heads": 4, "head_dim": 64, )"
            R"("hidden_dim": 128, "num_kv_heads": 2}]})"},
        InvalidSuiteJsonCase{"MissingNumHeads",
                             R"({"presets": [{"name": "P", "head_dim": 64, )"
                             R"("hidden_dim": 128, "num_kv_heads": 2}]})"},
        InvalidSuiteJsonCase{"MissingHeadDim",
                             R"({"presets": [{"name": "P", "num_heads": 4, )"
                             R"("hidden_dim": 128, "num_kv_heads": 2}]})"},
        InvalidSuiteJsonCase{
            "MissingHiddenDim",
            R"({"presets": [{"name": "P", "num_heads": 4, "head_dim": 64, )"
            R"("num_kv_heads": 2}]})"},
        InvalidSuiteJsonCase{
            "MissingNumKvHeads",
            R"({"presets": [{"name": "P", "num_heads": 4, "head_dim": 64, )"
            R"("hidden_dim": 128}]})"},
        InvalidSuiteJsonCase{
            "ZeroNumHeads",
            R"({"presets": [{"name": "P", "num_heads": 0, "head_dim": 64, )"
            R"("hidden_dim": 128, "num_kv_heads": 2}]})"},
        InvalidSuiteJsonCase{
            "FractionalHeadDim",
            R"({"presets": [{"name": "P", "num_heads": 4, "head_dim": 1.5, )"
            R"("hidden_dim": 128, "num_kv_heads": 2}]})"},
        InvalidSuiteJsonCase{
            "NumHeadsAboveInt32",
            R"({"presets": [{"name": "P", "num_heads": 4294967296, )"
            R"("head_dim": 64, "hidden_dim": 128, "num_kv_heads": 2}]})"}),
    InvalidSuiteJsonCaseName);

// 1e-60 is positive in double precision but underflows single precision;
// 1e60 exceeds the largest single-precision value.
INSTANTIATE_TEST_SUITE_P(
    PresetOptionalFields, InvalidSuiteJsonTest,
    Values(
        InvalidSuiteJsonCase{"NegativeEmbedDim",
                             SuiteWithPresetMember(R"("embed_dim": -1)")},
        InvalidSuiteJsonCase{"FractionalEmbedDim",
                             SuiteWithPresetMember(R"("embed_dim": 1.5)")},
        InvalidSuiteJsonCase{"NullEmbedDim",
                             SuiteWithPresetMember(R"("embed_dim": null)")},
        InvalidSuiteJsonCase{
            "EmbedDimAboveInt32",
            SuiteWithPresetMember(R"("embed_dim": 4294967296)")},
        InvalidSuiteJsonCase{"ZeroEps", SuiteWithPresetMember(R"("eps": 0)")},
        InvalidSuiteJsonCase{"NegativeEps",
                             SuiteWithPresetMember(R"("eps": -1e-6)")},
        InvalidSuiteJsonCase{"NonNumericEps",
                             SuiteWithPresetMember(R"("eps": "1e-6")")},
        InvalidSuiteJsonCase{"EpsUnderflowingSinglePrecision",
                             SuiteWithPresetMember(R"("eps": 1e-60)")},
        InvalidSuiteJsonCase{"EpsOverflowingSinglePrecision",
                             SuiteWithPresetMember(R"("eps": 1e60)")},
        InvalidSuiteJsonCase{
            "UnsupportedActivation",
            SuiteWithPresetMember(R"("mlp_activation": "relu")")},
        InvalidSuiteJsonCase{"NonStringActivation",
                             SuiteWithPresetMember(R"("mlp_activation": 5)")},
        InvalidSuiteJsonCase{"NonBooleanQkNorm",
                             SuiteWithPresetMember(R"("qk_norm": "yes")")},
        InvalidSuiteJsonCase{"NonBooleanPostNorms",
                             SuiteWithPresetMember(R"("post_norms": 1)")},
        InvalidSuiteJsonCase{"NegativeSlidingWindow",
                             SuiteWithPresetMember(R"("sliding_window": -1)")},
        InvalidSuiteJsonCase{"FractionalSlidingWindow",
                             SuiteWithPresetMember(R"("sliding_window": 2.5)")},
        InvalidSuiteJsonCase{
            "NonNumericSlidingWindow",
            SuiteWithPresetMember(R"("sliding_window": "256")")},
        InvalidSuiteJsonCase{
            "SlidingWindowAboveInt32",
            SuiteWithPresetMember(R"("sliding_window": 4294967296)")}),
    InvalidSuiteJsonCaseName);

// Families of the built-in suite that the ExpandSuite tests expand.
FamilyConfig RmsNormFamily() {
  return {"BM_RmsNorm", LayerType::kRmsNorm, AttentionMode::kPrefill,
          MlpActivation::kSwiGLU};
}

FamilyConfig RopeFamily() {
  return {"BM_RoPE", LayerType::kRoPE, AttentionMode::kPrefill,
          MlpActivation::kSwiGLU};
}

FamilyConfig SwiGluMlpFamily() {
  return {"BM_GatedMlp_SwiGLU", LayerType::kGatedMlp, AttentionMode::kPrefill,
          MlpActivation::kSwiGLU};
}

FamilyConfig BlockPrefillFamily() {
  return {"BM_TransformerBlock_Prefill", LayerType::kTransformerBlock,
          AttentionMode::kPrefill, MlpActivation::kSwiGLU};
}

FamilyConfig BlockDecodeFamily() {
  return {"BM_TransformerBlock_Decode", LayerType::kTransformerBlock,
          AttentionMode::kDecode, MlpActivation::kSwiGLU};
}

// Returns a preset called `name`: H=4, D=16, F=96, N_kv=2.
ModelPreset SmallPreset(std::string name) {
  ModelPreset preset;
  preset.name = std::move(name);
  preset.num_heads = 4;
  preset.head_dim = 16;
  preset.hidden_dim = 96;
  preset.num_kv_heads = 2;
  return preset;
}

// Returns the suite of one case: `family` on `preset` at B=2, S=64, 4 threads.
SuiteSpec SingleCaseSuite(const FamilyConfig& family,
                          const ModelPreset& preset) {
  SuiteSpec suite;
  suite.families = {family};
  suite.presets = {preset};
  suite.batch_sizes = {2};
  suite.seq_lengths = {64};
  suite.thread_counts = {4};
  return suite;
}

TEST(ExpandSuiteTest, NamesCaseByFamilyPresetAndOperatingPoint) {
  const SuiteSpec suite =
      SingleCaseSuite(RmsNormFamily(), SmallPreset("Small"));

  std::vector<BenchmarkCase> cases = ExpandSuite(suite);

  ASSERT_THAT(cases, SizeIs(1));
  EXPECT_EQ(cases.front().name, "BM_RmsNorm/Small/B:2/S:64/Threads:4");
  EXPECT_EQ(cases.front().num_threads, 4);
}

TEST(ExpandSuiteTest, CopiesPresetIntoLayerSpec) {
  ModelPreset preset = SmallPreset("Small");
  preset.embed_dim = 48;
  preset.eps = 1e-6f;
  preset.qk_norm = true;
  preset.post_norms = true;
  preset.sliding_window = 32;
  const SuiteSpec suite = SingleCaseSuite(BlockPrefillFamily(), preset);

  std::vector<BenchmarkCase> cases = ExpandSuite(suite);

  ASSERT_THAT(cases, SizeIs(1));
  const LayerSpec& spec = cases.front().spec;
  EXPECT_EQ(spec.model_name, "Small");
  EXPECT_EQ(spec.layer_type, LayerType::kTransformerBlock);
  EXPECT_EQ(spec.batch_size, 2);
  EXPECT_EQ(spec.num_heads, 4);
  EXPECT_EQ(spec.head_dim, 16);
  EXPECT_EQ(spec.hidden_dim, 96);
  EXPECT_EQ(spec.num_kv_heads, 2);
  EXPECT_EQ(spec.embed_dim, 48);
  EXPECT_EQ(spec.sliding_window, 32);
  EXPECT_EQ(spec.eps, 1e-6f);
  EXPECT_TRUE(spec.qk_norm);
  EXPECT_TRUE(spec.post_norms);
  EXPECT_EQ(spec.dtype, DataType::kFP32);
}

TEST(ExpandSuiteTest, PrefillFamilyQueriesTheWholeSequence) {
  const SuiteSpec suite =
      SingleCaseSuite(BlockPrefillFamily(), SmallPreset("Small"));

  std::vector<BenchmarkCase> cases = ExpandSuite(suite);

  ASSERT_THAT(cases, SizeIs(1));
  EXPECT_EQ(cases.front().spec.attention_mode, AttentionMode::kPrefill);
  EXPECT_EQ(cases.front().spec.seq_len_kv, 64);
  EXPECT_EQ(cases.front().spec.seq_len_q, 64);
}

TEST(ExpandSuiteTest, DecodeFamilyQueriesOneTokenAgainstTheSequence) {
  const SuiteSpec suite =
      SingleCaseSuite(BlockDecodeFamily(), SmallPreset("Small"));

  std::vector<BenchmarkCase> cases = ExpandSuite(suite);

  ASSERT_THAT(cases, SizeIs(1));
  EXPECT_EQ(cases.front().spec.attention_mode, AttentionMode::kDecode);
  EXPECT_EQ(cases.front().spec.seq_len_kv, 64);
  EXPECT_EQ(cases.front().spec.seq_len_q, 1);
}

TEST(ExpandSuiteTest, BlockFamilyUsesPresetMlpActivation) {
  ModelPreset preset = SmallPreset("Small");
  preset.block_mlp_activation = MlpActivation::kGeGLU;
  const SuiteSpec suite = SingleCaseSuite(BlockPrefillFamily(), preset);

  std::vector<BenchmarkCase> cases = ExpandSuite(suite);

  ASSERT_THAT(cases, SizeIs(1));
  EXPECT_EQ(cases.front().spec.mlp_activation, MlpActivation::kGeGLU);
}

TEST(ExpandSuiteTest, BlockFamilyWithoutPresetActivationKeepsFamilyActivation) {
  const SuiteSpec suite =
      SingleCaseSuite(BlockPrefillFamily(), SmallPreset("Small"));

  std::vector<BenchmarkCase> cases = ExpandSuite(suite);

  ASSERT_THAT(cases, SizeIs(1));
  EXPECT_EQ(cases.front().spec.mlp_activation, MlpActivation::kSwiGLU);
}

TEST(ExpandSuiteTest, MlpFamilyKeepsItsOwnActivation) {
  ModelPreset preset = SmallPreset("Small");
  preset.block_mlp_activation = MlpActivation::kGeGLU;
  const SuiteSpec suite = SingleCaseSuite(SwiGluMlpFamily(), preset);

  std::vector<BenchmarkCase> cases = ExpandSuite(suite);

  ASSERT_THAT(cases, SizeIs(1));
  EXPECT_EQ(cases.front().spec.mlp_activation, MlpActivation::kSwiGLU);
}

TEST(ExpandSuiteTest, NestsPresetsInsideFamilies) {
  SuiteSpec suite = SingleCaseSuite(RmsNormFamily(), SmallPreset("First"));
  suite.families = {RmsNormFamily(), RopeFamily()};
  suite.presets = {SmallPreset("First"), SmallPreset("Second")};

  std::vector<BenchmarkCase> cases = ExpandSuite(suite);

  EXPECT_THAT(
      cases,
      ElementsAre(
          Field(&BenchmarkCase::name, "BM_RmsNorm/First/B:2/S:64/Threads:4"),
          Field(&BenchmarkCase::name, "BM_RmsNorm/Second/B:2/S:64/Threads:4"),
          Field(&BenchmarkCase::name, "BM_RoPE/First/B:2/S:64/Threads:4"),
          Field(&BenchmarkCase::name, "BM_RoPE/Second/B:2/S:64/Threads:4")));
}

TEST(ExpandSuiteTest, NestsThreadCountsInsideSequenceLengthsInsideBatchSizes) {
  SuiteSpec suite = SingleCaseSuite(RmsNormFamily(), SmallPreset("Small"));
  suite.batch_sizes = {1, 2};
  suite.seq_lengths = {8, 16};
  suite.thread_counts = {1, 4};

  std::vector<BenchmarkCase> cases = ExpandSuite(suite);

  EXPECT_THAT(
      cases,
      ElementsAre(
          Field(&BenchmarkCase::name, "BM_RmsNorm/Small/B:1/S:8/Threads:1"),
          Field(&BenchmarkCase::name, "BM_RmsNorm/Small/B:1/S:8/Threads:4"),
          Field(&BenchmarkCase::name, "BM_RmsNorm/Small/B:1/S:16/Threads:1"),
          Field(&BenchmarkCase::name, "BM_RmsNorm/Small/B:1/S:16/Threads:4"),
          Field(&BenchmarkCase::name, "BM_RmsNorm/Small/B:2/S:8/Threads:1"),
          Field(&BenchmarkCase::name, "BM_RmsNorm/Small/B:2/S:8/Threads:4"),
          Field(&BenchmarkCase::name, "BM_RmsNorm/Small/B:2/S:16/Threads:1"),
          Field(&BenchmarkCase::name, "BM_RmsNorm/Small/B:2/S:16/Threads:4")));
}

// Matches a BenchmarkCase whose layer spec passes LayerSpec::Validate().
MATCHER(HasValidLayerSpec, "has a layer spec that passes Validate()") {
  const absl::Status status = arg.spec.Validate();
  if (!status.ok()) {
    *result_listener << arg.name << ": " << status.message();
  }
  return status.ok();
}

TEST(ExpandSuiteTest, EveryBuiltInCaseHasValidLayerSpec) {
  absl::StatusOr<SuiteSpec> suite = BuildSuiteSpec("");
  ASSERT_OK(suite.status());

  std::vector<BenchmarkCase> cases = ExpandSuite(*suite);

  // 9 families x 14 presets x 2 batch sizes x 4 sequence lengths x 5 thread
  // counts.
  EXPECT_THAT(cases, AllOf(SizeIs(5040), Each(HasValidLayerSpec())));
}

TEST(ExpandSuiteTest, SuiteWithoutPresetsHasNoCases) {
  SuiteSpec suite = SingleCaseSuite(RmsNormFamily(), SmallPreset("Small"));
  suite.presets.clear();

  EXPECT_THAT(ExpandSuite(suite), SizeIs(0));
}

struct RuntimeBackendCase {
  std::string name;
  std::string backend_flag;
  std::string argv0;
  RuntimeBackend expected_backend;
};

class ResolvedRuntimeBackendTest
    : public testing::TestWithParam<RuntimeBackendCase> {};

TEST_P(ResolvedRuntimeBackendTest, ParseReturnsBackend) {
  absl::StatusOr<RuntimeBackend> backend =
      ParseRuntimeBackend(GetParam().backend_flag, GetParam().argv0);

  ASSERT_OK(backend.status());
  EXPECT_EQ(*backend, GetParam().expected_backend);
}

INSTANTIATE_TEST_SUITE_P(
    FlagAndBinaryName, ResolvedRuntimeBackendTest,
    Values(RuntimeBackendCase{.name = "EmptyFlagDefaultsToXnnpack",
                              .backend_flag = "",
                              .argv0 = "layer_bench",
                              .expected_backend = RuntimeBackend::kXnnpack},
           RuntimeBackendCase{.name = "EmptyFlagFollowsYnnpackBinarySuffix",
                              .backend_flag = "",
                              .argv0 = "/opt/bin/layer_bench_ynnpack",
                              .expected_backend = RuntimeBackend::kYnnpack},
           RuntimeBackendCase{.name = "SuffixMustEndBinaryName",
                              .backend_flag = "",
                              .argv0 = "/opt/bin/layer_bench_ynnpack_v2",
                              .expected_backend = RuntimeBackend::kXnnpack},
           RuntimeBackendCase{.name = "XnnpackFlagOverridesBinarySuffix",
                              .backend_flag = "xnnpack",
                              .argv0 = "/opt/bin/layer_bench_ynnpack",
                              .expected_backend = RuntimeBackend::kXnnpack},
           RuntimeBackendCase{.name = "YnnpackFlagOverridesBinaryName",
                              .backend_flag = "ynnpack",
                              .argv0 = "layer_bench",
                              .expected_backend = RuntimeBackend::kYnnpack}),
    [](const testing::TestParamInfo<RuntimeBackendCase>& info) {
      return info.param.name;
    });

TEST(ParseRuntimeBackendTest, UnsupportedFlagReturnsInvalidArgument) {
  absl::StatusOr<RuntimeBackend> backend = ParseRuntimeBackend("tflite");

  EXPECT_EQ(backend.status().code(), absl::StatusCode::kInvalidArgument)
      << backend.status();
}

}  // namespace
}  // namespace layer_ml_bench
