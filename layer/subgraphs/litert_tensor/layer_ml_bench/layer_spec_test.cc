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

#include "tensor/benchmarks/layer/layer_spec.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensor/benchmarks/layer/fake_layer_backend.h"
#include "tensor/benchmarks/layer/layer_backend.h"

namespace layer_ml_bench {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::testing::Values;

constexpr uint32_t kSeed = 42;
constexpr uint32_t kOtherSeed = 99;

// LLaMA3-8B transformer block in prefill: B=1, S=T=512, H=32, D=128, N_kv=8,
// F=14336, M=H*D=4096, FP32. Reserved for analytical tests, which allocate no
// buffers.
LayerSpec CreateLlama3Spec() {
  LayerSpec spec;
  spec.model_name = "LLaMA3-8B";
  spec.layer_type = LayerType::kTransformerBlock;
  spec.attention_mode = AttentionMode::kPrefill;
  spec.mlp_activation = MlpActivation::kSwiGLU;
  spec.batch_size = 1;
  spec.seq_len_kv = 512;
  spec.seq_len_q = 512;
  spec.num_heads = 32;
  spec.head_dim = 128;
  spec.num_kv_heads = 8;
  spec.hidden_dim = 14336;
  spec.dtype = DataType::kFP32;
  return spec;
}

// Gemma4-E2B transformer block in prefill: B=1, S=T=1024, H=8, D=256, N_kv=1,
// F=6144, M=1536 decoupled from H*D=2048, sliding window 512, QK-norm and
// post-norms, FP32. Reserved for analytical tests, which allocate no buffers.
LayerSpec CreateGemma4E2bSpec() {
  LayerSpec spec;
  spec.model_name = "Gemma4-E2B";
  spec.layer_type = LayerType::kTransformerBlock;
  spec.attention_mode = AttentionMode::kPrefill;
  spec.mlp_activation = MlpActivation::kGeGLU;
  spec.batch_size = 1;
  spec.seq_len_kv = 1024;
  spec.seq_len_q = 1024;
  spec.embed_dim = 1536;
  spec.num_heads = 8;
  spec.head_dim = 256;
  spec.num_kv_heads = 1;
  spec.hidden_dim = 6144;
  spec.sliding_window = 512;
  spec.qk_norm = true;
  spec.post_norms = true;
  spec.dtype = DataType::kFP32;
  return spec;
}

// Small layer whose seeded buffers total under 100 KB: B=1, S=T=8, H=4, D=16,
// N_kv=2, F=96, and M=48 decoupled from H*D=64 so that buffer sizes
// distinguish the model dimension from the query dimension.
LayerSpec CreateSmallSpec(LayerType layer_type, AttentionMode attention_mode =
                                                    AttentionMode::kPrefill) {
  LayerSpec spec;
  spec.model_name = "TestModel";
  spec.layer_type = layer_type;
  spec.attention_mode = attention_mode;
  spec.batch_size = 1;
  spec.seq_len_kv = 8;
  spec.seq_len_q = 8;
  spec.embed_dim = 48;
  spec.num_heads = 4;
  spec.head_dim = 16;
  spec.num_kv_heads = 2;
  spec.hidden_dim = 96;
  return spec;
}

// Returns the element count of every non-empty buffer of `data`, keyed by
// field name, so that a test can assert the complete set of allocated buffers.
std::map<std::string, size_t> NonEmptyBufferSizes(const SeededLayerData& data) {
  const std::pair<std::string, const std::vector<float>*> buffers[] = {
      {"input", &data.input},
      {"q", &data.q},
      {"k", &data.k},
      {"v", &data.v},
      {"rms_scale", &data.rms_scale},
      {"rope_cos", &data.rope_cos},
      {"rope_sin", &data.rope_sin},
      {"w_q", &data.w_q},
      {"w_k", &data.w_k},
      {"w_v", &data.w_v},
      {"w_o", &data.w_o},
      {"w_gate", &data.w_gate},
      {"w_up", &data.w_up},
      {"w_down", &data.w_down},
      {"ffn_rms_scale", &data.ffn_rms_scale},
      {"q_norm_scale", &data.q_norm_scale},
      {"k_norm_scale", &data.k_norm_scale},
      {"post_attn_rms_scale", &data.post_attn_rms_scale},
      {"post_ffn_rms_scale", &data.post_ffn_rms_scale},
  };
  std::map<std::string, size_t> sizes;
  for (const auto& [name, buffer] : buffers) {
    if (!buffer->empty()) {
      sizes.emplace(name, buffer->size());
    }
  }
  return sizes;
}

// A named single-field edit applied to an otherwise valid LayerSpec.
struct LayerSpecMutation {
  std::string name;
  void (*mutate)(LayerSpec&);
};

std::string MutationName(
    const testing::TestParamInfo<LayerSpecMutation>& info) {
  return info.param.name;
}

TEST(LayerSpecTest, ValidateAcceptsFullScaleTransformerBlock) {
  EXPECT_EQ(CreateLlama3Spec().Validate(), absl::OkStatus());
}

class ValidLayerSpecTest : public testing::TestWithParam<LayerSpecMutation> {};

TEST_P(ValidLayerSpecTest, ValidateAccepts) {
  LayerSpec spec = CreateLlama3Spec();
  GetParam().mutate(spec);

  EXPECT_EQ(spec.Validate(), absl::OkStatus());
}

INSTANTIATE_TEST_SUITE_P(
    SupportedVariants, ValidLayerSpecTest,
    Values(LayerSpecMutation{"StandardAttentionPrefillWithShorterQuery",
                             [](LayerSpec& spec) {
                               spec.layer_type = LayerType::kStandardAttention;
                               spec.seq_len_q = 256;
                             }},
           LayerSpecMutation{"TransformerBlockDecodeWithUnusedQueryLength",
                             [](LayerSpec& spec) {
                               spec.attention_mode = AttentionMode::kDecode;
                               spec.seq_len_q = 256;
                             }},
           LayerSpecMutation{"DecodeOptimizedAttentionInDecode",
                             [](LayerSpec& spec) {
                               spec.layer_type =
                                   LayerType::kDecodeOptimizedAttention;
                               spec.attention_mode = AttentionMode::kDecode;
                             }},
           LayerSpecMutation{"DecodeWithSlidingWindowAndExtraNorms",
                             [](LayerSpec& spec) {
                               spec.attention_mode = AttentionMode::kDecode;
                               spec.seq_len_kv = 2048;
                               spec.sliding_window = 512;
                               spec.qk_norm = true;
                               spec.post_norms = true;
                             }},
           LayerSpecMutation{"ExplicitEmbedDim",
                             [](LayerSpec& spec) { spec.embed_dim = 1536; }}),
    MutationName);

class InvalidLayerSpecTest : public testing::TestWithParam<LayerSpecMutation> {
};

TEST_P(InvalidLayerSpecTest, ValidateReturnsInvalidArgument) {
  LayerSpec spec = CreateLlama3Spec();
  GetParam().mutate(spec);

  EXPECT_EQ(spec.Validate().code(), absl::StatusCode::kInvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    ViolatedInvariants, InvalidLayerSpecTest,
    Values(LayerSpecMutation{"ZeroBatchSize",
                             [](LayerSpec& spec) { spec.batch_size = 0; }},
           LayerSpecMutation{"NegativeSeqLenKv",
                             [](LayerSpec& spec) { spec.seq_len_kv = -1; }},
           LayerSpecMutation{"ZeroSeqLenQ",
                             [](LayerSpec& spec) { spec.seq_len_q = 0; }},
           LayerSpecMutation{"ZeroNumHeads",
                             [](LayerSpec& spec) { spec.num_heads = 0; }},
           LayerSpecMutation{"ZeroHeadDim",
                             [](LayerSpec& spec) { spec.head_dim = 0; }},
           LayerSpecMutation{"ZeroNumKvHeads",
                             [](LayerSpec& spec) { spec.num_kv_heads = 0; }},
           LayerSpecMutation{"ZeroHiddenDim",
                             [](LayerSpec& spec) { spec.hidden_dim = 0; }},
           LayerSpecMutation{"NegativeEmbedDim",
                             [](LayerSpec& spec) { spec.embed_dim = -1; }},
           LayerSpecMutation{"NegativeSlidingWindow",
                             [](LayerSpec& spec) { spec.sliding_window = -1; }},
           LayerSpecMutation{"HeadsNotDivisibleByKvHeads",
                             [](LayerSpec& spec) { spec.num_kv_heads = 7; }},
           LayerSpecMutation{"OddHeadDim",
                             [](LayerSpec& spec) { spec.head_dim = 127; }},
           LayerSpecMutation{"ZeroEps",
                             [](LayerSpec& spec) { spec.eps = 0.0f; }},
           LayerSpecMutation{"NegativeEps",
                             [](LayerSpec& spec) { spec.eps = -1e-5f; }},
           LayerSpecMutation{"NanEps",
                             [](LayerSpec& spec) {
                               spec.eps =
                                   std::numeric_limits<float>::quiet_NaN();
                             }},
           LayerSpecMutation{"InfiniteEps",
                             [](LayerSpec& spec) {
                               spec.eps =
                                   std::numeric_limits<float>::infinity();
                             }},
           LayerSpecMutation{"PrefillQueryLongerThanContext",
                             [](LayerSpec& spec) {
                               spec.layer_type = LayerType::kStandardAttention;
                               spec.seq_len_kv = 256;
                             }},
           LayerSpecMutation{"DecodeOptimizedAttentionInPrefill",
                             [](LayerSpec& spec) {
                               spec.layer_type =
                                   LayerType::kDecodeOptimizedAttention;
                             }},
           LayerSpecMutation{"TransformerBlockPrefillQueryShorterThanContext",
                             [](LayerSpec& spec) { spec.seq_len_q = 256; }}),
    MutationName);

struct EffectiveLengthCase {
  std::string name;
  LayerType layer_type;
  AttentionMode attention_mode;
  int64_t seq_len_q;
  int64_t seq_len_kv;
  int64_t sliding_window;
  int64_t expected_seq_len_q;
  int64_t expected_seq_len_kv;
};

class EffectiveSeqLenTest : public testing::TestWithParam<EffectiveLengthCase> {
};

TEST_P(EffectiveSeqLenTest, ResolvesQueryAndContextLengths) {
  const EffectiveLengthCase& tc = GetParam();
  LayerSpec spec = CreateLlama3Spec();
  spec.layer_type = tc.layer_type;
  spec.attention_mode = tc.attention_mode;
  spec.seq_len_q = tc.seq_len_q;
  spec.seq_len_kv = tc.seq_len_kv;
  spec.sliding_window = tc.sliding_window;

  EXPECT_EQ(spec.EffectiveSeqLenQ(), tc.expected_seq_len_q);
  EXPECT_EQ(spec.EffectiveSeqLenKV(), tc.expected_seq_len_kv);
}

// The decode-optimized layer cases keep kPrefill so that the layer type alone,
// not the attention mode, must trigger the decode rules.
INSTANTIATE_TEST_SUITE_P(
    ModesAndWindows, EffectiveSeqLenTest,
    Values(EffectiveLengthCase{.name = "PrefillKeepsRequestedLengths",
                               .layer_type = LayerType::kStandardAttention,
                               .attention_mode = AttentionMode::kPrefill,
                               .seq_len_q = 256,
                               .seq_len_kv = 512,
                               .sliding_window = 0,
                               .expected_seq_len_q = 256,
                               .expected_seq_len_kv = 512},
           EffectiveLengthCase{.name = "PrefillIgnoresSlidingWindow",
                               .layer_type = LayerType::kTransformerBlock,
                               .attention_mode = AttentionMode::kPrefill,
                               .seq_len_q = 512,
                               .seq_len_kv = 512,
                               .sliding_window = 128,
                               .expected_seq_len_q = 512,
                               .expected_seq_len_kv = 512},
           EffectiveLengthCase{.name = "DecodeAttendsFromSingleToken",
                               .layer_type = LayerType::kTransformerBlock,
                               .attention_mode = AttentionMode::kDecode,
                               .seq_len_q = 512,
                               .seq_len_kv = 512,
                               .sliding_window = 0,
                               .expected_seq_len_q = 1,
                               .expected_seq_len_kv = 512},
           EffectiveLengthCase{
               .name = "DecodeOptimizedLayerAttendsFromSingleToken",
               .layer_type = LayerType::kDecodeOptimizedAttention,
               .attention_mode = AttentionMode::kPrefill,
               .seq_len_q = 1024,
               .seq_len_kv = 512,
               .sliding_window = 0,
               .expected_seq_len_q = 1,
               .expected_seq_len_kv = 512},
           EffectiveLengthCase{.name = "DecodeLimitsContextToSlidingWindow",
                               .layer_type = LayerType::kTransformerBlock,
                               .attention_mode = AttentionMode::kDecode,
                               .seq_len_q = 1,
                               .seq_len_kv = 2048,
                               .sliding_window = 512,
                               .expected_seq_len_q = 1,
                               .expected_seq_len_kv = 512},
           EffectiveLengthCase{
               .name = "DecodeOptimizedLayerLimitsContextToSlidingWindow",
               .layer_type = LayerType::kDecodeOptimizedAttention,
               .attention_mode = AttentionMode::kPrefill,
               .seq_len_q = 1,
               .seq_len_kv = 2048,
               .sliding_window = 512,
               .expected_seq_len_q = 1,
               .expected_seq_len_kv = 512},
           EffectiveLengthCase{
               .name = "DecodeKeepsContextShorterThanSlidingWindow",
               .layer_type = LayerType::kTransformerBlock,
               .attention_mode = AttentionMode::kDecode,
               .seq_len_q = 1,
               .seq_len_kv = 512,
               .sliding_window = 2048,
               .expected_seq_len_q = 1,
               .expected_seq_len_kv = 512}),
    [](const testing::TestParamInfo<EffectiveLengthCase>& info) {
      return info.param.name;
    });

TEST(LayerSpecTest, ModelDimDefaultsToQueryDim) {
  LayerSpec spec = CreateLlama3Spec();

  // H=32, D=128, N_kv=8.
  EXPECT_EQ(spec.QDim(), 4096);
  EXPECT_EQ(spec.KVDim(), 1024);
  EXPECT_EQ(spec.ModelDim(), 4096);
}

TEST(LayerSpecTest, ExplicitEmbedDimDecouplesModelDimFromQueryDim) {
  LayerSpec spec = CreateGemma4E2bSpec();

  // H=8, D=256, N_kv=1, embed_dim=1536.
  EXPECT_EQ(spec.QDim(), 2048);
  EXPECT_EQ(spec.KVDim(), 256);
  EXPECT_EQ(spec.ModelDim(), 1536);
}

struct OutputGeometryCase {
  std::string name;
  LayerType layer_type;
  AttentionMode attention_mode;
  int64_t expected_output_dim;
  std::vector<int64_t> expected_output_shape;
};

class OutputGeometryTest : public testing::TestWithParam<OutputGeometryCase> {};

TEST_P(OutputGeometryTest, ReportsOutputDimAndShape) {
  const OutputGeometryCase& tc = GetParam();
  LayerSpec spec = CreateGemma4E2bSpec();
  spec.layer_type = tc.layer_type;
  spec.attention_mode = tc.attention_mode;

  EXPECT_EQ(spec.OutputDim(), tc.expected_output_dim);
  EXPECT_EQ(spec.OutputShape(), tc.expected_output_shape);
}

// Gemma4-E2B: B=1, T=1024 (1 in decode), H=8, D=256, H*D=2048, M=1536.
INSTANTIATE_TEST_SUITE_P(
    DecoupledEmbedDim, OutputGeometryTest,
    Values(OutputGeometryCase{.name = "RmsNormPrefill",
                              .layer_type = LayerType::kRmsNorm,
                              .attention_mode = AttentionMode::kPrefill,
                              .expected_output_dim = 1536,
                              .expected_output_shape = {1, 1024, 1536}},
           OutputGeometryCase{.name = "RoPEPrefill",
                              .layer_type = LayerType::kRoPE,
                              .attention_mode = AttentionMode::kPrefill,
                              .expected_output_dim = 2048,
                              .expected_output_shape = {1, 1024, 2048}},
           OutputGeometryCase{.name = "GatedMlpPrefill",
                              .layer_type = LayerType::kGatedMlp,
                              .attention_mode = AttentionMode::kPrefill,
                              .expected_output_dim = 1536,
                              .expected_output_shape = {1, 1024, 1536}},
           OutputGeometryCase{.name = "StandardAttentionPrefill",
                              .layer_type = LayerType::kStandardAttention,
                              .attention_mode = AttentionMode::kPrefill,
                              .expected_output_dim = 2048,
                              .expected_output_shape = {1, 8, 1024, 256}},
           OutputGeometryCase{.name = "StandardAttentionDecode",
                              .layer_type = LayerType::kStandardAttention,
                              .attention_mode = AttentionMode::kDecode,
                              .expected_output_dim = 2048,
                              .expected_output_shape = {1, 8, 1, 256}},
           OutputGeometryCase{
               .name = "DecodeOptimizedAttentionDecode",
               .layer_type = LayerType::kDecodeOptimizedAttention,
               .attention_mode = AttentionMode::kDecode,
               .expected_output_dim = 2048,
               .expected_output_shape = {1, 8, 1, 256}},
           OutputGeometryCase{.name = "TransformerBlockPrefill",
                              .layer_type = LayerType::kTransformerBlock,
                              .attention_mode = AttentionMode::kPrefill,
                              .expected_output_dim = 1536,
                              .expected_output_shape = {1, 1024, 1536}},
           OutputGeometryCase{.name = "TransformerBlockDecode",
                              .layer_type = LayerType::kTransformerBlock,
                              .attention_mode = AttentionMode::kDecode,
                              .expected_output_dim = 1536,
                              .expected_output_shape = {1, 1, 1536}}),
    [](const testing::TestParamInfo<OutputGeometryCase>& info) {
      return info.param.name;
    });

TEST(LayerSpecTest, OutputShapeLeadsWithBatchSize) {
  LayerSpec spec = CreateLlama3Spec();
  spec.batch_size = 4;

  EXPECT_THAT(spec.OutputShape(), ElementsAre(4, 512, 4096));
}

struct AnalyticalCostCase {
  std::string name;
  LayerType layer_type;
  AttentionMode attention_mode;
  int64_t expected_flops;
  int64_t expected_bytes;
  double expected_intensity;
};

class AnalyticalCostTest : public testing::TestWithParam<AnalyticalCostCase> {};

TEST_P(AnalyticalCostTest, ReportsFlopsBytesAndArithmeticIntensity) {
  const AnalyticalCostCase& tc = GetParam();
  LayerSpec spec = CreateLlama3Spec();
  spec.layer_type = tc.layer_type;
  spec.attention_mode = tc.attention_mode;

  EXPECT_EQ(spec.TotalFlops(), tc.expected_flops);
  EXPECT_EQ(spec.TotalMemoryBytes(), tc.expected_bytes);
  EXPECT_NEAR(spec.ArithmeticIntensity(), tc.expected_intensity, 1e-9);
}

// LLaMA3-8B in FP32 (4 bytes per element): B=1, S=512, T=512 in prefill and 1
// in decode, H=32, D=128, N_kv=8, M=4096, F=14336. The intensity is FLOPs
// divided by bytes.
INSTANTIATE_TEST_SUITE_P(
    Llama3, AnalyticalCostTest,
    Values(
        // FLOPs: 4*T*M. Bytes: 4 * (2*T*M activations + M scales).
        AnalyticalCostCase{.name = "RmsNormPrefill",
                           .layer_type = LayerType::kRmsNorm,
                           .attention_mode = AttentionMode::kPrefill,
                           .expected_flops = 8388608,
                           .expected_bytes = 16793600,
                           .expected_intensity = 0.499512195122},
        // FLOPs: 3*T*H*D. Bytes: 4 * (2*T*H*D activations + 2*T*D tables).
        AnalyticalCostCase{.name = "RoPEPrefill",
                           .layer_type = LayerType::kRoPE,
                           .attention_mode = AttentionMode::kPrefill,
                           .expected_flops = 6291456,
                           .expected_bytes = 17301504,
                           .expected_intensity = 0.363636363636},
        // FLOPs: 6*T*M*F + 3*T*F. Bytes: 4 * (2*T*M activations + 3*M*F
        // weights).
        AnalyticalCostCase{.name = "GatedMlpPrefill",
                           .layer_type = LayerType::kGatedMlp,
                           .attention_mode = AttentionMode::kPrefill,
                           .expected_flops = 180410646528,
                           .expected_bytes = 721420288,
                           .expected_intensity = 250.077034883721},
        // FLOPs: 4*H*T*S*D + 3*H*T*S. Bytes: 4 * (2*H*T*D queries and outputs
        // + 2*N_kv*S*D keys and values).
        AnalyticalCostCase{.name = "StandardAttentionPrefill",
                           .layer_type = LayerType::kStandardAttention,
                           .attention_mode = AttentionMode::kPrefill,
                           .expected_flops = 4320133120,
                           .expected_bytes = 20971520,
                           .expected_intensity = 206.0},
        // The prefill attention formulas with T=1.
        AnalyticalCostCase{.name = "StandardAttentionDecode",
                           .layer_type = LayerType::kStandardAttention,
                           .attention_mode = AttentionMode::kDecode,
                           .expected_flops = 8437760,
                           .expected_bytes = 4227072,
                           .expected_intensity = 1.996124031008},
        AnalyticalCostCase{.name = "DecodeOptimizedAttentionDecode",
                           .layer_type = LayerType::kDecodeOptimizedAttention,
                           .attention_mode = AttentionMode::kDecode,
                           .expected_flops = 8437760,
                           .expected_bytes = 4227072,
                           .expected_intensity = 1.996124031008},
        // FLOPs: two RMSNorms (2 * 8388608), Q/K/V projections (25769803776),
        // RoPE on Q and K (7864320), attention (4320133120), output projection
        // (17179869184), two residual adds (4194304) and the gated MLP
        // (180410646528). Bytes: 4 * (2*T*M + 2*N_kv*S*D activations) +
        // 4 * (2*T*D tables + T*S mask + 2*M scales) + 4 * 218103808 weights.
        AnalyticalCostCase{.name = "TransformerBlockPrefill",
                           .layer_type = LayerType::kTransformerBlock,
                           .attention_mode = AttentionMode::kPrefill,
                           .expected_flops = 227709288448,
                           .expected_bytes = 894992384,
                           .expected_intensity = 254.425951012338},
        // FLOPs: two RMSNorms (2 * 16384), Q/K/V projections (50331648), RoPE
        // on Q and K (15360), attention (8437760), output projection
        // (33554432), two residual adds (8192) and the gated MLP (352364544).
        // Bytes: the prefill formula without the T*S mask term.
        AnalyticalCostCase{.name = "TransformerBlockDecode",
                           .layer_type = LayerType::kTransformerBlock,
                           .attention_mode = AttentionMode::kDecode,
                           .expected_flops = 444744704,
                           .expected_bytes = 876676096,
                           .expected_intensity = 0.507307894021}),
    [](const testing::TestParamInfo<AnalyticalCostCase>& info) {
      return info.param.name;
    });

TEST(LayerSpecTest, PrefillBlockCostCountsExtraNormsAndDecoupledEmbedDim) {
  LayerSpec spec = CreateGemma4E2bSpec();

  // T=S=1024, M=1536, H*D=2048, N_kv*D=256, F=6144, with QK-norm and post-norm
  // terms added to the block formulas.
  EXPECT_EQ(spec.TotalFlops(), 81156374528);
  EXPECT_EQ(spec.TotalMemoryBytes(), 162555904);
}

TEST(LayerSpecTest, DecodeBlockCostUsesSlidingWindowContext) {
  LayerSpec spec = CreateGemma4E2bSpec();
  spec.attention_mode = AttentionMode::kDecode;

  // T=1 attends over the 512-token window rather than the 1024-token context.
  EXPECT_EQ(spec.TotalFlops(), 75047680);
  EXPECT_EQ(spec.TotalMemoryBytes(), 142647296);
}

TEST(LayerSpecTest, BatchSizeScalesMlpWorkAndActivationsButNotWeights) {
  LayerSpec spec = CreateLlama3Spec();
  spec.layer_type = LayerType::kGatedMlp;
  spec.attention_mode = AttentionMode::kDecode;
  spec.batch_size = 4;

  // Four tokens (B=4, T=1) do four times the 352364544 FLOPs of one token.
  // They touch 4*2*M = 32768 activation elements, while the 3*M*F = 176160768
  // weight elements are shared by the batch.
  EXPECT_EQ(spec.TotalFlops(), 1409458176);
  EXPECT_EQ(spec.TotalMemoryBytes(), 704774144);
}

TEST(LayerSpecTest, BatchSizeScalesBlockActivationsButNotWeightsOrMask) {
  LayerSpec spec = CreateLlama3Spec();
  spec.batch_size = 2;

  // Two sequences do twice the 227709288448 FLOPs of one. The second sequence
  // adds its 4194304 input/output and 1048576 KV-cache elements (4 bytes
  // each) to the 894992384 bytes of one sequence; weights, norm scales,
  // rotation tables, and the attention mask are shared by the batch.
  EXPECT_EQ(spec.TotalFlops(), 455418576896);
  EXPECT_EQ(spec.TotalMemoryBytes(), 915963904);
}

struct QuantizedMemoryCase {
  DataType dtype;
  int64_t expected_bytes;
};

class QuantizedMemoryTest : public testing::TestWithParam<QuantizedMemoryCase> {
};

TEST_P(QuantizedMemoryTest, ScalesActivationAndWeightBytes) {
  LayerSpec spec = CreateLlama3Spec();
  spec.layer_type = LayerType::kGatedMlp;
  spec.attention_mode = AttentionMode::kDecode;
  spec.dtype = GetParam().dtype;

  EXPECT_EQ(spec.TotalMemoryBytes(), GetParam().expected_bytes);
}

// GatedMlp with T=1 touches 2*M = 8192 activation elements and 3*M*F =
// 176160768 weight elements.
INSTANTIATE_TEST_SUITE_P(
    Precisions, QuantizedMemoryTest,
    Values(
        // 4-byte activations and weights.
        QuantizedMemoryCase{DataType::kFP32, 704675840},
        // 2-byte activations and weights.
        QuantizedMemoryCase{DataType::kFP16, 352337920},
        QuantizedMemoryCase{DataType::kBF16, 352337920},
        // 4-byte activations and 1-byte weights.
        QuantizedMemoryCase{DataType::kQD8, 176193536},
        // 4-byte activations and 4-bit weights.
        QuantizedMemoryCase{DataType::kQB4W, 88113152}),
    [](const testing::TestParamInfo<QuantizedMemoryCase>& info) {
      return DataTypeToString(info.param.dtype);
    });

TEST(LayerSpecTest, UndeclaredLayerTypeHasZeroAnalyticalCost) {
  LayerSpec spec = CreateLlama3Spec();
  spec.layer_type = static_cast<LayerType>(99);

  EXPECT_EQ(spec.TotalFlops(), 0);
  EXPECT_EQ(spec.TotalMemoryBytes(), 0);
  EXPECT_DOUBLE_EQ(spec.ArithmeticIntensity(), 0.0);
}

TEST(LayerSpecTest, ToStringListsModelLayerModeDimensionsAndPrecision) {
  EXPECT_EQ(CreateLlama3Spec().ToString(),
            "LLaMA3-8B_TransformerBlock_Prefill_B1_S512_T512_H32_D128_Nkv8_"
            "F14336_FP32");
}

TEST(LayerSpecTest, ToStringReportsSingleQueryTokenInDecode) {
  LayerSpec spec = CreateLlama3Spec();
  spec.attention_mode = AttentionMode::kDecode;

  EXPECT_EQ(spec.ToString(),
            "LLaMA3-8B_TransformerBlock_Decode_B1_S512_T1_H32_D128_Nkv8_"
            "F14336_FP32");
}

TEST(LayerSpecTest, DataTypeToStringNamesEachPrecision) {
  EXPECT_EQ(DataTypeToString(DataType::kFP32), "FP32");
  EXPECT_EQ(DataTypeToString(DataType::kFP16), "FP16");
  EXPECT_EQ(DataTypeToString(DataType::kBF16), "BF16");
  EXPECT_EQ(DataTypeToString(DataType::kQD8), "QD8");
  EXPECT_EQ(DataTypeToString(DataType::kQB4W), "QB4W");
}

TEST(LayerSpecTest, LayerTypeToStringNamesEachLayerType) {
  EXPECT_EQ(LayerTypeToString(LayerType::kRmsNorm), "RmsNorm");
  EXPECT_EQ(LayerTypeToString(LayerType::kRoPE), "RoPE");
  EXPECT_EQ(LayerTypeToString(LayerType::kGatedMlp), "GatedMlp");
  EXPECT_EQ(LayerTypeToString(LayerType::kStandardAttention),
            "StandardAttention");
  EXPECT_EQ(LayerTypeToString(LayerType::kDecodeOptimizedAttention),
            "DecodeOptimizedAttention");
  EXPECT_EQ(LayerTypeToString(LayerType::kTransformerBlock),
            "TransformerBlock");
}

TEST(LayerSpecTest, MlpActivationToStringNamesEachActivation) {
  EXPECT_EQ(MlpActivationToString(MlpActivation::kSwiGLU), "SwiGLU");
  EXPECT_EQ(MlpActivationToString(MlpActivation::kGeGLU), "GeGLU");
}

TEST(LayerSpecTest, AttentionModeToStringNamesEachMode) {
  EXPECT_EQ(AttentionModeToString(AttentionMode::kPrefill), "Prefill");
  EXPECT_EQ(AttentionModeToString(AttentionMode::kDecode), "Decode");
}

TEST(LayerSpecTest, UndeclaredEnumeratorsAreNamedUnknown) {
  EXPECT_EQ(DataTypeToString(static_cast<DataType>(99)), "Unknown");
  EXPECT_EQ(LayerTypeToString(static_cast<LayerType>(99)), "Unknown");
  EXPECT_EQ(MlpActivationToString(static_cast<MlpActivation>(99)), "Unknown");
  EXPECT_EQ(AttentionModeToString(static_cast<AttentionMode>(99)), "Unknown");
}

TEST(CreateSeededLayerDataTest, SameSeedReproducesBuffers) {
  LayerSpec spec = CreateSmallSpec(LayerType::kTransformerBlock);

  SeededLayerData first = CreateSeededLayerData(spec, kSeed);
  SeededLayerData second = CreateSeededLayerData(spec, kSeed);

  EXPECT_EQ(first.input, second.input);
  EXPECT_EQ(first.rms_scale, second.rms_scale);
  EXPECT_EQ(first.w_q, second.w_q);
  EXPECT_EQ(first.w_down, second.w_down);
}

TEST(CreateSeededLayerDataTest, DifferentSeedsProduceDifferentBuffers) {
  LayerSpec spec = CreateSmallSpec(LayerType::kTransformerBlock);

  SeededLayerData first = CreateSeededLayerData(spec, kSeed);
  SeededLayerData second = CreateSeededLayerData(spec, kOtherSeed);

  EXPECT_NE(first.input, second.input);
  EXPECT_NE(first.w_q, second.w_q);
}

TEST(CreateSeededLayerDataTest, RmsNormAllocatesInputAndScale) {
  SeededLayerData data =
      CreateSeededLayerData(CreateSmallSpec(LayerType::kRmsNorm), kSeed);

  // input: T*M = 8*48. rms_scale: M.
  EXPECT_THAT(NonEmptyBufferSizes(data),
              UnorderedElementsAre(Pair("input", 384), Pair("rms_scale", 48)));
}

TEST(CreateSeededLayerDataTest, RoPEAllocatesInputAndRotationTables) {
  SeededLayerData data =
      CreateSeededLayerData(CreateSmallSpec(LayerType::kRoPE), kSeed);

  // input: T*H*D = 8*64. rope_cos and rope_sin: T*D = 8*16.
  EXPECT_THAT(NonEmptyBufferSizes(data),
              UnorderedElementsAre(Pair("input", 512), Pair("rope_cos", 128),
                                   Pair("rope_sin", 128)));
}

TEST(CreateSeededLayerDataTest, GatedMlpAllocatesInputAndMlpWeights) {
  SeededLayerData data =
      CreateSeededLayerData(CreateSmallSpec(LayerType::kGatedMlp), kSeed);

  // input: T*M = 8*48. w_gate, w_up and w_down: F*M = 96*48.
  EXPECT_THAT(NonEmptyBufferSizes(data),
              UnorderedElementsAre(Pair("input", 384), Pair("w_gate", 4608),
                                   Pair("w_up", 4608), Pair("w_down", 4608)));
}

TEST(CreateSeededLayerDataTest, PrefillAttentionAllocatesQueryKeyValue) {
  SeededLayerData data = CreateSeededLayerData(
      CreateSmallSpec(LayerType::kStandardAttention), kSeed);

  // q: H*T*D = 4*8*16. k and v: N_kv*S*D = 2*8*16.
  EXPECT_THAT(
      NonEmptyBufferSizes(data),
      UnorderedElementsAre(Pair("q", 512), Pair("k", 256), Pair("v", 256)));
}

TEST(CreateSeededLayerDataTest, DecodeAttentionAllocatesSingleQueryToken) {
  SeededLayerData data = CreateSeededLayerData(
      CreateSmallSpec(LayerType::kDecodeOptimizedAttention,
                      AttentionMode::kDecode),
      kSeed);

  // q: H*1*D = 4*16. k and v: N_kv*S*D = 2*8*16.
  EXPECT_THAT(
      NonEmptyBufferSizes(data),
      UnorderedElementsAre(Pair("q", 64), Pair("k", 256), Pair("v", 256)));
}

TEST(CreateSeededLayerDataTest, PrefillBlockAllocatesWeightsWithoutKvCache) {
  SeededLayerData data = CreateSeededLayerData(
      CreateSmallSpec(LayerType::kTransformerBlock), kSeed);

  // input: T*M = 8*48. Norm scales: M. RoPE tables: T*D = 8*16. w_q and w_o:
  // H*D*M = 64*48. w_k and w_v: N_kv*D*M = 32*48. MLP weights: F*M = 96*48.
  EXPECT_THAT(
      NonEmptyBufferSizes(data),
      UnorderedElementsAre(
          Pair("input", 384), Pair("rms_scale", 48), Pair("rope_cos", 128),
          Pair("rope_sin", 128), Pair("w_q", 3072), Pair("w_k", 1536),
          Pair("w_v", 1536), Pair("w_o", 3072), Pair("w_gate", 4608),
          Pair("w_up", 4608), Pair("w_down", 4608), Pair("ffn_rms_scale", 48)));
}

TEST(CreateSeededLayerDataTest, DecodeBlockAllocatesPriorKvCache) {
  SeededLayerData data = CreateSeededLayerData(
      CreateSmallSpec(LayerType::kTransformerBlock, AttentionMode::kDecode),
      kSeed);

  // input: 1*M. RoPE tables: 1*D. k and v hold the S-1 = 7 prior tokens:
  // N_kv*7*D = 2*7*16. Weights match the prefill block.
  EXPECT_THAT(
      NonEmptyBufferSizes(data),
      UnorderedElementsAre(
          Pair("input", 48), Pair("k", 224), Pair("v", 224),
          Pair("rms_scale", 48), Pair("rope_cos", 16), Pair("rope_sin", 16),
          Pair("w_q", 3072), Pair("w_k", 1536), Pair("w_v", 1536),
          Pair("w_o", 3072), Pair("w_gate", 4608), Pair("w_up", 4608),
          Pair("w_down", 4608), Pair("ffn_rms_scale", 48)));
}

TEST(CreateSeededLayerDataTest, DecodeBlockKvCacheFollowsSlidingWindow) {
  LayerSpec spec =
      CreateSmallSpec(LayerType::kTransformerBlock, AttentionMode::kDecode);
  spec.sliding_window = 4;

  SeededLayerData data = CreateSeededLayerData(spec, kSeed);

  // The window keeps 4 tokens, 3 of them prior: N_kv*3*D = 2*3*16.
  EXPECT_THAT(data.k, SizeIs(96));
  EXPECT_THAT(data.v, SizeIs(96));
}

TEST(CreateSeededLayerDataTest, DecodeBlockWithSingleTokenContextHasNoKvCache) {
  LayerSpec spec =
      CreateSmallSpec(LayerType::kTransformerBlock, AttentionMode::kDecode);
  spec.seq_len_kv = 1;

  SeededLayerData data = CreateSeededLayerData(spec, kSeed);

  EXPECT_THAT(data.k, IsEmpty());
  EXPECT_THAT(data.v, IsEmpty());
}

TEST(CreateSeededLayerDataTest, QkNormAllocatesPerHeadScales) {
  LayerSpec spec = CreateSmallSpec(LayerType::kTransformerBlock);
  spec.qk_norm = true;

  SeededLayerData data = CreateSeededLayerData(spec, kSeed);

  // One scale per head dimension: D.
  EXPECT_THAT(data.q_norm_scale, SizeIs(16));
  EXPECT_THAT(data.k_norm_scale, SizeIs(16));
}

TEST(CreateSeededLayerDataTest, PostNormsAllocateModelDimScales) {
  LayerSpec spec = CreateSmallSpec(LayerType::kTransformerBlock);
  spec.post_norms = true;

  SeededLayerData data = CreateSeededLayerData(spec, kSeed);

  // One scale per model dimension: M.
  EXPECT_THAT(data.post_attn_rms_scale, SizeIs(48));
  EXPECT_THAT(data.post_ffn_rms_scale, SizeIs(48));
}

TEST(LayerBackendTest, GetThreadpoolDefaultsToNull) {
  FakeLayerBackend backend;

  EXPECT_EQ(backend.GetThreadpool(), nullptr);
}

TEST(LayerBackendTest, RvalueSetupForwardsToLvalueOverload) {
  LayerSpec spec = CreateSmallSpec(LayerType::kRmsNorm);
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);
  const std::vector<float> expected_input = data.input;
  FakeLayerBackend fake;
  // The rvalue overload is only reachable through the base interface: the
  // fake's own Setup() override hides it.
  LayerBackend& backend = fake;

  EXPECT_EQ(
      backend.Setup(spec, std::move(data), BackendOptions{.num_threads = 3}),
      absl::OkStatus());
  EXPECT_EQ(fake.setup_call_count(), 1);
  EXPECT_EQ(fake.data().input, expected_input);
  EXPECT_EQ(fake.options().num_threads, 3);
}

}  // namespace
}  // namespace layer_ml_bench
