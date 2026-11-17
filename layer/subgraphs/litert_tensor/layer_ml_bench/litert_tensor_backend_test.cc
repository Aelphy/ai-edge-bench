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

#include "tensor/benchmarks/layer/litert_tensor_backend.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/ynnpack/arithmetic.h"
#include "tensor/benchmarks/layer/fake_layer_backend.h"
#include "tensor/benchmarks/layer/layer_backend.h"
#include "tensor/benchmarks/layer/layer_builders.h"
#include "tensor/benchmarks/layer/layer_spec.h"
#include "tensor/benchmarks/layer/status_test_util.h"
#include "tensor/benchmarks/layer/test_matchers.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"
#include "tensor/examples/ops/transformer/transformer_ops_ynnpack.h"
#include "tensor/runners/common_nnpack/runner.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/runners/ynnpack/runner.h"
#include "tensor/tensor.h"

namespace layer_ml_bench::litert {
namespace {

using ::litert::tensor::Add;
using ::litert::tensor::FillAttentionMask;
using ::litert::tensor::FillRopeCosSin;
using ::litert::tensor::LockedBufferSpan;
using ::litert::tensor::NnpackRunner;
using ::litert::tensor::QkNorm;
using ::litert::tensor::RmsNorm;
using ::litert::tensor::TensorHandle;
using ::litert::tensor::Type;
using ::litert::tensor::XnnpackMixinTag;
using ::litert::tensor::XnnpackRunner;
using ::litert::tensor::YnnpackMixinTag;
using ::litert::tensor::YnnpackRunner;
using ::testing::AllOf;
using ::testing::Combine;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::IsEmpty;
using ::testing::Matcher;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::Pair;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::testing::Values;

constexpr uint32_t kSeed = 42;

// Largest per-element difference accepted between two float32 implementations
// of the same layer over seeded inputs of magnitude 0.1.
constexpr float kParityTolerance = 1e-4f;

// Largest per-element error accepted against hand-computed expectations.
constexpr float kAnalyticalTolerance = 1e-5f;

// Largest per-element error accepted in rotation tables, whose entries are
// single float32 evaluations of cos() or sin().
constexpr float kRopeTableTolerance = 1e-6f;

// Outputs whose every magnitude is below this bound count as all zeros.
constexpr float kZeroTolerance = 1e-5f;

// Additive attention bias that excludes a position.
constexpr float kMasked = std::numeric_limits<float>::lowest();

// Natural logarithm of 3: a query-key score of ln(3) against a score of 0
// yields softmax weights of 3/4 and 1/4.
constexpr float kLn3 = 1.0986123f;

std::string RuntimeName(RuntimeBackend runtime) {
  return runtime == RuntimeBackend::kYnnpack ? "Ynnpack" : "Xnnpack";
}

std::string RuntimeParamName(
    const testing::TestParamInfo<RuntimeBackend>& info) {
  return RuntimeName(info.param);
}

// Small layer with grouped-query attention: B=1, S=T=8, H=4, D=16, N_kv=2,
// F=64, M=H*D=64.
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

// Small layer shaped like Gemma4-E2B: B=1, S=T=8, H=4, D=16, N_kv=1, F=96,
// GeGLU, and M=48 decoupled from H*D=64.
LayerSpec CreateDecoupledEmbedDimSpec(LayerType layer_type,
                                      AttentionMode attention_mode) {
  LayerSpec spec;
  spec.model_name = "Gemma4-E2B";
  spec.layer_type = layer_type;
  spec.attention_mode = attention_mode;
  spec.mlp_activation = MlpActivation::kGeGLU;
  spec.batch_size = 1;
  spec.seq_len_kv = 8;
  spec.seq_len_q = 8;
  spec.embed_dim = 48;
  spec.num_heads = 4;
  spec.head_dim = 16;
  spec.num_kv_heads = 1;
  spec.hidden_dim = 96;
  spec.eps = 1e-6f;
  spec.dtype = DataType::kFP32;
  return spec;
}

// Single-head layer small enough for hand-computed expectations: B=1,
// H=N_kv=1, F=2, M=D=`head_dim` and S=T=`seq_len`.
LayerSpec CreateTinySpec(LayerType layer_type, AttentionMode attention_mode,
                         int64_t head_dim, int64_t seq_len) {
  LayerSpec spec;
  spec.model_name = "TestModel";
  spec.layer_type = layer_type;
  spec.attention_mode = attention_mode;
  spec.mlp_activation = MlpActivation::kSwiGLU;
  spec.batch_size = 1;
  spec.seq_len_kv = seq_len;
  spec.seq_len_q = seq_len;
  spec.num_heads = 1;
  spec.head_dim = head_dim;
  spec.num_kv_heads = 1;
  spec.hidden_dim = 2;
  spec.eps = 1e-5f;
  spec.dtype = DataType::kFP32;
  return spec;
}

// Weights of a transformer block built by CreateTinySpec() with `head_dim` 2,
// chosen so that its forward pass can be followed by hand: unit norm scales,
// identity rotation tables for `num_query_tokens` positions, a zero query
// projection (every visible token is then attended equally), identity key,
// value and MLP down projections, output projection diag(8, -4), MLP gate
// projection diag(0.8, -8) and MLP up projection diag(0.8, -4). The caller
// supplies the input tokens.
SeededLayerData CreateTinyBlockData(size_t num_query_tokens) {
  SeededLayerData data;
  data.rms_scale = {1.0f, 1.0f};
  data.ffn_rms_scale = {1.0f, 1.0f};
  data.rope_cos.assign(num_query_tokens * 2, 1.0f);
  data.rope_sin.assign(num_query_tokens * 2, 0.0f);
  data.w_q = {0.0f, 0.0f, 0.0f, 0.0f};
  data.w_k = {1.0f, 0.0f, 0.0f, 1.0f};
  data.w_v = {1.0f, 0.0f, 0.0f, 1.0f};
  data.w_o = {8.0f, 0.0f, 0.0f, -4.0f};
  data.w_gate = {0.8f, 0.0f, 0.0f, -8.0f};
  data.w_up = {0.8f, 0.0f, 0.0f, -4.0f};
  data.w_down = {1.0f, 0.0f, 0.0f, 1.0f};
  return data;
}

// Sets up `backend` for `spec` over `data`, runs one forward pass and returns
// the output tensor, or the first error.
absl::StatusOr<std::vector<float>> RunLayer(
    LayerBackend& backend, const LayerSpec& spec, const SeededLayerData& data,
    const BackendOptions& options = BackendOptions{}) {
  absl::Status setup_status = backend.Setup(spec, data, options);
  if (!setup_status.ok()) {
    return setup_status;
  }
  absl::Status execute_status = backend.Execute();
  if (!execute_status.ok()) {
    return execute_status;
  }
  return backend.GetOutputData();
}

// Matches a tensor of `size` finite elements that are not all zero.
Matcher<const std::vector<float>&> IsFiniteNonZeroTensorOfSize(size_t size) {
  return AllOf(SizeIs(size), Each(IsFinite()),
               Contains(Not(FloatNear(0.0f, kZeroTolerance))));
}

// Compiles the graph that produces `outputs` for `runtime`.
absl::StatusOr<std::unique_ptr<NnpackRunner>> CreateRunner(
    RuntimeBackend runtime, std::vector<TensorHandle> outputs) {
  if (runtime == RuntimeBackend::kYnnpack) {
    absl::StatusOr<YnnpackRunner> runner =
        YnnpackRunner::Create(std::move(outputs));
    if (!runner.ok()) {
      return runner.status();
    }
    return std::make_unique<YnnpackRunner>(*std::move(runner));
  }
  absl::StatusOr<XnnpackRunner> runner =
      XnnpackRunner::Create(std::move(outputs));
  if (!runner.ok()) {
    return runner.status();
  }
  return std::make_unique<XnnpackRunner>(*std::move(runner));
}

// Runs the graph that produces `output` on `runtime` with `input_values` bound
// to `input`, and returns the elements of `output`.
absl::StatusOr<std::vector<float>> Evaluate(
    RuntimeBackend runtime, const BackendTensor& output,
    const BackendTensor& input, const std::vector<float>& input_values) {
  absl::StatusOr<std::unique_ptr<NnpackRunner>> runner =
      CreateRunner(runtime, {output});
  if (!runner.ok()) {
    return runner.status();
  }
  absl::Status input_status = (*runner)->SetInput(input, input_values);
  if (!input_status.ok()) {
    return input_status;
  }
  absl::Status run_status = (*runner)->Run();
  if (!run_status.ok()) {
    return run_status;
  }
  absl::StatusOr<LockedBufferSpan<const float>> values =
      (*runner)->ReadOutputAs<float>(output);
  if (!values.ok()) {
    return values.status();
  }
  return std::vector<float>(values->begin(), values->end());
}

// Returns the elements of `constant`, a graph constant of `shape`, as
// materialized by `runtime`. Runners only expose computed outputs, so the
// constant is read through an addition with a zero-filled input.
absl::StatusOr<std::vector<float>> ReadConstant(RuntimeBackend runtime,
                                                const BackendTensor& constant,
                                                const std::vector<int>& shape) {
  BackendTensor zeros({.type = Type::kFP32, .shape = shape});
  BackendTensor sum = Add(constant, zeros);
  size_t element_count = 1;
  for (int dim : shape) {
    element_count *= static_cast<size_t>(dim);
  }
  return Evaluate(runtime, sum, zeros, std::vector<float>(element_count, 0.0f));
}

// Returns the name of the `data` buffer that `buffer` points at.
std::string BufferName(const std::vector<float>* buffer,
                       const SeededLayerData& data) {
  if (buffer == &data.input) return "input";
  if (buffer == &data.q) return "q";
  if (buffer == &data.k) return "k";
  if (buffer == &data.v) return "v";
  return "unknown";
}

// Describes each external input of `graph` as the name of the `data` buffer it
// is bound to and the bound element count (0 for the whole buffer).
std::vector<std::pair<std::string, size_t>> DescribeExternalInputs(
    const LitertLayerGraph& graph, const SeededLayerData& data) {
  std::vector<std::pair<std::string, size_t>> described;
  for (const ExternalInputBinding& binding : graph.external_inputs) {
    described.emplace_back(BufferName(binding.host_buffer, data),
                           binding.num_elements);
  }
  return described;
}

// A named factory of one LayerBackend implementation.
struct BackendFactoryCase {
  std::string name;
  std::unique_ptr<LayerBackend> (*create)();
};

// Behavior every LayerBackend implementation shares. Running it against
// FakeLayerBackend keeps the fake faithful to the real backends.
class LayerBackendContractTest
    : public testing::TestWithParam<BackendFactoryCase> {};

TEST_P(LayerBackendContractTest, NameIsNotEmpty) {
  std::unique_ptr<LayerBackend> backend = GetParam().create();

  EXPECT_THAT(backend->Name(), Not(IsEmpty()));
}

TEST_P(LayerBackendContractTest, ExecuteBeforeSetupIsFailedPrecondition) {
  std::unique_ptr<LayerBackend> backend = GetParam().create();

  EXPECT_EQ(backend->Execute().code(), absl::StatusCode::kFailedPrecondition);
}

TEST_P(LayerBackendContractTest, GetOutputDataBeforeSetupIsFailedPrecondition) {
  std::unique_ptr<LayerBackend> backend = GetParam().create();

  EXPECT_EQ(backend->GetOutputData().status().code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST_P(LayerBackendContractTest, SetupRejectsInvalidSpecWithInvalidArgument) {
  std::unique_ptr<LayerBackend> backend = GetParam().create();
  LayerSpec spec = CreateTestSpec(LayerType::kRmsNorm);
  spec.num_heads = 0;

  absl::Status status =
      backend->Setup(spec, SeededLayerData(), BackendOptions{});

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
}

TEST_P(LayerBackendContractTest, FailedSetupLeavesBackendWithoutLayer) {
  std::unique_ptr<LayerBackend> backend = GetParam().create();
  LayerSpec spec = CreateTestSpec(LayerType::kRmsNorm);
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);
  ASSERT_OK(backend->Setup(spec, data, BackendOptions{}));
  LayerSpec invalid_spec = spec;
  invalid_spec.head_dim = 15;  // Odd head dimensions cannot be rotated.

  absl::Status setup_status =
      backend->Setup(invalid_spec, data, BackendOptions{});

  EXPECT_EQ(setup_status.code(), absl::StatusCode::kInvalidArgument)
      << setup_status;
  EXPECT_EQ(backend->Execute().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(backend->GetOutputData().status().code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST_P(LayerBackendContractTest, OutputHasOneFiniteValuePerOutputElement) {
  std::unique_ptr<LayerBackend> backend = GetParam().create();
  LayerSpec spec = CreateTestSpec(LayerType::kRmsNorm);

  absl::StatusOr<std::vector<float>> output =
      RunLayer(*backend, spec, CreateSeededLayerData(spec, kSeed));

  // B*T*M = 1*8*64.
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, AllOf(SizeIs(512), Each(IsFinite())));
}

INSTANTIATE_TEST_SUITE_P(
    Implementations, LayerBackendContractTest,
    Values(BackendFactoryCase{"LitertTensorXnnpack",
                              []() -> std::unique_ptr<LayerBackend> {
                                return std::make_unique<LitertTensorBackend>(
                                    RuntimeBackend::kXnnpack);
                              }},
           BackendFactoryCase{"LitertTensorYnnpack",
                              []() -> std::unique_ptr<LayerBackend> {
                                return std::make_unique<LitertTensorBackend>(
                                    RuntimeBackend::kYnnpack);
                              }},
           BackendFactoryCase{"Fake",
                              []() -> std::unique_ptr<LayerBackend> {
                                return std::make_unique<FakeLayerBackend>();
                              }}),
    [](const testing::TestParamInfo<BackendFactoryCase>& info) {
      return info.param.name;
    });

TEST(LitertTensorBackendTest, NameIdentifiesBackend) {
  EXPECT_EQ(LitertTensorBackend().Name(), "LitertTensorBackend");
}

TEST(LitertTensorBackendTest, IsNeitherCopyableNorMovable) {
  EXPECT_FALSE(std::is_copy_constructible_v<LitertTensorBackend>);
  EXPECT_FALSE(std::is_copy_assignable_v<LitertTensorBackend>);
  EXPECT_FALSE(std::is_move_constructible_v<LitertTensorBackend>);
  EXPECT_FALSE(std::is_move_assignable_v<LitertTensorBackend>);
}

TEST(LitertTensorBackendTest, MultithreadedXnnpackRuntimeExposesThreadpool) {
  LayerSpec spec = CreateTestSpec(LayerType::kTransformerBlock);
  LitertTensorBackend backend(RuntimeBackend::kXnnpack);

  absl::StatusOr<std::vector<float>> output =
      RunLayer(backend, spec, CreateSeededLayerData(spec, kSeed),
               BackendOptions{.num_threads = 2});

  ASSERT_OK(output.status());
  EXPECT_THAT(backend.GetThreadpool(), NotNull());
  EXPECT_THAT(*output, IsFiniteNonZeroTensorOfSize(512));
}

// The YNNPACK tests below request two threads, which makes an XNNPACK runtime
// expose its threadpool. A null threadpool therefore shows that YNNPACK, which
// schedules work internally, ran the layer.

TEST(LitertTensorBackendTest, YnnpackRuntimeHidesThreadpool) {
  LayerSpec spec = CreateTestSpec(LayerType::kRmsNorm);
  LitertTensorBackend backend(RuntimeBackend::kYnnpack);

  absl::StatusOr<std::vector<float>> output =
      RunLayer(backend, spec, CreateSeededLayerData(spec, kSeed),
               BackendOptions{.num_threads = 2});

  ASSERT_OK(output.status());
  EXPECT_EQ(backend.GetThreadpool(), nullptr);
  EXPECT_THAT(*output, IsFiniteNonZeroTensorOfSize(512));
}

TEST(LitertTensorBackendTest, UseYnnpackOptionOverridesXnnpackRuntime) {
  LayerSpec spec = CreateTestSpec(LayerType::kRmsNorm);
  LitertTensorBackend backend(RuntimeBackend::kXnnpack);

  absl::StatusOr<std::vector<float>> output =
      RunLayer(backend, spec, CreateSeededLayerData(spec, kSeed),
               BackendOptions{.num_threads = 2, .use_ynnpack = true});

  ASSERT_OK(output.status());
  EXPECT_EQ(backend.GetThreadpool(), nullptr);
  EXPECT_THAT(*output, IsFiniteNonZeroTensorOfSize(512));
}

TEST(LitertTensorBackendTest, SecondSetupReplacesPreviousLayer) {
  LitertTensorBackend backend;
  LayerSpec mlp_spec = CreateTestSpec(LayerType::kGatedMlp);
  ASSERT_OK(backend.Setup(mlp_spec, CreateSeededLayerData(mlp_spec, kSeed),
                          BackendOptions{}));
  ASSERT_OK(backend.Execute());
  LayerSpec norm_spec = CreateTestSpec(LayerType::kRmsNorm);
  norm_spec.seq_len_q = 4;
  norm_spec.seq_len_kv = 4;

  absl::StatusOr<std::vector<float>> output =
      RunLayer(backend, norm_spec, CreateSeededLayerData(norm_spec, kSeed));

  // T*M = 4*64 elements of the second layer, not 8*64 of the first.
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, IsFiniteNonZeroTensorOfSize(256));
}

TEST(LitertTensorBackendTest, UndersizedDataSetupLeavesBackendWithoutLayer) {
  LitertTensorBackend backend;
  LayerSpec spec = CreateTestSpec(LayerType::kRmsNorm);
  ASSERT_OK(backend.Setup(spec, CreateSeededLayerData(spec, kSeed),
                          BackendOptions{}));

  // Empty data cannot back the scale weights of the layer.
  absl::Status setup_status =
      backend.Setup(spec, SeededLayerData(), BackendOptions{});

  EXPECT_EQ(setup_status.code(), absl::StatusCode::kInvalidArgument)
      << setup_status;
  EXPECT_EQ(backend.Execute().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(backend.GetOutputData().status().code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(LitertTensorBackendTest, SetupRejectsNonFp32SpecAsUnimplemented) {
  LitertTensorBackend backend;
  LayerSpec spec = CreateTestSpec(LayerType::kGatedMlp);
  spec.dtype = DataType::kQD8;

  absl::Status status =
      backend.Setup(spec, CreateSeededLayerData(spec, kSeed), BackendOptions{});

  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented) << status;
  EXPECT_EQ(backend.Execute().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(LitertTensorBackendTest, SetupRejectsInputBufferOfWrongSize) {
  LitertTensorBackend backend;
  LayerSpec spec = CreateTestSpec(LayerType::kRmsNorm);
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);
  data.input.pop_back();

  absl::Status status = backend.Setup(spec, data, BackendOptions{});

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
}

TEST(LitertTensorBackendTest, SetupRejectsKeyBufferShorterThanContext) {
  LitertTensorBackend backend;
  LayerSpec spec = CreateTestSpec(LayerType::kStandardAttention);
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);
  // The keys of the S=8 context tokens take N_kv*S*D = 2*8*16 elements.
  data.k.pop_back();

  absl::Status status = backend.Setup(spec, data, BackendOptions{});

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
}

struct ExecutionCase {
  std::string name;
  LayerType layer_type;
  AttentionMode attention_mode;
  MlpActivation mlp_activation;
  size_t expected_output_size;
};

std::string ExecutionCaseName(
    const testing::TestParamInfo<ExecutionCase>& info) {
  return info.param.name;
}

class LayerExecutionTest : public testing::TestWithParam<ExecutionCase> {};

TEST_P(LayerExecutionTest, ProducesFiniteNonZeroOutput) {
  const ExecutionCase& tc = GetParam();
  LayerSpec spec = CreateTestSpec(tc.layer_type, tc.attention_mode);
  spec.mlp_activation = tc.mlp_activation;
  LitertTensorBackend backend;

  absl::StatusOr<std::vector<float>> output =
      RunLayer(backend, spec, CreateSeededLayerData(spec, kSeed));

  ASSERT_OK(output.status());
  EXPECT_THAT(*output, IsFiniteNonZeroTensorOfSize(tc.expected_output_size));
}

// Output sizes are B*T*H*D with B=1, H*D=64, and T=8 in prefill or T=1 in
// decode.
INSTANTIATE_TEST_SUITE_P(
    AllLayers, LayerExecutionTest,
    Values(ExecutionCase{.name = "RmsNorm",
                         .layer_type = LayerType::kRmsNorm,
                         .attention_mode = AttentionMode::kPrefill,
                         .mlp_activation = MlpActivation::kSwiGLU,
                         .expected_output_size = 512},
           ExecutionCase{.name = "RoPE",
                         .layer_type = LayerType::kRoPE,
                         .attention_mode = AttentionMode::kPrefill,
                         .mlp_activation = MlpActivation::kSwiGLU,
                         .expected_output_size = 512},
           ExecutionCase{.name = "GatedMlpSwiGLU",
                         .layer_type = LayerType::kGatedMlp,
                         .attention_mode = AttentionMode::kPrefill,
                         .mlp_activation = MlpActivation::kSwiGLU,
                         .expected_output_size = 512},
           ExecutionCase{.name = "GatedMlpGeGLU",
                         .layer_type = LayerType::kGatedMlp,
                         .attention_mode = AttentionMode::kPrefill,
                         .mlp_activation = MlpActivation::kGeGLU,
                         .expected_output_size = 512},
           ExecutionCase{.name = "StandardAttentionPrefill",
                         .layer_type = LayerType::kStandardAttention,
                         .attention_mode = AttentionMode::kPrefill,
                         .mlp_activation = MlpActivation::kSwiGLU,
                         .expected_output_size = 512},
           ExecutionCase{.name = "StandardAttentionDecode",
                         .layer_type = LayerType::kStandardAttention,
                         .attention_mode = AttentionMode::kDecode,
                         .mlp_activation = MlpActivation::kSwiGLU,
                         .expected_output_size = 64},
           ExecutionCase{.name = "DecodeOptimizedAttention",
                         .layer_type = LayerType::kDecodeOptimizedAttention,
                         .attention_mode = AttentionMode::kDecode,
                         .mlp_activation = MlpActivation::kSwiGLU,
                         .expected_output_size = 64},
           ExecutionCase{.name = "TransformerBlockPrefill",
                         .layer_type = LayerType::kTransformerBlock,
                         .attention_mode = AttentionMode::kPrefill,
                         .mlp_activation = MlpActivation::kSwiGLU,
                         .expected_output_size = 512},
           ExecutionCase{.name = "TransformerBlockDecode",
                         .layer_type = LayerType::kTransformerBlock,
                         .attention_mode = AttentionMode::kDecode,
                         .mlp_activation = MlpActivation::kSwiGLU,
                         .expected_output_size = 64}),
    ExecutionCaseName);

class DecoupledEmbedDimExecutionTest
    : public testing::TestWithParam<ExecutionCase> {};

TEST_P(DecoupledEmbedDimExecutionTest, ProducesFiniteNonZeroOutput) {
  const ExecutionCase& tc = GetParam();
  LayerSpec spec =
      CreateDecoupledEmbedDimSpec(tc.layer_type, tc.attention_mode);
  spec.mlp_activation = tc.mlp_activation;
  LitertTensorBackend backend;

  absl::StatusOr<std::vector<float>> output =
      RunLayer(backend, spec, CreateSeededLayerData(spec, kSeed));

  ASSERT_OK(output.status());
  EXPECT_THAT(*output, IsFiniteNonZeroTensorOfSize(tc.expected_output_size));
}

// With M=48 and H*D=64, RoPE and standalone attention emit T*64 elements
// while the other layers emit T*48; T=8 in prefill and T=1 in decode.
INSTANTIATE_TEST_SUITE_P(
    AllLayers, DecoupledEmbedDimExecutionTest,
    Values(ExecutionCase{.name = "RmsNorm",
                         .layer_type = LayerType::kRmsNorm,
                         .attention_mode = AttentionMode::kPrefill,
                         .mlp_activation = MlpActivation::kGeGLU,
                         .expected_output_size = 384},
           ExecutionCase{.name = "RoPE",
                         .layer_type = LayerType::kRoPE,
                         .attention_mode = AttentionMode::kPrefill,
                         .mlp_activation = MlpActivation::kGeGLU,
                         .expected_output_size = 512},
           ExecutionCase{.name = "GatedMlp",
                         .layer_type = LayerType::kGatedMlp,
                         .attention_mode = AttentionMode::kPrefill,
                         .mlp_activation = MlpActivation::kGeGLU,
                         .expected_output_size = 384},
           ExecutionCase{.name = "StandardAttentionPrefill",
                         .layer_type = LayerType::kStandardAttention,
                         .attention_mode = AttentionMode::kPrefill,
                         .mlp_activation = MlpActivation::kGeGLU,
                         .expected_output_size = 512},
           ExecutionCase{.name = "StandardAttentionDecode",
                         .layer_type = LayerType::kStandardAttention,
                         .attention_mode = AttentionMode::kDecode,
                         .mlp_activation = MlpActivation::kGeGLU,
                         .expected_output_size = 64},
           ExecutionCase{.name = "DecodeOptimizedAttention",
                         .layer_type = LayerType::kDecodeOptimizedAttention,
                         .attention_mode = AttentionMode::kDecode,
                         .mlp_activation = MlpActivation::kGeGLU,
                         .expected_output_size = 64},
           ExecutionCase{.name = "TransformerBlockPrefill",
                         .layer_type = LayerType::kTransformerBlock,
                         .attention_mode = AttentionMode::kPrefill,
                         .mlp_activation = MlpActivation::kGeGLU,
                         .expected_output_size = 384},
           ExecutionCase{.name = "TransformerBlockDecode",
                         .layer_type = LayerType::kTransformerBlock,
                         .attention_mode = AttentionMode::kDecode,
                         .mlp_activation = MlpActivation::kGeGLU,
                         .expected_output_size = 48}),
    ExecutionCaseName);

// Layers over hand-written inputs with hand-computed outputs, on each runtime.
class AnalyticalLayerTest : public testing::TestWithParam<RuntimeBackend> {};

TEST_P(AnalyticalLayerTest, RmsNormNormalizesAndScalesEachToken) {
  LayerSpec spec = CreateTinySpec(LayerType::kRmsNorm, AttentionMode::kPrefill,
                                  /*head_dim=*/4, /*seq_len=*/2);
  spec.eps = 3.5f;
  SeededLayerData data;
  data.input = {3.0f, 4.0f, 3.0f, 4.0f, 4.0f, 7.0f, 4.0f, 7.0f};
  data.rms_scale = {2.0f, 0.5f, 2.0f, 0.5f};
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  // Token [3, 4, 3, 4]: mean square 12.5 + eps 3.5 = 16, so it is divided by
  // 4. Token [4, 7, 4, 7]: mean square 32.5 + eps 3.5 = 36, so it is divided
  // by 6. Each result is then multiplied by the scale [2, 0.5, 2, 0.5].
  const std::vector<float> expected = {
      1.5f, 0.5f, 1.5f, 0.5f, 1.3333334f, 0.5833333f, 1.3333334f, 0.5833333f};
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(AnalyticalLayerTest, RoPERotatesHalfSplitPairs) {
  LayerSpec spec = CreateTinySpec(LayerType::kRoPE, AttentionMode::kPrefill,
                                  /*head_dim=*/4, /*seq_len=*/1);
  SeededLayerData data;
  data.input = {1.0f, 2.0f, 3.0f, 4.0f};
  data.rope_cos = {0.6f, 0.0f, 0.6f, 0.0f};
  data.rope_sin = {0.8f, 1.0f, 0.8f, 1.0f};
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  // x * cos + [-x2, -x3, x0, x1] * sin:
  // [0.6 - 2.4, 0 - 4, 1.8 + 0.8, 0 + 2].
  const std::vector<float> expected = {-1.8f, -4.0f, 2.6f, 2.0f};
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(AnalyticalLayerTest, SwiGluMlpGatesUpProjectionAndProjectsDown) {
  LayerSpec spec = CreateTinySpec(LayerType::kGatedMlp, AttentionMode::kPrefill,
                                  /*head_dim=*/2, /*seq_len=*/1);
  spec.mlp_activation = MlpActivation::kSwiGLU;
  SeededLayerData data;
  data.input = {1.0f, 2.0f};
  data.w_gate = {1.0f, 0.0f, 0.0f, 1.0f};
  data.w_up = {0.5f, 0.0f, 0.0f, 0.5f};
  data.w_down = {1.0f, 1.0f, 0.0f, 2.0f};
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  // gate = [1, 2] and up = [0.5, 1], so the hidden activation is
  // [swish(1) * 0.5, swish(2) * 1] = [0.3655293, 1.7615942]. The down
  // projection rows [1, 1] and [0, 2] give [0.3655293 + 1.7615942,
  // 2 * 1.7615942].
  const std::vector<float> expected = {2.1271234f, 3.5231884f};
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(AnalyticalLayerTest, GeGluMlpGatesWithTanhApproximatedGelu) {
  LayerSpec spec = CreateTinySpec(LayerType::kGatedMlp, AttentionMode::kPrefill,
                                  /*head_dim=*/2, /*seq_len=*/1);
  spec.mlp_activation = MlpActivation::kGeGLU;
  SeededLayerData data;
  data.input = {1.0f, 2.0f};
  data.w_gate = {1.0f, 0.0f, 0.0f, 1.0f};
  data.w_up = {0.5f, 0.0f, 0.0f, 0.5f};
  data.w_down = {1.0f, 1.0f, 0.0f, 2.0f};
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  // gate = [1, 2] and up = [0.5, 1]. Under the tanh approximation gelu(1) =
  // 0.841192 and gelu(2) = 1.9545977, so the hidden activation is
  // [0.420596, 1.9545977] and the down projection rows [1, 1] and [0, 2] give
  // [0.420596 + 1.9545977, 2 * 1.9545977]. The exact GELU would give
  // [2.3751721, 3.9089994] instead.
  const std::vector<float> expected = {2.3751936f, 3.9091954f};
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(AnalyticalLayerTest, PrefillAttentionWeighsValuesPerQueryToken) {
  LayerSpec spec =
      CreateTinySpec(LayerType::kStandardAttention, AttentionMode::kPrefill,
                     /*head_dim=*/4, /*seq_len=*/2);
  SeededLayerData data;
  data.q = {2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  data.k = {0.0f, 0.0f, 0.0f, 0.0f, kLn3, 0.0f, 0.0f, 0.0f};
  data.v = {4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f};
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  // Scores are q.k / sqrt(D) = q.k / 2. The first query scores [0, ln 3], so
  // it weighs the values 1/4 and 3/4; the zero query weighs them equally.
  const std::vector<float> expected = {1.0f, 3.0f, 0.0f, 0.0f,
                                       2.0f, 2.0f, 0.0f, 0.0f};
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(AnalyticalLayerTest, PrefillBlockAttendsCausallyAndAddsBothBranches) {
  LayerSpec spec =
      CreateTinySpec(LayerType::kTransformerBlock, AttentionMode::kPrefill,
                     /*head_dim=*/2, /*seq_len=*/2);
  spec.eps = 3.0f;
  SeededLayerData data = CreateTinyBlockData(/*num_query_tokens=*/2);
  data.input = {1.0f, 1.0f, 1.0f, -1.0f};
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  // Mean square 1 + eps 3 = 4 halves each token, so the values are
  // [0.5, 0.5] and [0.5, -0.5]. Token 0 sees only its own value, which the
  // output projection maps to [4, -2]; token 1 averages both values to
  // [0.5, 0], mapped to [4, 0]. Either way the residual stream becomes
  // [5, -1], whose mean square 13 + eps 3 = 16 normalizes it to
  // [1.25, -0.25]. With gate = [1, 2] and up = [1, 1] the MLP adds
  // [swish(1), swish(2)] = [0.7310586, 1.7615942].
  const std::vector<float> expected = {5.7310586f, 0.7615942f,  //
                                       5.7310586f, 0.7615942f};
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(AnalyticalLayerTest, PrefillBlockLimitsAttentionToSlidingWindow) {
  LayerSpec spec =
      CreateTinySpec(LayerType::kTransformerBlock, AttentionMode::kPrefill,
                     /*head_dim=*/2, /*seq_len=*/2);
  spec.eps = 3.0f;
  spec.sliding_window = 1;
  SeededLayerData data = CreateTinyBlockData(/*num_query_tokens=*/2);
  data.input = {1.0f, 1.0f, 1.0f, -1.0f};
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  // Mean square 1 + eps 3 = 4 halves each token, so the values are
  // [0.5, 0.5] and [0.5, -0.5], and a window of one token leaves each token
  // with only its own value. The output projection maps them to [4, -2] and
  // [4, 2], so the residual streams are [5, -1] and [5, 1], normalized by
  // mean square 13 + eps 3 = 16 to [1.25, -0.25] and [1.25, 0.25]. The first
  // gives gate = [1, 2] and up = [1, 1], so the MLP adds [swish(1), swish(2)]
  // = [0.7310586, 1.7615942]; the second gives gate = [1, -2] and
  // up = [1, -1], so the MLP adds [swish(1), -swish(-2)] =
  // [0.7310586, 0.2384058].
  const std::vector<float> expected = {5.7310586f, 0.7615942f,  //
                                       5.7310586f, 1.2384058f};
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(AnalyticalLayerTest, DecodeBlockAttendsOverCachedAndCurrentToken) {
  LayerSpec spec =
      CreateTinySpec(LayerType::kTransformerBlock, AttentionMode::kDecode,
                     /*head_dim=*/2, /*seq_len=*/2);
  spec.eps = 3.0f;
  SeededLayerData data = CreateTinyBlockData(/*num_query_tokens=*/1);
  data.input = {1.0f, 1.0f};
  // Key and value of the one cached token. The zero query projection scores
  // every key equally, whatever its content.
  data.k = {3.0f, 7.0f};
  data.v = {0.5f, -0.5f};
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  // Mean square 1 + eps 3 = 4 halves the token to the value [0.5, 0.5],
  // which is averaged with the cached value to [0.5, 0]. The output
  // projection maps that to [4, 0], so the residual stream is [5, 1],
  // normalized by mean square 13 + eps 3 = 16 to [1.25, 0.25]. With
  // gate = [1, -2] and up = [1, -1] the MLP adds [swish(1), -swish(-2)] =
  // [0.7310586, 0.2384058].
  const std::vector<float> expected = {5.7310586f, 1.2384058f};
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(AnalyticalLayerTest, BlockQkNormRescalesQueryAndKeyBeforeScoring) {
  LayerSpec spec =
      CreateTinySpec(LayerType::kTransformerBlock, AttentionMode::kDecode,
                     /*head_dim=*/2, /*seq_len=*/2);
  spec.eps = 3.0f;
  spec.qk_norm = true;
  SeededLayerData data = CreateTinyBlockData(/*num_query_tokens=*/1);
  data.input = {1.0f, 1.0f};
  data.w_q = {2.0f, 0.0f, 0.0f, 2.0f};
  data.w_k = {2.0f, 0.0f, 0.0f, 2.0f};
  data.q_norm_scale = {2.0f, 0.0f};
  data.k_norm_scale = {6.0f, 2.0f};
  // Key and value of the one cached token.
  data.k = {3.0f, 9.0f};
  data.v = {0.5f, -0.5f};
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  // The token is halved to [0.5, 0.5] and projected to the query and key
  // [1, 1]. Mean square 1 + eps 3 = 4 halves both again, and the scales turn
  // them into the query [1, 0] and the key [3, 1]. That query scores the key
  // and the cached key [3, 9] equally, so the value [0.5, 0.5] is averaged
  // with the cached value to [0.5, 0]; the unnormalized query [1, 1] would
  // favor the cached key instead. The output projection maps [0.5, 0] to
  // [4, 0], so the residual stream is [5, 1], normalized by mean square
  // 13 + eps 3 = 16 to [1.25, 0.25]. With gate = [1, -2] and up = [1, -1] the
  // MLP adds [swish(1), -swish(-2)] = [0.7310586, 0.2384058].
  const std::vector<float> expected = {5.7310586f, 1.2384058f};
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(AnalyticalLayerTest, BlockPostNormsRescaleBothBranchOutputs) {
  LayerSpec spec =
      CreateTinySpec(LayerType::kTransformerBlock, AttentionMode::kPrefill,
                     /*head_dim=*/2, /*seq_len=*/1);
  spec.eps = 3.0f;
  spec.post_norms = true;
  SeededLayerData data = CreateTinyBlockData(/*num_query_tokens=*/1);
  data.input = {1.0f, 1.0f};
  data.w_o = {10.0f, 0.0f, 0.0f, 2.0f};
  data.post_attn_rms_scale = {3.2f, -8.0f};
  data.post_ffn_rms_scale = {2.0f, -1.0f};
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  // Mean square 1 + eps 3 = 4 halves the token to the value [0.5, 0.5], the
  // only one it attends to, and the output projection maps it to [5, 1]. The
  // post-attention norm divides that by sqrt(13 + 3) = 4 and scales it to
  // [1.25 * 3.2, 0.25 * -8] = [4, -2], so the residual stream is [5, -1],
  // normalized to [1.25, -0.25]. With gate = [1, 2] and up = [1, 1] the MLP
  // returns [swish(1), swish(2)] = [0.7310586, 1.7615942]. The post-MLP norm
  // divides that by sqrt(1.8188304 + 3) = 2.1951835 and scales it to
  // [0.3330285 * 2, 0.8024815 * -1] = [0.6660569, -0.8024815].
  const std::vector<float> expected = {5.6660569f, -1.8024815f};
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(AnalyticalLayerTest, PrefillBlockWithZeroOutputProjectionsIsIdentity) {
  LayerSpec spec = CreateTestSpec(LayerType::kTransformerBlock);
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);
  // With no contribution from the attention and MLP branches, only the
  // residual connections reach the output.
  data.w_o.assign(data.w_o.size(), 0.0f);
  data.w_down.assign(data.w_down.size(), 0.0f);
  const std::vector<float> expected = data.input;
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatEq(), expected));
}

TEST_P(AnalyticalLayerTest, DecodeBlockWithZeroOutputProjectionsIsIdentity) {
  LayerSpec spec =
      CreateTestSpec(LayerType::kTransformerBlock, AttentionMode::kDecode);
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);
  data.w_o.assign(data.w_o.size(), 0.0f);
  data.w_down.assign(data.w_down.size(), 0.0f);
  const std::vector<float> expected = data.input;
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatEq(), expected));
}

TEST_P(AnalyticalLayerTest,
       SingleTokenContextDecodeBlockWithZeroOutputProjectionsIsIdentity) {
  LayerSpec spec =
      CreateTestSpec(LayerType::kTransformerBlock, AttentionMode::kDecode);
  // A context of one token leaves no earlier tokens to cache.
  spec.seq_len_kv = 1;
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);
  data.w_o.assign(data.w_o.size(), 0.0f);
  data.w_down.assign(data.w_down.size(), 0.0f);
  const std::vector<float> expected = data.input;
  LitertTensorBackend backend(GetParam());

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatEq(), expected));
}

INSTANTIATE_TEST_SUITE_P(Runtimes, AnalyticalLayerTest,
                         Values(RuntimeBackend::kXnnpack,
                                RuntimeBackend::kYnnpack),
                         RuntimeParamName);

// Decode attention over hand-written inputs, for both attention layer types on
// each runtime.
class AnalyticalDecodeAttentionTest
    : public testing::TestWithParam<std::tuple<RuntimeBackend, LayerType>> {};

TEST_P(AnalyticalDecodeAttentionTest, WeighsValuesBySoftmaxOfScaledScores) {
  LayerSpec spec =
      CreateTinySpec(std::get<1>(GetParam()), AttentionMode::kDecode,
                     /*head_dim=*/4, /*seq_len=*/2);
  SeededLayerData data;
  data.q = {2.0f, 0.0f, 0.0f, 0.0f};
  data.k = {0.0f, 0.0f, 0.0f, 0.0f, kLn3, 0.0f, 0.0f, 0.0f};
  data.v = {4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f};
  LitertTensorBackend backend(std::get<0>(GetParam()));

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  // Scores are q.k / sqrt(D) = [0, ln 3], so the two values are weighed 1/4
  // and 3/4.
  const std::vector<float> expected = {1.0f, 3.0f, 0.0f, 0.0f};
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(AnalyticalDecodeAttentionTest, QueryHeadsShareTheirGroupsKvHead) {
  LayerSpec spec =
      CreateTinySpec(std::get<1>(GetParam()), AttentionMode::kDecode,
                     /*head_dim=*/2, /*seq_len=*/2);
  spec.num_heads = 4;
  spec.num_kv_heads = 2;
  SeededLayerData data;
  // Zero queries score every key equally, so each head averages the two
  // positions of its KV head.
  data.q = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  data.k = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  data.v = {1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f};
  LitertTensorBackend backend(std::get<0>(GetParam()));

  absl::StatusOr<std::vector<float>> output = RunLayer(backend, spec, data);

  // Heads 0 and 1 read KV head 0: mean of [1, 2] and [3, 4]. Heads 2 and 3
  // read KV head 1: mean of [10, 20] and [30, 40].
  const std::vector<float> expected = {2.0f,  3.0f,  2.0f,  3.0f,
                                       20.0f, 30.0f, 20.0f, 30.0f};
  ASSERT_OK(output.status());
  EXPECT_THAT(*output, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

INSTANTIATE_TEST_SUITE_P(
    RuntimesAndLayers, AnalyticalDecodeAttentionTest,
    Combine(Values(RuntimeBackend::kXnnpack, RuntimeBackend::kYnnpack),
            Values(LayerType::kStandardAttention,
                   LayerType::kDecodeOptimizedAttention)),
    [](const testing::TestParamInfo<std::tuple<RuntimeBackend, LayerType>>&
           info) {
      return RuntimeName(std::get<0>(info.param)) +
             LayerTypeToString(std::get<1>(info.param));
    });

// The parity tests below extend the hand-computed checks above to seeded
// random inputs and full-size test layers, for which no closed-form
// expectation exists: one implementation serves as the reference for the
// other.

TEST(LitertTensorBackendTest,
     DecodeOptimizedAttentionMatchesStandardAttention) {
  LayerSpec standard_spec =
      CreateTestSpec(LayerType::kStandardAttention, AttentionMode::kDecode);
  LayerSpec optimized_spec = CreateTestSpec(
      LayerType::kDecodeOptimizedAttention, AttentionMode::kDecode);
  SeededLayerData data = CreateSeededLayerData(standard_spec, kSeed);
  LitertTensorBackend standard_backend;
  LitertTensorBackend optimized_backend;

  absl::StatusOr<std::vector<float>> standard_output =
      RunLayer(standard_backend, standard_spec, data);
  absl::StatusOr<std::vector<float>> optimized_output =
      RunLayer(optimized_backend, optimized_spec, data);

  ASSERT_OK(standard_output.status());
  ASSERT_OK(optimized_output.status());
  EXPECT_THAT(*optimized_output,
              Pointwise(FloatNear(kParityTolerance), *standard_output));
}

struct ParityCase {
  std::string name;
  LayerType layer_type;
  AttentionMode attention_mode;
  size_t num_threads;
};

class CrossRuntimeParityTest : public testing::TestWithParam<ParityCase> {};

TEST_P(CrossRuntimeParityTest, YnnpackMatchesXnnpack) {
  const ParityCase& tc = GetParam();
  LayerSpec spec = CreateTestSpec(tc.layer_type, tc.attention_mode);
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);
  BackendOptions options;
  options.num_threads = tc.num_threads;
  LitertTensorBackend xnnpack_backend(RuntimeBackend::kXnnpack);
  LitertTensorBackend ynnpack_backend(RuntimeBackend::kYnnpack);

  absl::StatusOr<std::vector<float>> xnnpack_output =
      RunLayer(xnnpack_backend, spec, data, options);
  absl::StatusOr<std::vector<float>> ynnpack_output =
      RunLayer(ynnpack_backend, spec, data, options);

  ASSERT_OK(xnnpack_output.status());
  ASSERT_OK(ynnpack_output.status());
  EXPECT_THAT(*ynnpack_output,
              Pointwise(FloatNear(kParityTolerance), *xnnpack_output));
}

// Prefill cases run on two threads and decode cases on one, so that both
// threading paths of each runtime are compared.
INSTANTIATE_TEST_SUITE_P(
    AllLayers, CrossRuntimeParityTest,
    Values(ParityCase{.name = "RmsNormPrefill",
                      .layer_type = LayerType::kRmsNorm,
                      .attention_mode = AttentionMode::kPrefill,
                      .num_threads = 2},
           ParityCase{.name = "RoPEPrefill",
                      .layer_type = LayerType::kRoPE,
                      .attention_mode = AttentionMode::kPrefill,
                      .num_threads = 2},
           ParityCase{.name = "GatedMlpPrefill",
                      .layer_type = LayerType::kGatedMlp,
                      .attention_mode = AttentionMode::kPrefill,
                      .num_threads = 2},
           ParityCase{.name = "StandardAttentionPrefill",
                      .layer_type = LayerType::kStandardAttention,
                      .attention_mode = AttentionMode::kPrefill,
                      .num_threads = 2},
           ParityCase{.name = "TransformerBlockPrefill",
                      .layer_type = LayerType::kTransformerBlock,
                      .attention_mode = AttentionMode::kPrefill,
                      .num_threads = 2},
           ParityCase{.name = "StandardAttentionDecode",
                      .layer_type = LayerType::kStandardAttention,
                      .attention_mode = AttentionMode::kDecode,
                      .num_threads = 1},
           ParityCase{.name = "DecodeOptimizedAttentionDecode",
                      .layer_type = LayerType::kDecodeOptimizedAttention,
                      .attention_mode = AttentionMode::kDecode,
                      .num_threads = 1},
           ParityCase{.name = "TransformerBlockDecode",
                      .layer_type = LayerType::kTransformerBlock,
                      .attention_mode = AttentionMode::kDecode,
                      .num_threads = 1}),
    [](const testing::TestParamInfo<ParityCase>& info) {
      return info.param.name;
    });

class FullBlockCrossRuntimeParityTest
    : public testing::TestWithParam<AttentionMode> {};

TEST_P(FullBlockCrossRuntimeParityTest, YnnpackMatchesXnnpack) {
  LayerSpec spec =
      CreateDecoupledEmbedDimSpec(LayerType::kTransformerBlock, GetParam());
  spec.sliding_window = 4;
  spec.qk_norm = true;
  spec.post_norms = true;
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);
  LitertTensorBackend xnnpack_backend(RuntimeBackend::kXnnpack);
  LitertTensorBackend ynnpack_backend(RuntimeBackend::kYnnpack);

  absl::StatusOr<std::vector<float>> xnnpack_output =
      RunLayer(xnnpack_backend, spec, data, BackendOptions{.num_threads = 2});
  absl::StatusOr<std::vector<float>> ynnpack_output =
      RunLayer(ynnpack_backend, spec, data, BackendOptions{.num_threads = 2});

  ASSERT_OK(xnnpack_output.status());
  ASSERT_OK(ynnpack_output.status());
  EXPECT_THAT(*ynnpack_output,
              Pointwise(FloatNear(kParityTolerance), *xnnpack_output));
}

INSTANTIATE_TEST_SUITE_P(QkNormPostNormsAndSlidingWindow,
                         FullBlockCrossRuntimeParityTest,
                         Values(AttentionMode::kPrefill,
                                AttentionMode::kDecode),
                         [](const testing::TestParamInfo<AttentionMode>& info) {
                           return AttentionModeToString(info.param);
                         });

struct ExternalInputCase {
  std::string name;
  LayerType layer_type;
  AttentionMode attention_mode;
  std::vector<std::pair<std::string, size_t>> expected_inputs;
};

class ExternalInputBindingTest
    : public testing::TestWithParam<ExternalInputCase> {};

TEST_P(ExternalInputBindingTest, BuildLayerBindsExternalInputsToHostBuffers) {
  const ExternalInputCase& tc = GetParam();
  LayerSpec spec = CreateTestSpec(tc.layer_type, tc.attention_mode);
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);

  absl::StatusOr<LitertLayerGraph> graph = BuildLayer(spec, data);

  ASSERT_OK(graph.status());
  EXPECT_THAT(DescribeExternalInputs(*graph, data),
              ElementsAreArray(tc.expected_inputs));
}

// Key and value bindings cover N_kv*S*D = 2*8*16 = 256 elements, or
// 2*7*16 = 224 elements for the S-1 prior tokens of a decode block.
INSTANTIATE_TEST_SUITE_P(
    AllLayers, ExternalInputBindingTest,
    Values(ExternalInputCase{"RmsNorm",
                             LayerType::kRmsNorm,
                             AttentionMode::kPrefill,
                             {{"input", 0}}},
           ExternalInputCase{"RoPE",
                             LayerType::kRoPE,
                             AttentionMode::kPrefill,
                             {{"input", 0}}},
           ExternalInputCase{"GatedMlp",
                             LayerType::kGatedMlp,
                             AttentionMode::kPrefill,
                             {{"input", 0}}},
           ExternalInputCase{"StandardAttentionPrefill",
                             LayerType::kStandardAttention,
                             AttentionMode::kPrefill,
                             {{"q", 0}, {"k", 256}, {"v", 256}}},
           ExternalInputCase{"StandardAttentionDecode",
                             LayerType::kStandardAttention,
                             AttentionMode::kDecode,
                             {{"q", 0}, {"k", 256}, {"v", 256}}},
           ExternalInputCase{"DecodeOptimizedAttention",
                             LayerType::kDecodeOptimizedAttention,
                             AttentionMode::kDecode,
                             {{"q", 0}, {"k", 256}, {"v", 256}}},
           ExternalInputCase{"TransformerBlockPrefill",
                             LayerType::kTransformerBlock,
                             AttentionMode::kPrefill,
                             {{"input", 0}}},
           ExternalInputCase{"TransformerBlockDecode",
                             LayerType::kTransformerBlock,
                             AttentionMode::kDecode,
                             {{"input", 0}, {"k", 224}, {"v", 224}}}),
    [](const testing::TestParamInfo<ExternalInputCase>& info) {
      return info.param.name;
    });

TEST(LayerBuildersTest, SingleTokenContextDecodeBlockBindsOnlyItsInput) {
  LayerSpec spec =
      CreateTestSpec(LayerType::kTransformerBlock, AttentionMode::kDecode);
  spec.seq_len_kv = 1;
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);

  absl::StatusOr<LitertLayerGraph> graph = BuildLayer(spec, data);

  ASSERT_OK(graph.status());
  EXPECT_THAT(DescribeExternalInputs(*graph, data),
              ElementsAre(Pair("input", 0)));
}

TEST(LayerBuildersTest, DecodeOptimizedAttentionRejectsPrefill) {
  LayerSpec spec = CreateTestSpec(LayerType::kDecodeOptimizedAttention,
                                  AttentionMode::kPrefill);
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);

  absl::StatusOr<LitertLayerGraph> graph =
      BuildDecodeOptimizedAttention(spec, data);

  EXPECT_EQ(graph.status().code(), absl::StatusCode::kInvalidArgument)
      << graph.status();
}

TEST(LayerBuildersTest, PrefillBlockRejectsQueryShorterThanContext) {
  LayerSpec spec = CreateTestSpec(LayerType::kTransformerBlock);
  spec.seq_len_q = 4;
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);

  absl::StatusOr<LitertLayerGraph> graph = BuildTransformerBlock(spec, data);

  EXPECT_EQ(graph.status().code(), absl::StatusCode::kInvalidArgument)
      << graph.status();
}

TEST(LayerBuildersTest, BuildLayerRejectsUndeclaredLayerType) {
  LayerSpec spec = CreateTestSpec(static_cast<LayerType>(99));

  absl::StatusOr<LitertLayerGraph> graph = BuildLayer(spec, SeededLayerData());

  EXPECT_EQ(graph.status().code(), absl::StatusCode::kInvalidArgument)
      << graph.status();
}

TEST(LayerBuildersTest, WeightBufferOfWrongSizeIsInvalidArgument) {
  LayerSpec spec = CreateTestSpec(LayerType::kRmsNorm);
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);
  data.rms_scale.pop_back();

  absl::StatusOr<LitertLayerGraph> graph = BuildRmsNorm(spec, data);

  EXPECT_EQ(graph.status().code(), absl::StatusCode::kInvalidArgument)
      << graph.status();
}

// A layer builder and a layer it builds in single precision.
struct BuilderCase {
  std::string name;
  absl::StatusOr<LitertLayerGraph> (*build)(const LayerSpec&,
                                            const SeededLayerData&);
  LayerType layer_type;
  AttentionMode attention_mode;
};

class NonFp32LayerBuilderTest
    : public testing::TestWithParam<std::tuple<BuilderCase, DataType>> {};

TEST_P(NonFp32LayerBuilderTest, BuilderReturnsUnimplemented) {
  const BuilderCase& builder = std::get<0>(GetParam());
  LayerSpec spec = CreateTestSpec(builder.layer_type, builder.attention_mode);
  spec.dtype = std::get<1>(GetParam());
  SeededLayerData data = CreateSeededLayerData(spec, kSeed);

  absl::StatusOr<LitertLayerGraph> graph = builder.build(spec, data);

  EXPECT_EQ(graph.status().code(), absl::StatusCode::kUnimplemented)
      << graph.status();
}

INSTANTIATE_TEST_SUITE_P(
    BuildersAndPrecisions, NonFp32LayerBuilderTest,
    Combine(Values(BuilderCase{"BuildRmsNorm", &BuildRmsNorm,
                               LayerType::kRmsNorm, AttentionMode::kPrefill},
                   BuilderCase{"BuildRoPE", &BuildRoPE, LayerType::kRoPE,
                               AttentionMode::kPrefill},
                   BuilderCase{"BuildGatedMlp", &BuildGatedMlp,
                               LayerType::kGatedMlp, AttentionMode::kPrefill},
                   BuilderCase{
                       "BuildStandardAttention", &BuildStandardAttention,
                       LayerType::kStandardAttention, AttentionMode::kPrefill},
                   BuilderCase{"BuildDecodeOptimizedAttention",
                               &BuildDecodeOptimizedAttention,
                               LayerType::kDecodeOptimizedAttention,
                               AttentionMode::kDecode},
                   BuilderCase{"BuildTransformerBlock", &BuildTransformerBlock,
                               LayerType::kTransformerBlock,
                               AttentionMode::kPrefill},
                   BuilderCase{"BuildLayer", &BuildLayer, LayerType::kRmsNorm,
                               AttentionMode::kPrefill}),
            Values(DataType::kFP16, DataType::kBF16, DataType::kQD8,
                   DataType::kQB4W)),
    [](const testing::TestParamInfo<std::tuple<BuilderCase, DataType>>& info) {
      return std::get<0>(info.param).name +
             DataTypeToString(std::get<1>(info.param));
    });

struct MaskCase {
  std::string name;
  std::vector<int> shape;
  bool is_local;
  int sliding_window_size;
  std::vector<float> expected_mask;
};

class FillAttentionMaskTest
    : public testing::TestWithParam<std::tuple<RuntimeBackend, MaskCase>> {};

TEST_P(FillAttentionMaskTest, MaterializesAdditiveMask) {
  const MaskCase& tc = std::get<1>(GetParam());
  BackendTensor params({.type = Type::kI32, .shape = {1}});
  BackendTensor mask =
      FillAttentionMask(params, tc.shape, tc.is_local, tc.sliding_window_size);

  absl::StatusOr<std::vector<float>> values =
      ReadConstant(std::get<0>(GetParam()), mask, tc.shape);

  ASSERT_OK(values.status());
  EXPECT_THAT(*values, ElementsAreArray(tc.expected_mask));
}

// Row i of each [S, S] matrix is the bias of query position i over key
// positions j: 0 where j may be attended, kMasked where j lies in the future
// or, for local attention, at least `sliding_window_size` positions back.
INSTANTIATE_TEST_SUITE_P(
    RuntimesAndMasks, FillAttentionMaskTest,
    Combine(Values(RuntimeBackend::kXnnpack, RuntimeBackend::kYnnpack),
            Values(MaskCase{"FullCausal",
                            {1, 1, 4, 4},
                            /*is_local=*/false,
                            /*sliding_window_size=*/0,
                            {0.0f, kMasked, kMasked, kMasked,  //
                             0.0f, 0.0f, kMasked, kMasked,     //
                             0.0f, 0.0f, 0.0f, kMasked,        //
                             0.0f, 0.0f, 0.0f, 0.0f}},
                   MaskCase{"SlidingWindow",
                            {1, 1, 4, 4},
                            /*is_local=*/true,
                            /*sliding_window_size=*/2,
                            {0.0f, kMasked, kMasked, kMasked,  //
                             0.0f, 0.0f, kMasked, kMasked,     //
                             kMasked, 0.0f, 0.0f, kMasked,     //
                             kMasked, kMasked, 0.0f, 0.0f}},
                   MaskCase{"SlidingWindowOfOne",
                            {1, 1, 3, 3},
                            /*is_local=*/true,
                            /*sliding_window_size=*/1,
                            {0.0f, kMasked, kMasked,  //
                             kMasked, 0.0f, kMasked,  //
                             kMasked, kMasked, 0.0f}},
                   MaskCase{"LocalWithoutWindowIsFullCausal",
                            {1, 1, 3, 3},
                            /*is_local=*/true,
                            /*sliding_window_size=*/0,
                            {0.0f, kMasked, kMasked,  //
                             0.0f, 0.0f, kMasked,     //
                             0.0f, 0.0f, 0.0f}},
                   MaskCase{"GlobalIgnoresWindowSize",
                            {1, 1, 3, 3},
                            /*is_local=*/false,
                            /*sliding_window_size=*/1,
                            {0.0f, kMasked, kMasked,  //
                             0.0f, 0.0f, kMasked,     //
                             0.0f, 0.0f, 0.0f}},
                   MaskCase{"MatrixOnly",
                            {2, 2},
                            /*is_local=*/false,
                            /*sliding_window_size=*/0,
                            {0.0f, kMasked,  //
                             0.0f, 0.0f}},
                   MaskCase{"RepeatedOverLeadingDims",
                            {2, 1, 2, 2},
                            /*is_local=*/false,
                            /*sliding_window_size=*/0,
                            {0.0f, kMasked,  //
                             0.0f, 0.0f,     //
                             0.0f, kMasked,  //
                             0.0f, 0.0f}})),
    [](const testing::TestParamInfo<std::tuple<RuntimeBackend, MaskCase>>&
           info) {
      return RuntimeName(std::get<0>(info.param)) +
             std::get<1>(info.param).name;
    });

struct MaskShapeCase {
  std::string name;
  std::vector<int> shape;
};

class InvalidAttentionMaskShapeTest
    : public testing::TestWithParam<std::tuple<RuntimeBackend, MaskShapeCase>> {
};

TEST_P(InvalidAttentionMaskShapeTest, LoweringReturnsInvalidArgument) {
  BackendTensor params({.type = Type::kI32, .shape = {1}});
  BackendTensor mask =
      FillAttentionMask(params, std::get<1>(GetParam()).shape,
                        /*is_local=*/false, /*sliding_window_size=*/0);

  absl::StatusOr<std::unique_ptr<NnpackRunner>> runner =
      CreateRunner(std::get<0>(GetParam()), {mask});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

// The last three shapes exceed the 2^31 - 1 element limit and must be rejected
// before any table is allocated: 65536 * 65536 leading matrices, a 65536 x
// 65536 matrix, and 1024 matrices of 2048 x 2048.
INSTANTIATE_TEST_SUITE_P(
    RuntimesAndShapes, InvalidAttentionMaskShapeTest,
    Combine(
        Values(RuntimeBackend::kXnnpack, RuntimeBackend::kYnnpack),
        Values(MaskShapeCase{"Scalar", {}}, MaskShapeCase{"Vector", {4}},
               MaskShapeCase{"NonSquareMatrix", {1, 1, 2, 4}},
               MaskShapeCase{"ZeroSequenceLength", {1, 1, 0, 0}},
               MaskShapeCase{"NegativeSequenceLength", {1, 1, -2, -2}},
               MaskShapeCase{"ZeroLeadingDim", {0, 1, 2, 2}},
               MaskShapeCase{"NegativeLeadingDim", {-1, 1, 2, 2}},
               MaskShapeCase{"LeadingDimsAboveElementLimit",
                             {65536, 65536, 1, 1}},
               MaskShapeCase{"MatrixAboveElementLimit", {1, 1, 65536, 65536}},
               MaskShapeCase{"TotalAboveElementLimit", {1024, 1, 2048, 2048}})),
    [](const testing::TestParamInfo<std::tuple<RuntimeBackend, MaskShapeCase>>&
           info) {
      return RuntimeName(std::get<0>(info.param)) +
             std::get<1>(info.param).name;
    });

TEST(FillAttentionMaskYnnpackTest, NonFp32OutputIsInvalidArgument) {
  BackendTensor params({.type = Type::kI32, .shape = {1}});
  BackendTensor mask =
      FillAttentionMask(params, {1, 1, 2, 2}, /*is_local=*/false,
                        /*sliding_window_size=*/0);
  mask.SetType(Type::kI32);

  absl::StatusOr<std::unique_ptr<NnpackRunner>> runner =
      CreateRunner(RuntimeBackend::kYnnpack, {mask});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

class FillRopeCosSinTest : public testing::TestWithParam<RuntimeBackend> {};

TEST_P(FillRopeCosSinTest, MaterializesRotationTables) {
  auto [cos_table, sin_table] =
      FillRopeCosSin<XnnpackMixinTag, YnnpackMixinTag>(
          /*seq_len=*/3, /*head_dim=*/4, /*rope_base=*/10000.0f);

  absl::StatusOr<std::vector<float>> cos_values =
      ReadConstant(GetParam(), cos_table, {1, 1, 3, 4});
  absl::StatusOr<std::vector<float>> sin_values =
      ReadConstant(GetParam(), sin_table, {1, 1, 3, 4});

  // Row `pos` holds cos or sin of pos * f_i for the frequencies f_0 = 1 and
  // f_1 = 10000^(-2/4) = 0.01, repeated for both halves of the head dimension.
  const std::vector<float> expected_cos = {
      1.0f,         1.0f,     1.0f,         1.0f,      //
      0.5403023f,   0.99995f, 0.5403023f,   0.99995f,  //
      -0.41614684f, 0.9998f,  -0.41614684f, 0.9998f};
  const std::vector<float> expected_sin = {
      0.0f,        0.0f,         0.0f,        0.0f,          //
      0.84147096f, 0.009999833f, 0.84147096f, 0.009999833f,  //
      0.9092974f,  0.019998666f, 0.9092974f,  0.019998666f};
  ASSERT_OK(cos_values.status());
  ASSERT_OK(sin_values.status());
  EXPECT_THAT(*cos_values,
              Pointwise(FloatNear(kRopeTableTolerance), expected_cos));
  EXPECT_THAT(*sin_values,
              Pointwise(FloatNear(kRopeTableTolerance), expected_sin));
}

INSTANTIATE_TEST_SUITE_P(Runtimes, FillRopeCosSinTest,
                         Values(RuntimeBackend::kXnnpack,
                                RuntimeBackend::kYnnpack),
                         RuntimeParamName);

struct RopeTableCase {
  std::string name;
  int seq_len;
  int head_dim;
  float rope_base;
};

class InvalidRopeTableTest : public testing::TestWithParam<RopeTableCase> {};

TEST_P(InvalidRopeTableTest, YnnpackLoweringReturnsInvalidArgument) {
  auto [cos_table, sin_table] =
      FillRopeCosSin<XnnpackMixinTag, YnnpackMixinTag>(
          GetParam().seq_len, GetParam().head_dim, GetParam().rope_base);

  absl::StatusOr<std::unique_ptr<NnpackRunner>> runner =
      CreateRunner(RuntimeBackend::kYnnpack, {cos_table, sin_table});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

// The 65536 x 65536 table exceeds the 2^31 - 1 element limit and must be
// rejected before it is allocated.
INSTANTIATE_TEST_SUITE_P(
    ShapesAndBases, InvalidRopeTableTest,
    Values(RopeTableCase{.name = "OddHeadDim",
                         .seq_len = 4,
                         .head_dim = 7,
                         .rope_base = 10000.0f},
           RopeTableCase{.name = "ZeroHeadDim",
                         .seq_len = 4,
                         .head_dim = 0,
                         .rope_base = 10000.0f},
           RopeTableCase{.name = "ZeroSequenceLength",
                         .seq_len = 0,
                         .head_dim = 4,
                         .rope_base = 10000.0f},
           RopeTableCase{.name = "NegativeSequenceLength",
                         .seq_len = -1,
                         .head_dim = 4,
                         .rope_base = 10000.0f},
           RopeTableCase{.name = "ZeroBase",
                         .seq_len = 4,
                         .head_dim = 4,
                         .rope_base = 0.0f},
           RopeTableCase{.name = "NegativeBase",
                         .seq_len = 4,
                         .head_dim = 4,
                         .rope_base = -10000.0f},
           RopeTableCase{.name = "NanBase",
                         .seq_len = 4,
                         .head_dim = 4,
                         .rope_base = std::numeric_limits<float>::quiet_NaN()},
           RopeTableCase{.name = "InfiniteBase",
                         .seq_len = 4,
                         .head_dim = 4,
                         .rope_base = std::numeric_limits<float>::infinity()},
           RopeTableCase{.name = "TableAboveElementLimit",
                         .seq_len = 65536,
                         .head_dim = 65536,
                         .rope_base = 10000.0f}),
    [](const testing::TestParamInfo<RopeTableCase>& info) {
      return info.param.name;
    });

// A named edit of the [1, 1, 3, 4] FP32 tables returned by FillRopeCosSin().
struct RopeOutputEdit {
  std::string name;
  void (*edit)(BackendTensor& cos_table, BackendTensor& sin_table);
};

class MalformedRopeOutputTest : public testing::TestWithParam<RopeOutputEdit> {
};

TEST_P(MalformedRopeOutputTest, YnnpackLoweringReturnsInvalidArgument) {
  auto [cos_table, sin_table] =
      FillRopeCosSin<XnnpackMixinTag, YnnpackMixinTag>(
          /*seq_len=*/3, /*head_dim=*/4, /*rope_base=*/10000.0f);
  GetParam().edit(cos_table, sin_table);

  absl::StatusOr<std::unique_ptr<NnpackRunner>> runner =
      CreateRunner(RuntimeBackend::kYnnpack, {cos_table, sin_table});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

INSTANTIATE_TEST_SUITE_P(
    OutputMetadata, MalformedRopeOutputTest,
    Values(
        RopeOutputEdit{"RankThreeTables",
                       [](BackendTensor& cos_table, BackendTensor& sin_table) {
                         cos_table.SetShape({1, 3, 4});
                         sin_table.SetShape({1, 3, 4});
                       }},
        RopeOutputEdit{"NonUnitBatchDim",
                       [](BackendTensor& cos_table, BackendTensor& sin_table) {
                         cos_table.SetShape({2, 1, 3, 4});
                         sin_table.SetShape({2, 1, 3, 4});
                       }},
        RopeOutputEdit{"NonUnitHeadDim",
                       [](BackendTensor& cos_table, BackendTensor& sin_table) {
                         cos_table.SetShape({1, 2, 3, 4});
                         sin_table.SetShape({1, 2, 3, 4});
                       }},
        RopeOutputEdit{"MismatchedTableShapes",
                       [](BackendTensor&, BackendTensor& sin_table) {
                         sin_table.SetShape({1, 1, 3, 6});
                       }},
        RopeOutputEdit{"NonFp32CosTable",
                       [](BackendTensor& cos_table, BackendTensor&) {
                         cos_table.SetType(Type::kI32);
                       }},
        RopeOutputEdit{"NonFp32SinTable",
                       [](BackendTensor&, BackendTensor& sin_table) {
                         sin_table.SetType(Type::kI32);
                       }}),
    [](const testing::TestParamInfo<RopeOutputEdit>& info) {
      return info.param.name;
    });

// RmsNorm and QkNorm over the rows [3, 4] and [4, 7] with epsilon 3.5. The
// mean squares 12.5 and 32.5 plus epsilon are 16 and 36, so the rows are
// divided by 4 and 6.
class NormLoweringTest : public testing::TestWithParam<RuntimeBackend> {};

TEST_P(NormLoweringTest, RmsNormWithoutScaleNormalizesEachRow) {
  BackendTensor input({.type = Type::kFP32, .shape = {2, 2}});
  BackendTensor no_scale(TensorHandle::Invalid());
  BackendTensor epsilon({.type = Type::kFP32, .shape = {1}, .buffer = 3.5f});
  BackendTensor normalized = RmsNorm(input, no_scale, epsilon);

  absl::StatusOr<std::vector<float>> values =
      Evaluate(GetParam(), normalized, input, {3.0f, 4.0f, 4.0f, 7.0f});

  const std::vector<float> expected = {0.75f, 1.0f, 0.6666667f, 1.1666667f};
  ASSERT_OK(values.status());
  EXPECT_THAT(*values, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(NormLoweringTest, RmsNormMultipliesNormalizedRowsByScale) {
  BackendTensor input({.type = Type::kFP32, .shape = {2, 2}});
  BackendTensor scale({.type = Type::kFP32,
                       .shape = {2},
                       .buffer = std::vector<float>{2.0f, 0.5f}});
  BackendTensor epsilon({.type = Type::kFP32, .shape = {1}, .buffer = 3.5f});
  BackendTensor normalized = RmsNorm(input, scale, epsilon);

  absl::StatusOr<std::vector<float>> values =
      Evaluate(GetParam(), normalized, input, {3.0f, 4.0f, 4.0f, 7.0f});

  const std::vector<float> expected = {1.5f, 0.5f, 1.3333334f, 0.5833333f};
  ASSERT_OK(values.status());
  EXPECT_THAT(*values, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(NormLoweringTest, QkNormWithoutScaleUsesEpsilonAttribute) {
  BackendTensor input({.type = Type::kFP32, .shape = {2, 2}});
  BackendTensor no_scale(TensorHandle::Invalid());
  BackendTensor normalized = QkNorm(input, no_scale, /*epsilon=*/3.5f);

  absl::StatusOr<std::vector<float>> values =
      Evaluate(GetParam(), normalized, input, {3.0f, 4.0f, 4.0f, 7.0f});

  const std::vector<float> expected = {0.75f, 1.0f, 0.6666667f, 1.1666667f};
  ASSERT_OK(values.status());
  EXPECT_THAT(*values, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(NormLoweringTest, QkNormWithScaleUsesEpsilonTensor) {
  BackendTensor input({.type = Type::kFP32, .shape = {2, 2}});
  BackendTensor scale({.type = Type::kFP32,
                       .shape = {2},
                       .buffer = std::vector<float>{2.0f, 0.5f}});
  BackendTensor epsilon({.type = Type::kFP32, .shape = {1}, .buffer = 3.5f});
  BackendTensor normalized = QkNorm(input, scale, epsilon);

  absl::StatusOr<std::vector<float>> values =
      Evaluate(GetParam(), normalized, input, {3.0f, 4.0f, 4.0f, 7.0f});

  const std::vector<float> expected = {1.5f, 0.5f, 1.3333334f, 0.5833333f};
  ASSERT_OK(values.status());
  EXPECT_THAT(*values, Pointwise(FloatNear(kAnalyticalTolerance), expected));
}

TEST_P(NormLoweringTest, RmsNormOfScalarInputIsInvalidArgument) {
  BackendTensor scalar_input({.type = Type::kFP32, .shape = {}});
  BackendTensor no_scale(TensorHandle::Invalid());
  BackendTensor epsilon({.type = Type::kFP32, .shape = {1}, .buffer = 3.5f});
  BackendTensor normalized = RmsNorm(scalar_input, no_scale, epsilon);

  absl::StatusOr<std::unique_ptr<NnpackRunner>> runner =
      CreateRunner(GetParam(), {normalized});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

TEST_P(NormLoweringTest, QkNormOfScalarInputIsInvalidArgument) {
  BackendTensor scalar_input({.type = Type::kFP32, .shape = {}});
  BackendTensor no_scale(TensorHandle::Invalid());
  BackendTensor normalized = QkNorm(scalar_input, no_scale, /*epsilon=*/3.5f);

  absl::StatusOr<std::unique_ptr<NnpackRunner>> runner =
      CreateRunner(GetParam(), {normalized});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

INSTANTIATE_TEST_SUITE_P(Runtimes, NormLoweringTest,
                         Values(RuntimeBackend::kXnnpack,
                                RuntimeBackend::kYnnpack),
                         RuntimeParamName);

TEST(YnnpackNormLoweringTest, RmsNormOfNonFp32InputIsInvalidArgument) {
  BackendTensor input({.type = Type::kI32, .shape = {2, 2}});
  BackendTensor no_scale(TensorHandle::Invalid());
  BackendTensor epsilon({.type = Type::kFP32, .shape = {1}, .buffer = 3.5f});
  BackendTensor normalized = RmsNorm(input, no_scale, epsilon);

  absl::StatusOr<std::unique_ptr<NnpackRunner>> runner =
      CreateRunner(RuntimeBackend::kYnnpack, {normalized});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

TEST(YnnpackNormLoweringTest, RmsNormWithNonFp32ScaleIsInvalidArgument) {
  BackendTensor input({.type = Type::kFP32, .shape = {2, 2}});
  BackendTensor scale(
      {.type = Type::kI32, .shape = {2}, .buffer = std::vector<int32_t>{2, 1}});
  BackendTensor epsilon({.type = Type::kFP32, .shape = {1}, .buffer = 3.5f});
  BackendTensor normalized = RmsNorm(input, scale, epsilon);

  absl::StatusOr<std::unique_ptr<NnpackRunner>> runner =
      CreateRunner(RuntimeBackend::kYnnpack, {normalized});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

TEST(YnnpackNormLoweringTest, RmsNormWithNonFp32EpsilonIsInvalidArgument) {
  BackendTensor input({.type = Type::kFP32, .shape = {2, 2}});
  BackendTensor no_scale(TensorHandle::Invalid());
  BackendTensor epsilon(
      {.type = Type::kI32, .shape = {1}, .buffer = int32_t{3}});
  BackendTensor normalized = RmsNorm(input, no_scale, epsilon);

  absl::StatusOr<std::unique_ptr<NnpackRunner>> runner =
      CreateRunner(RuntimeBackend::kYnnpack, {normalized});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

TEST(YnnpackNormLoweringTest, QkNormOfNonFp32InputIsInvalidArgument) {
  BackendTensor input({.type = Type::kI32, .shape = {2, 2}});
  BackendTensor no_scale(TensorHandle::Invalid());
  BackendTensor normalized = QkNorm(input, no_scale, /*epsilon=*/3.5f);

  absl::StatusOr<std::unique_ptr<NnpackRunner>> runner =
      CreateRunner(RuntimeBackend::kYnnpack, {normalized});

  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument)
      << runner.status();
}

}  // namespace
}  // namespace layer_ml_bench::litert
