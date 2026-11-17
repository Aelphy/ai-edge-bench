/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/examples/gemma4/aeb_driver.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xnnpack.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_runtime.h"
#include "tensor/examples/gemma4/gemma4_weights.h"
#include "tensor/examples/gemma4/helpers/quantized_embedding.h"
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::testing::HasSubstr;

Config MakeTinyE2BConfig() {
  Config c = Config::E2B();
  c.num_layers = 2;
  c.vocab_size = 16;
  c.embed_dim = 4;
  c.hidden_dim = 8;
  c.head_dim = 4;
  c.global_key_size = 4;
  c.num_heads = 2;
  c.num_kv_heads = 1;
  c.per_layer_input_dim = 2;
  c.frac_shared_layers = 0.0f;
  c.attention_pattern_size = 2;
  return c;
}

absl::StatusOr<LoadedTensors> CreateTinyLoadedTensors(const Config& c) {
  absl::flat_hash_map<std::string, TensorHandle> raw;

  std::vector<float> embed_data(c.vocab_size * c.embed_dim, 0.0f);
  for (int i = 0; i < c.vocab_size; ++i) {
    for (int j = 0; j < c.embed_dim; ++j) {
      embed_data[i * c.embed_dim + j] = 0.05f * (i + 1) + 0.01f * j;
    }
  }
  TensorHandle embed_tokens({.name = "model.embed_tokens.weight",
                             .type = Type::kFP32,
                             .shape = {c.vocab_size, c.embed_dim},
                             .buffer = embed_data});
  raw.emplace("model.embed_tokens.weight", embed_tokens);

  const int ple_total_dim = c.num_layers * c.per_layer_input_dim;
  std::vector<float> ple_embed_data(c.vocab_size * ple_total_dim, 0.02f);
  TensorHandle embed_tokens_per_layer(
      {.name = "model.embed_tokens_per_layer.weight",
       .type = Type::kFP32,
       .shape = {c.vocab_size, ple_total_dim},
       .buffer = ple_embed_data});
  raw.emplace("model.embed_tokens_per_layer.weight", embed_tokens_per_layer);

  raw.emplace(
      "model.per_layer_model_projection.weight",
      TensorInit{
          .name = "model.per_layer_model_projection.weight",
          .type = Type::kFP32,
          .shape = {c.num_layers, c.per_layer_input_dim, c.embed_dim},
          .buffer = std::vector<float>(
              c.num_layers * c.per_layer_input_dim * c.embed_dim, 0.1f)});
  raw.emplace("model.per_layer_projection_norm.weight",
              TensorInit{.name = "model.per_layer_projection_norm.weight",
                         .type = Type::kFP32,
                         .shape = {c.per_layer_input_dim},
                         .buffer = std::vector<float>(c.per_layer_input_dim,
                                                      1.0f)});
  raw.emplace("model.norm.weight",
              TensorInit{.name = "model.norm.weight",
                         .type = Type::kFP32,
                         .shape = {c.embed_dim},
                         .buffer = std::vector<float>(c.embed_dim, 1.0f)});

  for (int l = 0; l < c.num_layers; ++l) {
    std::string p = absl::StrCat("model.layers.", l);
    const int hd =
        c.GetLayerType(l) == Config::LayerType::kGlobal ? c.global_key_size
                                                        : c.head_dim;
    const int q_dim = c.num_heads * hd;
    const int kv_dim = c.num_kv_heads * hd;
    raw.emplace(absl::StrCat(p, ".self_attn.q_proj.weight"),
                TensorInit{.name = "q_proj",
                           .type = Type::kFP32,
                           .shape = {q_dim, c.embed_dim},
                           .buffer = std::vector<float>(q_dim * c.embed_dim,
                                                        0.05f)});
    raw.emplace(absl::StrCat(p, ".self_attn.k_proj.weight"),
                TensorInit{.name = "k_proj",
                           .type = Type::kFP32,
                           .shape = {kv_dim, c.embed_dim},
                           .buffer = std::vector<float>(kv_dim * c.embed_dim,
                                                        0.05f)});
    raw.emplace(absl::StrCat(p, ".self_attn.v_proj.weight"),
                TensorInit{.name = "v_proj",
                           .type = Type::kFP32,
                           .shape = {kv_dim, c.embed_dim},
                           .buffer = std::vector<float>(kv_dim * c.embed_dim,
                                                        0.05f)});
    raw.emplace(absl::StrCat(p, ".self_attn.o_proj.weight"),
                TensorInit{.name = "o_proj",
                           .type = Type::kFP32,
                           .shape = {c.embed_dim, q_dim},
                           .buffer = std::vector<float>(c.embed_dim * q_dim,
                                                        0.05f)});
    raw.emplace(absl::StrCat(p, ".self_attn.q_norm.weight"),
                TensorInit{.name = "q_norm",
                           .type = Type::kFP32,
                           .shape = {hd},
                           .buffer = std::vector<float>(hd, 1.0f)});
    raw.emplace(absl::StrCat(p, ".self_attn.k_norm.weight"),
                TensorInit{.name = "k_norm",
                           .type = Type::kFP32,
                           .shape = {hd},
                           .buffer = std::vector<float>(hd, 1.0f)});
    raw.emplace(absl::StrCat(p, ".input_layernorm.weight"),
                TensorInit{.name = "in_norm",
                           .type = Type::kFP32,
                           .shape = {c.embed_dim},
                           .buffer = std::vector<float>(c.embed_dim, 1.0f)});
    raw.emplace(absl::StrCat(p, ".post_attention_layernorm.weight"),
                TensorInit{.name = "post_attn_norm",
                           .type = Type::kFP32,
                           .shape = {c.embed_dim},
                           .buffer = std::vector<float>(c.embed_dim, 1.0f)});
    raw.emplace(absl::StrCat(p, ".pre_feedforward_layernorm.weight"),
                TensorInit{.name = "pre_ffn_norm",
                           .type = Type::kFP32,
                           .shape = {c.embed_dim},
                           .buffer = std::vector<float>(c.embed_dim, 1.0f)});
    raw.emplace(absl::StrCat(p, ".post_feedforward_layernorm.weight"),
                TensorInit{.name = "post_ffn_norm",
                           .type = Type::kFP32,
                           .shape = {c.embed_dim},
                           .buffer = std::vector<float>(c.embed_dim, 1.0f)});
    raw.emplace(absl::StrCat(p, ".mlp.gate_proj.weight"),
                TensorInit{.name = "gate_proj",
                           .type = Type::kFP32,
                           .shape = {c.hidden_dim, c.embed_dim},
                           .buffer = std::vector<float>(
                               c.hidden_dim * c.embed_dim, 0.05f)});
    raw.emplace(absl::StrCat(p, ".mlp.up_proj.weight"),
                TensorInit{.name = "up_proj",
                           .type = Type::kFP32,
                           .shape = {c.hidden_dim, c.embed_dim},
                           .buffer = std::vector<float>(
                               c.hidden_dim * c.embed_dim, 0.05f)});
    raw.emplace(absl::StrCat(p, ".mlp.down_proj.weight"),
                TensorInit{.name = "down_proj",
                           .type = Type::kFP32,
                           .shape = {c.embed_dim, c.hidden_dim},
                           .buffer = std::vector<float>(
                               c.embed_dim * c.hidden_dim, 0.05f)});
    raw.emplace(absl::StrCat(p, ".per_layer_input_gate.weight"),
                TensorInit{.name = "ple_gate",
                           .type = Type::kFP32,
                           .shape = {c.per_layer_input_dim, c.embed_dim},
                           .buffer = std::vector<float>(
                               c.per_layer_input_dim * c.embed_dim, 0.05f)});
    raw.emplace(absl::StrCat(p, ".per_layer_projection.weight"),
                TensorInit{.name = "ple_proj",
                           .type = Type::kFP32,
                           .shape = {c.embed_dim, c.per_layer_input_dim},
                           .buffer = std::vector<float>(
                               c.embed_dim * c.per_layer_input_dim, 0.05f)});
    raw.emplace(absl::StrCat(p, ".post_per_layer_input_norm.weight"),
                TensorInit{.name = "ple_norm",
                           .type = Type::kFP32,
                           .shape = {c.embed_dim},
                           .buffer = std::vector<float>(c.embed_dim, 1.0f)});
    raw.emplace(absl::StrCat(p, ".layer_scalar"),
                TensorInit{.name = "layer_scalar",
                           .type = Type::kFP32,
                           .shape = {1},
                           .buffer = 1.0f});
  }

  LazyTensorMapping mapping(std::move(raw));
  mapping.Register<Gemma4WeightHooks>(c);
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto tok_emb, GemmaEmbeddingTable::Create(embed_tokens, c.embed_dim));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto ple_emb,
      GemmaEmbeddingTable::Create(embed_tokens_per_layer, ple_total_dim));
  return LoadedTensors{std::move(mapping), std::move(tok_emb),
                       std::move(ple_emb)};
}

TEST(AebDriverTest, SampleTokenGreedyAndSeeded) {
  std::vector<float> logits = {0.1f, 2.5f, 9.0f, 1.2f};
  SamplingParams greedy;
  EXPECT_EQ(SampleToken(logits, greedy, nullptr), 2);

  SamplingParams seeded{.temperature = 1.0f, .top_k = 2, .top_p = 0.95f, .seed = 42};
  std::mt19937 rng1(seeded.seed);
  std::mt19937 rng2(seeded.seed);
  const int32_t tok1 = SampleToken(logits, seeded, &rng1);
  const int32_t tok2 = SampleToken(logits, seeded, &rng2);
  EXPECT_EQ(tok1, tok2);
  EXPECT_TRUE(tok1 == 2 || tok1 == 1);
}

TEST(AebDriverTest, SessionServeJsonProtocolAndPrefixCache) {
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  const Config config = MakeTinyE2BConfig();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LoadedTensors loaded,
                                  CreateTinyLoadedTensors(config));

  AebGemma4Session::Options opts;
  opts.backend_name = "xnnpack";
  opts.dtype = "fp32";
  opts.model_variant = "E2B";
  opts.num_threads = 2;
  opts.cache_prefix = true;

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto session,
      AebGemma4Session::Create(config, std::move(loaded), std::nullopt, nullptr,
                               opts, /*load_ns=*/12345));

  std::istringstream in(
      "{\"id\":1,\"op\":\"generate\",\"prompt_ids\":[2,3,4,5],"
      "\"max_tokens\":3,\"ignore_eos\":true,\"return_prompt_ids\":true}\n"
      "{\"id\":2,\"op\":\"generate\",\"prompt_ids\":[2,3,4,6],"
      "\"prefix_tokens\":3,\"cache_prefix\":true,\"max_tokens\":2,"
      "\"ignore_eos\":true}\n"
      "{\"id\":3,\"op\":\"generate\",\"prompt_ids\":[1,2,3,4,5,6,7],"
      "\"max_tokens\":2,\"ignore_eos\":true}\n"
      "{\"id\":4,\"op\":\"quit\"}\n");
  std::ostringstream out;
  ASSERT_TRUE(session->Serve(in, out).ok());

  const std::string output = out.str();
  EXPECT_THAT(output, HasSubstr("\"event\":\"ready\""));
  EXPECT_THAT(output, HasSubstr("\"framework\":\"litert-tensor\""));
  EXPECT_THAT(output, HasSubstr("\"id\":1,\"ok\":true,\"n_prompt\":4"));
  EXPECT_THAT(output, HasSubstr("\"n_gen\":3"));
  EXPECT_THAT(output, HasSubstr("\"id\":2,\"ok\":true,\"n_prompt\":4,"
                                "\"n_cached\":3,\"n_prefilled\":1,\"n_gen\":2"));
  EXPECT_THAT(output, HasSubstr("\"id\":3,\"ok\":true,\"n_prompt\":7,"
                                "\"n_cached\":0,\"n_prefilled\":7,\"n_gen\":2"));
  EXPECT_THAT(output, HasSubstr("\"id\":4,\"ok\":true"));
}

TEST(AebDriverTest, YnnpackSessionAndParityWithXnnpack) {
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  const Config config = MakeTinyE2BConfig();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LoadedTensors xnn_loaded,
                                  CreateTinyLoadedTensors(config));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LoadedTensors ynn_loaded,
                                  CreateTinyLoadedTensors(config));

  AebGemma4Session::Options xnn_opts;
  xnn_opts.backend_name = "xnnpack";
  xnn_opts.num_threads = 2;
  xnn_opts.cache_prefix = true;

  AebGemma4Session::Options ynn_opts = xnn_opts;
  ynn_opts.backend_name = "ynnpack";

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto xnn_session,
      AebGemma4Session::Create(config, std::move(xnn_loaded), std::nullopt,
                               nullptr, xnn_opts, /*load_ns=*/100));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto ynn_session,
      AebGemma4Session::Create(config, std::move(ynn_loaded), std::nullopt,
                               nullptr, ynn_opts, /*load_ns=*/100));

  GenerateRequest req1;
  req1.prompt_ids = {2, 3, 4, 5};
  req1.max_tokens = 3;
  req1.ignore_eos = true;
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(GenerateResult xnn_res1,
                                  xnn_session->Generate(req1));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(GenerateResult ynn_res1,
                                  ynn_session->Generate(req1));
  EXPECT_EQ(xnn_res1.gen_ids, ynn_res1.gen_ids);

  GenerateRequest req2;
  req2.prompt_ids = {2, 3, 4, 6};
  req2.prefix_tokens = 3;
  req2.cache_prefix = true;
  req2.max_tokens = 2;
  req2.ignore_eos = true;
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(GenerateResult xnn_res2,
                                  xnn_session->Generate(req2));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(GenerateResult ynn_res2,
                                  ynn_session->Generate(req2));
  EXPECT_EQ(ynn_res2.n_cached, 3);
  EXPECT_EQ(ynn_res2.n_prefilled, 1);
  EXPECT_EQ(xnn_res2.gen_ids, ynn_res2.gen_ids);
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
