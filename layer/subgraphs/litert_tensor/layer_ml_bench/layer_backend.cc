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

#include "tensor/benchmarks/layer/layer_backend.h"

#include <cstdint>
#include <random>
#include <vector>

#include "tensor/benchmarks/layer/layer_spec.h"

namespace layer_ml_bench {

SeededLayerData CreateSeededLayerData(const LayerSpec& spec, uint32_t seed) {
  const int64_t b = spec.batch_size;
  const int64_t s = spec.EffectiveSeqLenKV();
  const int64_t t = spec.EffectiveSeqLenQ();
  const int64_t h = spec.num_heads;
  const int64_t d = spec.head_dim;
  const int64_t m = spec.ModelDim();
  const int64_t q_dim = spec.QDim();
  const int64_t kv_dim = spec.KVDim();
  const int64_t n_kv = spec.num_kv_heads;
  const int64_t f = spec.hidden_dim;

  // Allocate only the buffers consumed by spec.layer_type so isolated
  // operator benchmarks (e.g., RmsNorm, RoPE) do not materialize gigabytes of
  // unused MLP or projection weights.
  SeededLayerData data;
  switch (spec.layer_type) {
    case LayerType::kRmsNorm:
      data.input.resize(b * t * m);
      data.rms_scale.resize(m);
      break;
    case LayerType::kRoPE:
      data.input.resize(b * t * q_dim);
      data.rope_cos.resize(t * d);
      data.rope_sin.resize(t * d);
      break;
    case LayerType::kGatedMlp:
      data.input.resize(b * t * m);
      data.w_gate.resize(f * m);
      data.w_up.resize(f * m);
      data.w_down.resize(m * f);
      break;
    case LayerType::kStandardAttention:
    case LayerType::kDecodeOptimizedAttention:
      data.q.resize(b * h * t * d);
      data.k.resize(b * n_kv * s * d);
      data.v.resize(b * n_kv * s * d);
      break;
    case LayerType::kTransformerBlock:
      data.input.resize(b * t * m);
      if (spec.attention_mode == AttentionMode::kDecode && s > 1) {
        data.k.resize(b * n_kv * (s - 1) * d);
        data.v.resize(b * n_kv * (s - 1) * d);
      }
      data.rms_scale.resize(m);
      data.rope_cos.resize(t * d);
      data.rope_sin.resize(t * d);
      data.w_q.resize(q_dim * m);
      data.w_k.resize(kv_dim * m);
      data.w_v.resize(kv_dim * m);
      data.w_o.resize(m * q_dim);
      data.w_gate.resize(f * m);
      data.w_up.resize(f * m);
      data.w_down.resize(m * f);
      data.ffn_rms_scale.resize(m);
      if (spec.qk_norm) {
        data.q_norm_scale.resize(d);
        data.k_norm_scale.resize(d);
      }
      if (spec.post_norms) {
        data.post_attn_rms_scale.resize(m);
        data.post_ffn_rms_scale.resize(m);
      }
      break;
  }

  std::mt19937 prng(seed);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

  auto fill = [&](std::vector<float>& buf) {
    for (float& val : buf) {
      val = dist(prng);
    }
  };

  // Populate allocated buffers in strict deterministic order.
  fill(data.input);
  fill(data.q);
  fill(data.k);
  fill(data.v);
  fill(data.rms_scale);
  fill(data.rope_cos);
  fill(data.rope_sin);
  fill(data.w_q);
  fill(data.w_k);
  fill(data.w_v);
  fill(data.w_o);
  fill(data.w_gate);
  fill(data.w_up);
  fill(data.w_down);
  fill(data.ffn_rms_scale);
  fill(data.q_norm_scale);
  fill(data.k_norm_scale);
  fill(data.post_attn_rms_scale);
  fill(data.post_ffn_rms_scale);

  return data;
}

}  // namespace layer_ml_bench
