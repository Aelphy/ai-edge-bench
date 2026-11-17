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

// This header explicitly declares some OpMixin specializations. When
// included, nothing declared/defined in this file will ever be explicitly
// used. Without including it, YNNPACK graph creation will not work for these
// ops.

// IWYU pragma: always_keep

#ifndef TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_YNNPACK_H_
#define TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_YNNPACK_H_

#include "absl/status/status.h"
#include "tensor/backends/ynnpack/arithmetic.h"
#include "tensor/examples/ops/transformer/transformer_ops_graph.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/mixin.h"

namespace litert::tensor::graph {

template <>
class OpMixin<FillAttentionMaskOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<FillRopeCosSinOperation, YnnpackMixinTag>
    : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RmsNormOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<QkNormOperation, YnnpackMixinTag> : public YnnpackOperation {
 public:
  absl::Status ToYnnpack(const graph::Operation& op,
                         YnnpackBuildContext& ctx) const override;
};

}  // namespace litert::tensor::graph

#endif  // TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_OPS_YNNPACK_H_
