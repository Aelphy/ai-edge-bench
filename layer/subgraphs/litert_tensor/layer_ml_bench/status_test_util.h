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

#ifndef TENSOR_BENCHMARKS_LAYER_STATUS_TEST_UTIL_H_
#define TENSOR_BENCHMARKS_LAYER_STATUS_TEST_UTIL_H_

#include <gtest/gtest.h>
#include "absl/status/status.h"

#ifndef ASSERT_OK
#define ASSERT_OK(expr)                                        \
  do {                                                         \
    const absl::Status _assert_ok_status = (expr);             \
    ASSERT_TRUE(_assert_ok_status.ok())                        \
        << _assert_ok_status.ToString();                       \
  } while (0)
#endif

#endif  // TENSOR_BENCHMARKS_LAYER_STATUS_TEST_UTIL_H_
