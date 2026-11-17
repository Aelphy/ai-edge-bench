# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "status_test_util",
    testonly = True,
    hdrs = ["status_test_util.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "layer_spec",
    srcs = ["layer_spec.cc"],
    hdrs = ["layer_spec.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "layer_backend",
    srcs = ["layer_backend.cc"],
    hdrs = ["layer_backend.h"],
    deps = [
        ":layer_spec",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@pthreadpool",
    ],
)

cc_library(
    name = "fake_layer_backend",
    testonly = True,
    srcs = ["fake_layer_backend.cc"],
    hdrs = ["fake_layer_backend.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":layer_backend",
        ":layer_spec",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "test_matchers",
    testonly = True,
    hdrs = ["test_matchers.h"],
    visibility = ["//visibility:private"],
    deps = ["@com_google_googletest//:gtest"],
)

cc_test(
    name = "layer_spec_test",
    srcs = ["layer_spec_test.cc"],
    deps = [
        ":fake_layer_backend",
        ":layer_backend",
        ":layer_spec",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tracking_allocator",
    srcs = ["tracking_allocator.cc"],
    hdrs = ["tracking_allocator.h"],
    deps = [
        "@XNNPACK",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "tracking_allocator_test",
    srcs = ["tracking_allocator_test.cc"],
    deps = [
        ":tracking_allocator",
        "@XNNPACK",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "layer_builders",
    srcs = ["layer_builders.cc"],
    hdrs = ["layer_builders.h"],
    deps = [
        ":layer_backend",
        ":layer_spec",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/backends/ynnpack:arithmetic",
        "//tensor/examples/ops/transformer:transformer_block",
        "//tensor/examples/ops/transformer:transformer_ops",
        "//tensor/examples/ops/transformer:transformer_ops_graph",
        "//tensor/examples/ops/transformer:transformer_ops_xnnpack",
        "//tensor/examples/ops/transformer:transformer_ops_ynnpack",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "litert_tensor_backend",
    srcs = ["litert_tensor_backend.cc"],
    hdrs = ["litert_tensor_backend.h"],
    deps = [
        ":layer_backend",
        ":layer_builders",
        ":layer_spec",
        ":tracking_allocator",
        "//tensor/runners/common_nnpack:runner",
        "//tensor/runners/xnnpack:runner",
        "//tensor/runners/ynnpack:runner",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@pthreadpool",
    ],
)

cc_test(
    name = "litert_tensor_backend_test",
    srcs = ["litert_tensor_backend_test.cc"],
    deps = [
        ":fake_layer_backend",
        ":layer_backend",
        ":layer_builders",
        ":layer_spec",
        ":litert_tensor_backend",
        ":status_test_util",
        ":test_matchers",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/backends/ynnpack:arithmetic",
        "//tensor/examples/ops/transformer:transformer_ops",
        "//tensor/examples/ops/transformer:transformer_ops_xnnpack",
        "//tensor/examples/ops/transformer:transformer_ops_ynnpack",
        "//tensor/runners/common_nnpack:runner",
        "//tensor/runners/xnnpack:runner",
        "//tensor/runners/ynnpack:runner",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "benchmark_runner",
    srcs = ["benchmark_runner.cc"],
    hdrs = ["benchmark_runner.h"],
    deps = [
        ":layer_backend",
        ":layer_spec",
        ":parity_validator",
        ":tracking_allocator",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_benchmark//:benchmark",
        "@pthreadpool",
    ],
)

cc_test(
    name = "benchmark_runner_test",
    srcs = ["benchmark_runner_test.cc"],
    deps = [
        ":benchmark_runner",
        ":fake_layer_backend",
        ":layer_backend",
        ":layer_spec",
        ":litert_tensor_backend",
        ":parity_validator",
        ":status_test_util",
        ":tracking_allocator",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "parity_validator",
    srcs = ["parity_validator.cc"],
    hdrs = ["parity_validator.h"],
    deps = [
        ":layer_backend",
        ":layer_spec",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "parity_test",
    srcs = ["parity_test.cc"],
    deps = [
        ":fake_layer_backend",
        ":layer_backend",
        ":layer_spec",
        ":litert_tensor_backend",
        ":parity_validator",
        ":status_test_util",
        ":test_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "suite_loader",
    srcs = ["suite_loader.cc"],
    hdrs = ["suite_loader.h"],
    deps = [
        ":layer_spec",
        ":litert_tensor_backend",
        "//tensor/examples/utils:minijson",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "suite_loader_test",
    srcs = ["suite_loader_test.cc"],
    deps = [
        ":layer_spec",
        ":litert_tensor_backend",
        ":status_test_util",
        ":suite_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "layer_bench",
    srcs = ["layer_benchmark.cc"],
    deps = [
        ":benchmark_runner",
        ":layer_backend",
        ":litert_tensor_backend",
        ":parity_validator",
        ":suite_loader",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_benchmark//:benchmark",
    ],
)
