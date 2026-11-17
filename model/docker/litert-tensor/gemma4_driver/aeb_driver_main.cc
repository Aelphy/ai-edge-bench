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

#include <chrono>  // NOLINT
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "xnnpack.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensor/buffer.h"
#include "tensor/examples/gemma3/tokenizer.h"
#include "tensor/examples/gemma4/aeb_driver.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_runtime.h"
#include "tensor/examples/utils/initialization.h"
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tflite/delegates/xnnpack/weight_cache.h"

ABSL_FLAG(std::string, weights, "",
          "Path to safetensors file or HuggingFace model directory.");
ABSL_FLAG(std::string, tokenizer, "",
          "Optional path to SentencePiece tokenizer.model file.");
ABSL_FLAG(int, num_threads, 4, "Number of CPU threads for XNNPACK/YNNPACK.");
ABSL_FLAG(int, ctx, 4096, "Maximum context length.");
ABSL_FLAG(std::string, weight_cache, "",
          "Path to XNNPACK weight cache file (or :auto / :memory).");
ABSL_FLAG(std::string, cache_dir, "",
          "Optional cache directory for XNNPACK weight cache.");
ABSL_FLAG(bool, cache_prefix, false, "Enable KV prefix caching across turns.");
ABSL_FLAG(std::string, backend_name, "xnnpack",
          "Backend label reported in ready metadata (xnnpack or ynnpack).");
ABSL_FLAG(std::string, dtype, "fp32",
          "Execution data type reported in ready metadata.");

namespace litert::tensor::examples::gemma4 {
namespace {

class XnnpackWeightHooks final : public TensorMappingHooks {
 public:
  explicit XnnpackWeightHooks(
      tflite::xnnpack::MMapWeightCacheProvider* weight_cache)
      : weight_cache_(weight_cache) {}

  absl::Status OnLoaded(absl::string_view model_name,
                        TensorHandle& weight) override {
    if (weight_cache_ == nullptr) {
      return absl::OkStatus();
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & buffer, weight.GetBuffer());
    auto locked = buffer.Lock();
    const uint64_t identifier =
        static_cast<uint64_t>(std::hash<absl::string_view>{}(model_name));
    if (!weight_cache_->MapBufferIdentifier(locked.data(), locked.size(),
                                            identifier)) {
      return absl::InternalError(
          absl::StrCat("Failed to map weight identifier for ", model_name));
    }
    return absl::OkStatus();
  }

 private:
  tflite::xnnpack::MMapWeightCacheProvider* weight_cache_;
};

int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

absl::Status RunServer(absl::string_view argv0) {
  const int64_t t_load0 = NowNs();
  std::string backend_name = absl::GetFlag(FLAGS_backend_name);
  if (backend_name == "xnnpack" && absl::EndsWith(argv0, "_ynnpack")) {
    backend_name = "ynnpack";
  }
  const bool use_ynnpack = (backend_name == "ynnpack");
  if (!use_ynnpack) {
    if (xnn_initialize(/*allocator=*/nullptr) != xnn_status_success) {
      return absl::InternalError("Failed to initialize XNNPACK");
    }
  }

  const std::string weights_path = absl::GetFlag(FLAGS_weights);
  if (weights_path.empty()) {
    return absl::InvalidArgumentError("--weights is required");
  }

  std::string tokenizer_path = absl::GetFlag(FLAGS_tokenizer);
  if (tokenizer_path.empty()) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path wp(weights_path);
    fs::path candidate =
        (fs::is_directory(wp, ec) ? wp : wp.parent_path()) / "tokenizer.model";
    if (fs::is_regular_file(candidate, ec)) {
      tokenizer_path = candidate.string();
    }
  }

  std::optional<GemmaTokenizerSP> tokenizer;
  if (!tokenizer_path.empty()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(GemmaTokenizerSP tok,
                                GemmaTokenizerSP::Load(tokenizer_path));
    tokenizer.emplace(std::move(tok));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorLoader loader,
                              SafetensorLoader::Load(weights_path));
  LRT_TENSOR_ASSIGN_OR_RETURN(ModelVariant variant,
                              DeduceModelVariant(loader));
  const Config config = Config::From(variant);

  std::string cache_dir = absl::GetFlag(FLAGS_cache_dir);
  std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache);
  if (weight_cache_path.empty() && !cache_dir.empty()) {
    if (cache_dir == ":memory") {
      weight_cache_path = ":memory";
    } else {
      std::error_code ec;
      std::filesystem::create_directories(cache_dir, ec);
      weight_cache_path =
          (std::filesystem::path(cache_dir) / "gemma4_weights.cache").string();
    }
  } else if (weight_cache_path == ":auto") {
    weight_cache_path = absl::StrCat(weights_path, ".cache");
  } else if (weight_cache_path.empty()) {
    weight_cache_path = ":memory";
  }

  tflite::xnnpack::MMapWeightCacheProvider weight_cache_provider;
  tflite::xnnpack::MMapWeightCacheProvider* weight_cache = nullptr;
  if (!use_ynnpack && !weight_cache_path.empty()) {
    if (!weight_cache_provider.LoadOrStartBuild(weight_cache_path.c_str())) {
      return absl::InternalError(absl::StrCat(
          "Failed to load or start build for XNNPACK weight cache: ",
          weight_cache_path));
    }
    weight_cache = &weight_cache_provider;
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      LoadedTensors loaded_tensors,
      LoadWeightsAndPrepareTensors(
          std::move(loader), config, [&](LazyTensorMapping& weights) {
            if (weight_cache != nullptr) {
              weights.Register<XnnpackWeightHooks>(weight_cache);
            }
          }));

  AebGemma4Session::Options options;
  options.backend_name = backend_name;
  options.dtype = absl::GetFlag(FLAGS_dtype);
  options.model_variant = AbslUnparseFlag(variant);
  options.num_threads = absl::GetFlag(FLAGS_num_threads);
  options.ctx = absl::GetFlag(FLAGS_ctx);
  options.cache_dir = cache_dir;
  options.cache_prefix = absl::GetFlag(FLAGS_cache_prefix);

  const int64_t load_ns = NowNs() - t_load0;
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::unique_ptr<AebGemma4Session> session,
      AebGemma4Session::Create(config, std::move(loaded_tensors),
                               std::move(tokenizer), weight_cache, options,
                               load_ns));
  return session->Serve(std::cin, std::cout);
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4

int main(int argc, char** argv) {
  const std::string argv0 = (argc > 0 && argv[0] != nullptr) ? argv[0] : "";
  litert::tensor::Initialize("gemma4_tensor_driver", argc, argv, true);
  absl::Status status = litert::tensor::examples::gemma4::RunServer(argv0);
  if (!status.ok()) {
    std::cout << "{\"event\":\"error\",\"error\":\"" << status.message()
              << "\"}\n"
              << std::flush;
    ABSL_LOG(ERROR) << "gemma4_tensor_driver failed: " << status;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
