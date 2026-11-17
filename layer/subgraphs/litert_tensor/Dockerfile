# syntax=docker/dockerfile:1.7
#
# Standalone Level 2 (`layer/`) LiteRT Tensor API benchmark image
# (`layer_bench_xnnpack` & `layer_bench_ynnpack` built from google-ai-edge/LiteRT).

FROM ubuntu:24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc clang cmake make pkg-config \
      openjdk-17-jdk-headless git curl unzip tar ca-certificates \
      python3 python3-dev python3-numpy npm \
 && rm -rf /var/lib/apt/lists/* \
 && npm install -g @bazel/bazelisk

ARG LITERT_REPO=https://github.com/google-ai-edge/LiteRT
ARG LITERT_REF=5d9e2a42e02b6ea56f9fab06a94fa425b9facfa6

WORKDIR /src
RUN git init -q . \
 && git remote add origin "${LITERT_REPO}" \
 && git fetch --depth 1 origin "${LITERT_REF}" \
 && git checkout -q --detach FETCH_HEAD

COPY layer_ml_bench/ /src/tensor/benchmarks/layer/
COPY ops_transformer/transformer_ops_ynnpack.h ops_transformer/transformer_ops_ynnpack.cc /src/tensor/examples/ops/transformer/
COPY ops_transformer/BUILD.ynnpack.snippet /tmp/BUILD.ynnpack.snippet
COPY patches/ /tmp/patches/
RUN set -eu; \
    if ! grep -q 'name = "transformer_ops_ynnpack"' /src/tensor/examples/ops/transformer/BUILD; then \
      cat /tmp/BUILD.ynnpack.snippet >> /src/tensor/examples/ops/transformer/BUILD; \
    fi; \
    git apply /tmp/patches/litert-tensor-xnnpack-tile-broadcast.patch; \
    rm -rf /tmp/patches /tmp/BUILD.ynnpack.snippet

ENV CC=clang CXX=clang++
RUN mkdir -p /opt/litert-tensor/bin \
 && bazelisk test -c opt --test_output=errors \
      //tensor/benchmarks/layer:layer_spec_test \
      //tensor/benchmarks/layer:tracking_allocator_test \
      //tensor/benchmarks/layer:litert_tensor_backend_test \
      //tensor/benchmarks/layer:benchmark_runner_test \
      //tensor/benchmarks/layer:parity_test \
      //tensor/benchmarks/layer:suite_loader_test \
 && bazelisk build -c opt //tensor/benchmarks/layer:layer_bench \
 && cp -L bazel-bin/tensor/benchmarks/layer/layer_bench /opt/litert-tensor/bin/layer_bench \
 && ln -sfn layer_bench /opt/litert-tensor/bin/layer_bench_xnnpack \
 && ln -sfn layer_bench /opt/litert-tensor/bin/layer_bench_ynnpack

FROM ubuntu:24.04 AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates libgomp1 python3 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/litert-tensor /opt/litert-tensor
ENV PATH="/opt/litert-tensor/bin:${PATH}"
ENTRYPOINT ["/opt/litert-tensor/bin/layer_bench"]
