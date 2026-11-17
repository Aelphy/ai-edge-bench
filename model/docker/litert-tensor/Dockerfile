# syntax=docker/dockerfile:1.7
#
# LiteRT Tensor API benchmark image (CPU: native XnnpackRunner & YnnpackRunner).
#
# Built with repo root (`..` from `model/`) as Docker build context so Level 2
# (`layer/subgraphs/litert_tensor/`) and Level 3 (`model/docker/litert-tensor/`)
# share a single source of truth without duplicating C++ files.

ARG AEB_BASE_IMAGE=aeb/base:latest
ARG BUILDER_BASE=ubuntu:24.04
ARG RUNTIME_BASE=ubuntu:24.04

FROM ${AEB_BASE_IMAGE} AS aebbase

# ---------------------------------------------------------------- build stage
FROM ${BUILDER_BASE} AS build

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc clang cmake make pkg-config \
      openjdk-17-jdk-headless \
      git git-lfs wget curl unzip tar ca-certificates \
      python3 python3-dev python3-pip python3-numpy \
      ccache zlib1g-dev libssl-dev libpng-dev libcurl4-openssl-dev \
      npm patchelf \
 && rm -rf /var/lib/apt/lists/*

RUN npm install -g @bazel/bazelisk

ARG LITERT_REPO=https://github.com/google-ai-edge/LiteRT
ARG LITERT_REF=5d9e2a42e02b6ea56f9fab06a94fa425b9facfa6

WORKDIR /src
RUN git init -q . \
 && git remote add origin "${LITERT_REPO}" \
 && git fetch --depth 1 origin "${LITERT_REF}" \
 && git checkout -q --detach FETCH_HEAD \
 && git rev-parse HEAD > /src/.litert-commit

# Overlay Level 2 `layer_ml_bench` + native `YnnpackMixinTag` transformer ops
# from `layer/subgraphs/litert_tensor/` and Level 3 Gemma 4 AEB driver + patches
# from `model/docker/litert-tensor/`.
COPY layer/subgraphs/litert_tensor/layer_ml_bench/ /src/tensor/benchmarks/layer/
COPY layer/subgraphs/litert_tensor/ops_transformer/transformer_ops_ynnpack.h layer/subgraphs/litert_tensor/ops_transformer/transformer_ops_ynnpack.cc /src/tensor/examples/ops/transformer/
COPY layer/subgraphs/litert_tensor/ops_transformer/BUILD.ynnpack.snippet /tmp/BUILD.ynnpack.snippet
COPY model/docker/litert-tensor/gemma4_driver/ /src/tensor/examples/gemma4/
COPY layer/subgraphs/litert_tensor/patches/ /tmp/patches/
RUN set -eu; \
    if ! grep -q 'name = "transformer_ops_ynnpack"' /src/tensor/examples/ops/transformer/BUILD; then \
      cat /tmp/BUILD.ynnpack.snippet >> /src/tensor/examples/ops/transformer/BUILD; \
    fi; \
    git apply /tmp/patches/litert-tensor-xnnpack-tile-broadcast.patch; \
    printf 'litert-tensor-xnnpack-tile-broadcast.patch\n' > /src/.litert-patches; \
    rm -rf /tmp/patches /tmp/BUILD.ynnpack.snippet

ARG BAZEL_CONFIG=linux_x86_64
ARG BAZEL_JOBS=0

ENV CC=clang CXX=clang++

# Single native dual-mixin build (`Tensor<XnnpackMixinTag, YnnpackMixinTag>`)
# supporting both `XnnpackRunner` and `YnnpackRunner` at runtime via `--backend`
# or `argv[0]` suffix (`_xnnpack` / `_ynnpack`).
RUN set -eu; \
    mkdir -p /opt/litert-tensor/bin; \
    for attempt in 1 2 3; do \
      if bazelisk build \
           -c opt \
           $( [ "${BAZEL_JOBS}" -gt 0 ] && echo "--jobs=${BAZEL_JOBS}" ) \
           //tensor/benchmarks/layer:layer_bench \
           //tensor/examples/gemma4:gemma4_tensor_driver \
           //tensor/examples/gemma4:gemma4_xnnpack_main; then \
        break; \
      fi; \
      [ "${attempt}" = 3 ] && { echo "bazel build litert-tensor failed" >&2; exit 1; }; \
      sleep 20; \
    done \
 && cp -L bazel-bin/tensor/benchmarks/layer/layer_bench /opt/litert-tensor/bin/layer_bench \
 && ln -sfn layer_bench /opt/litert-tensor/bin/layer_bench_xnnpack \
 && ln -sfn layer_bench /opt/litert-tensor/bin/layer_bench_ynnpack \
 && cp -L bazel-bin/tensor/examples/gemma4/gemma4_tensor_driver /opt/litert-tensor/bin/gemma4_tensor_driver \
 && ln -sfn gemma4_tensor_driver /opt/litert-tensor/bin/gemma4_tensor_driver_xnnpack \
 && ln -sfn gemma4_tensor_driver /opt/litert-tensor/bin/gemma4_tensor_driver_ynnpack \
 && cp -L bazel-bin/tensor/examples/gemma4/gemma4_xnnpack_main /opt/litert-tensor/bin/gemma4_xnnpack_main

# -------------------------------------------------------------- runtime stage
FROM ${RUNTIME_BASE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG EXTRA_RUNTIME_PACKAGES=""

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl jq file procps util-linux hostname tzdata \
      libgomp1 zlib1g python3 python3-pip \
      ${EXTRA_RUNTIME_PACKAGES} \
 && pip3 install --no-cache-dir --break-system-packages tokenizers \
 && rm -rf /var/lib/apt/lists/*

COPY --from=aebbase /opt/aeb /opt/aeb
COPY --from=build /opt/litert-tensor /opt/litert-tensor

ARG LITERT_REF
ARG BAZEL_CONFIG
COPY --from=build /src/.litert-commit /tmp/commit
COPY --from=build /src/.litert-patches /tmp/patches
RUN printf '{"framework":"litert-tensor","ref":"%s","commit":"%s","patches":"%s","bazel_config":"%s","binaries":"%s","built_at":"%s"}\n' \
      "${LITERT_REF}" "$(cat /tmp/commit)" "$(tr '\n' ' ' < /tmp/patches)" "${BAZEL_CONFIG}" \
      "$(ls /opt/litert-tensor/bin | tr '\n' ' ')" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > /opt/aeb/build-info.json \
 && rm -f /tmp/commit /tmp/patches

ENV PATH="/opt/aeb/bin:/opt/litert-tensor/bin:${PATH}" \
    PYTHONPATH="/opt/aeb/harness" \
    AEB_MODELS_DIR=/models \
    AEB_RESULTS_DIR=/results \
    TZ=UTC

VOLUME ["/models", "/results"]
WORKDIR /work

ENTRYPOINT ["/opt/aeb/bin/aeb-entrypoint"]
CMD ["python3", "-m", "aeb.layer", "--help"]
