# AI Edge Bench - Level 2: Layer Tier Makefile

# Required for `bench`: machine_threading profile in catalog/hyperparams.json
# (or a custom results label when paired with MACHINE_PROFILE=).
MACHINE ?=
# hyperparams.json machine_threading profile (thread counts, Roofline ceilings).
# Defaults to MACHINE; set MACHINE_PROFILE=<profile> when MACHINE is a custom
# results label, or MACHINE_PROFILE= (empty) to run without a host profile.
MACHINE_PROFILE ?= $(MACHINE)
SUITE ?= suites/gemma4-e2b-cpu.json
THREADS ?=
SEQ_LEN ?=
BATCH_SIZE ?=
MIN_TIME ?= 0.2s
RESULTS_DIR ?= $(CURDIR)/results
LAYER_IMAGE ?= aeb-layer-litert-tensor

.DEFAULT_GOAL := help
.PHONY: help image bench validate test

help:
	@echo "AI Edge Bench - Layer Tier (Level 2)$(if $(strip $(MACHINE)), - MACHINE=$(strip $(MACHINE)),)"
	@echo
	@echo "  make image                     build standalone LiteRT Tensor API layer_bench Docker image"
	@echo "  make validate                  validate layer/catalog/*.json and layer/suites/*.json schemas & invariants"
	@echo "  make MACHINE=<m> bench         run compound subgraph benchmarks (requires layer_bench in PATH or AEB_LAYER_BIN)"
	@echo "  make test                      validate catalog + run layer runner, scout, Roofline, and LMLB parity unit tests"
	@echo
	@echo "  bench variables:"
	@echo "    MACHINE=<label>              required results label (results/<label>/...)"
	@echo "    MACHINE_PROFILE=<profile>    hyperparams.json machine_threading profile; defaults to MACHINE,"
	@echo "                                 MACHINE_PROFILE= (empty) runs without a host profile"
	@echo "    SUITE, THREADS, SEQ_LEN, BATCH_SIZE, MIN_TIME, RESULTS_DIR"

image:
	docker build -t $(LAYER_IMAGE) subgraphs/litert_tensor

validate:
	python3 scripts/layer_catalog.py validate

bench:
	$(if $(strip $(MACHINE)),,$(error MACHINE is required; pass MACHINE=<profile> (e.g. sapphire-rapids, epyc-7443p, apple-m5-max, dgx-spark) or MACHINE=<label> MACHINE_PROFILE=))
	AEB_MACHINE=$(strip $(MACHINE)) AEB_RESULTS_DIR=$(RESULTS_DIR) \
	  python3 scripts/run_layer_bench.py \
	  --framework litert-tensor --label xnnpack --backend xnnpack \
	  --suite-file $(SUITE) \
	  $(if $(strip $(MACHINE_PROFILE)),--machine-profile $(strip $(MACHINE_PROFILE)),) \
	  $(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
	  $(if $(SEQ_LEN),--seq-len $(SEQ_LEN),) \
	  $(if $(THREADS),--threads $(THREADS),) --min-time $(MIN_TIME) \
	  --dump-parity-dir $(RESULTS_DIR)/$(strip $(MACHINE))/_parity_ref
	AEB_MACHINE=$(strip $(MACHINE)) AEB_RESULTS_DIR=$(RESULTS_DIR) \
	  python3 scripts/run_layer_bench.py \
	  --framework litert-tensor --label ynnpack --backend ynnpack \
	  --suite-file $(SUITE) \
	  $(if $(strip $(MACHINE_PROFILE)),--machine-profile $(strip $(MACHINE_PROFILE)),) \
	  $(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
	  $(if $(SEQ_LEN),--seq-len $(SEQ_LEN),) \
	  $(if $(THREADS),--threads $(THREADS),) --min-time $(MIN_TIME) \
	  --compare-parity-dir $(RESULTS_DIR)/$(strip $(MACHINE))/_parity_ref

test: validate
	python3 -m unittest discover -s scripts -p 'test*.py'
