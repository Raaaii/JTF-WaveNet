# JTF-WaveNet Makefile
# Usage:
#   make help
#   make install
#   make train RUN=train/RUN_001 STAGE=stage1
#   make train RUN=train/RUN_001 STAGE=stage2
#   make eval STAGE=stage2 CKPT=checkpoints/stage2 BINNING=soft
#   make exp_preprocess RUN_DIR=exp_data_predic
#   make exp_predict RUN_DIR=exp_data_predic
#   make exp_view RUN_DIR=exp_data_predic

SHELL := /bin/bash

# ---- Python / venv ----
PY              ?= python3.10
VENV_DIR        ?= .venv
VENV_PY         := $(VENV_DIR)/bin/python
PIP             := $(VENV_DIR)/bin/pip

# ---- Project paths ----
RUN             ?= train/RUN_001
STAGE           ?= stage1
LOGDIR          ?= $(RUN)/outputs
CKPT            ?= checkpoints/stage2
BINNING         ?= soft
OUTDIR          ?= runs/$(shell date +%Y-%m-%d_%H-%M-%S)

# ---- Experimental pipeline ----
RUN_DIR         ?= exp_data_predic

.DEFAULT_GOAL := help

# -----------------------------
# Helpers
# -----------------------------
help:
	@echo ""
	@echo "JTF-WaveNet targets:"
	@echo "  make venv                     Create venv ($(VENV_DIR))"
	@echo "  make install                  pip install -e ."
	@echo "  make install-dev              pip install -e '.[dev]'"
	@echo "  make install-viz              pip install -e '.[viz]'"
	@echo "  make install-ml               pip install -e '.[ml]'"
	@echo "  make install-tracking         pip install -e '.[tracking]'"
	@echo "  make install-apps             pip install -e '.[apps]'"
	@echo "  make install-mac              pip install -e '.[mac]'"
	@echo ""
	@echo "Training:"
	@echo "  make init-run RUN=$(RUN)       Create outputs/checkpoints dirs under run"
	@echo "  make train RUN=$(RUN) STAGE=stage1   Run training stage1"
	@echo "  make train RUN=$(RUN) STAGE=stage2   Run training stage2"
	@echo "  make train-gpu RUN=$(RUN) STAGE=stage1 GPU=0   Pin to a GPU"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval STAGE=stage2 CKPT=$(CKPT) BINNING=$(BINNING)"
	@echo ""
	@echo "Experimental data pipeline (exp_data_predic):"
	@echo "  make exp_preprocess RUN_DIR=$(RUN_DIR)     Run run_preprocess.sh"
	@echo "  make exp_organize   RUN_DIR=$(RUN_DIR)     Run organize_fids.py"
	@echo "  make exp_predict    RUN_DIR=$(RUN_DIR)     Run predict_all.py"
	@echo "  make exp_view       RUN_DIR=$(RUN_DIR)     Run plot_viewer.py"
	@echo ""
	@echo "Dev:"
	@echo "  make format                   Run ruff format (if installed)"
	@echo "  make lint                     Run ruff check (if installed)"
	@echo "  make test                     Run pytest (if installed)"
	@echo "  make clean                    Remove caches + logs (safe)"
	@echo ""

# -----------------------------
# Environment
# -----------------------------
venv:
	@$(PY) -m venv $(VENV_DIR)
	@$(VENV_PY) -m pip install --upgrade pip setuptools wheel
	@echo "Venv created. Activate with: source $(VENV_DIR)/bin/activate"

install: venv
	@$(PIP) install -e .

install-dev: venv
	@$(PIP) install -e ".[dev]"

install-viz: venv
	@$(PIP) install -e ".[viz]"

install-ml: venv
	@$(PIP) install -e ".[ml]"

install-tracking: venv
	@$(PIP) install -e ".[tracking]"

install-apps: venv
	@$(PIP) install -e ".[apps]"

install-mac: venv
	@$(PIP) install -e ".[mac]"

# -----------------------------
# Training
# -----------------------------
init-run:
	@mkdir -p "$(RUN)/outputs"
	@mkdir -p "$(RUN)/outputs/checkpoints"
	@echo "Initialized run folder: $(RUN)"

train: init-run
	@echo "Running training: RUN=$(RUN) STAGE=$(STAGE)"
	@mkdir -p "$(RUN)/outputs"
	@cd "$(RUN)" && ../../$(VENV_PY) train.py --stage "$(STAGE)" 2>&1 | tee "outputs/$(STAGE).log"

# Same as train, but pin GPU explicitly: make train-gpu GPU=0
GPU ?=
train-gpu: init-run
	@echo "Running training (GPU pinned): GPU=$(GPU) RUN=$(RUN) STAGE=$(STAGE)"
	@mkdir -p "$(RUN)/outputs"
	@cd "$(RUN)" && CUDA_VISIBLE_DEVICES="$(GPU)" ../../$(VENV_PY) train.py --stage "$(STAGE)" 2>&1 | tee "outputs/$(STAGE)_gpu$(GPU).log"

# -----------------------------
# Evaluation (synthetic generator)
# -----------------------------
eval:
	@echo "Evaluating: STAGE=$(STAGE) CKPT=$(CKPT) BINNING=$(BINNING) RUNS=$(OUTDIR)"
	@mkdir -p "$(OUTDIR)"
	@$(VENV_PY) scripts/eval.py \
		--stage "$(STAGE)" \
		--binning "$(BINNING)" \
		--ckpt "$(CKPT)" \
		--runs "$(OUTDIR)" 2>&1 | tee "$(OUTDIR)/eval_$(STAGE).log"

# -----------------------------
# Experimental data pipeline
# -----------------------------
exp_preprocess:
	@echo "Running NMRPipe preprocessing in: $(RUN_DIR)"
	@cd "$(RUN_DIR)" && bash run_preprocess.sh 2>&1 | tee preprocess.log

exp_organize:
	@echo "Organizing phased FIDs in: $(RUN_DIR)"
	@cd "$(RUN_DIR)" && $(VENV_PY) organize_fids.py 2>&1 | tee organize_fids.log

exp_predict:
	@echo "Predicting experimental data in: $(RUN_DIR)"
	@cd "$(RUN_DIR)" && $(VENV_PY) predict_all.py 2>&1 | tee predict_all.log

exp_view:
	@echo "Launching interactive viewer in: $(RUN_DIR)"
	@cd "$(RUN_DIR)" && $(VENV_PY) plot_viewer.py

# -----------------------------
# Dev tools (optional)
# -----------------------------
format:
	@$(VENV_PY) -m ruff format .

lint:
	@$(VENV_PY) -m ruff check .

test:
	@$(VENV_PY) -m pytest -q

# -----------------------------
# Cleanup
# -----------------------------
clean:
	@echo "Removing caches and logs..."
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.log" -delete
	@rm -rf .pytest_cache .ruff_cache
	@echo "Done."
