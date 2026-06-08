#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

export UPAIR_STRESS_RECEIVER="${UPAIR_STRESS_RECEIVER:-baseline_ls_2dlmmse_lmmse}"
export UPAIR_STRESS_BATCHES="${UPAIR_STRESS_BATCHES:-40}"
export UPAIR_STRESS_MICRO="${UPAIR_STRESS_MICRO:-8}"
export UPAIR_OPTUNA_STAGEB_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB}"

python - <<'PY'
from pathlib import Path
import os

from upair5g.config import load_config, set_cfg, get_cfg
from upair5g.evaluation import evaluate_model
from scripts.run_comprehensive_mu32_ablation import (
    _variant_cfg,
    _apply_optuna_best_1dmrs,
    _eval_cfg,
)

RECEIVER = os.environ["UPAIR_STRESS_RECEIVER"]
BATCHES = int(os.environ["UPAIR_STRESS_BATCHES"])
MICRO = int(os.environ["UPAIR_STRESS_MICRO"])
B_PREFIX = os.environ["UPAIR_OPTUNA_STAGEB_PREFIX"]

cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")
variant = "main_d256_b4_r2"
dmrs_case = "1dmrs"
seed = 7

train_cfg = _variant_cfg(cfg, variant, dmrs_case, seed)
_apply_optuna_best_1dmrs(
    train_cfg,
    variant,
    dmrs_case,
    storage_dir="optuna",
    study_prefix=B_PREFIX,
    require_external=True,
)

ckpt = Path(
    "TWC_plots_comprehensive/runs_rx16/seed7/1dmrs/"
    "main_d256_b4_r2/checkpoints/best.weights.h5"
)
if not ckpt.exists():
    raise SystemExit(f"[STRESS-ONE] missing checkpoint: {ckpt}")

cfg_eval = _eval_cfg(train_cfg, variant, dmrs_case, 1)

set_cfg(cfg_eval, "experiment.output_root", "_stress_cdl_eval_one_receiver")
set_cfg(cfg_eval, "experiment.name", f"{variant}_u1_{RECEIVER}_m{MICRO}_b{BATCHES}")
set_cfg(cfg_eval, "system.ebno_db_eval", [-4.0])

set_cfg(cfg_eval, "baselines.enabled_receivers", [RECEIVER])

set_cfg(cfg_eval, "evaluation.nmse_receivers", [])
set_cfg(cfg_eval, "evaluation.save_example_batch", False)
set_cfg(cfg_eval, "evaluation.compiled_receiver_error_counts", False)
set_cfg(cfg_eval, "evaluation.receiver_call_jit_compile", False)
set_cfg(cfg_eval, "evaluation.receiver_microbatch_size", MICRO)
set_cfg(cfg_eval, "evaluation.stream_eval_microbatches", True)
set_cfg(cfg_eval, "evaluation.memory_cleanup_every_batches", 1)
set_cfg(cfg_eval, "evaluation.memory_cleanup_every_microbatch", True)

set_cfg(cfg_eval, "evaluation.min_num_batches_per_point", BATCHES)
set_cfg(cfg_eval, "evaluation.max_num_batches_per_point", BATCHES)
set_cfg(cfg_eval, "evaluation.target_block_errors_per_receiver", 0)
set_cfg(cfg_eval, "evaluation.per_receiver_stopping", False)
set_cfg(cfg_eval, "evaluation.force", True)
set_cfg(cfg_eval, "evaluation.progress_every_batches", 5)

set_cfg(cfg_eval, "baselines.covariance_estimation.reuse_cache", True)

print("[STRESS-ONE] receiver:", RECEIVER)
print("[STRESS-ONE] checkpoint:", ckpt)
print("[STRESS-ONE] batches:", BATCHES, "micro:", MICRO)
print("[STRESS-ONE] enabled_receivers:", get_cfg(cfg_eval, "baselines.enabled_receivers"))
print("[STRESS-ONE] compiled_receiver_error_counts:", get_cfg(cfg_eval, "evaluation.compiled_receiver_error_counts"))
print("[STRESS-ONE] nmse_receivers:", get_cfg(cfg_eval, "evaluation.nmse_receivers"))

result = evaluate_model(cfg_eval, checkpoint_path=str(ckpt), num_users=1)
print("[STRESS-ONE] result:", result)
PY
