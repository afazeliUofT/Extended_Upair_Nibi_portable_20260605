#!/usr/bin/env bash
# Smoke test for process-isolated UPAIR evaluation.
# It runs two independent chunks in two Python processes, then merges the results.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
source "${ROOT}/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

BATCHES="${UPAIR_SMOKE_ISO_BATCHES:-5}"
MICRO="${UPAIR_SMOKE_ISO_MICRO:-8}"
ROOT_OUT="${UPAIR_SMOKE_ISO_ROOT:-${ROOT}/_isolated_eval_chunks_smoke}"

rm -rf "${ROOT_OUT}"

for c in 0 1; do
  echo "================ isolated UPAIR chunk ${c} ================"
  python -u scripts/run_isolated_eval_chunk.py \
    --variant main_d256_b4_r2 \
    --dmrs-case 1dmrs \
    --seed 7 \
    --num-users 1 \
    --receiver upair5g_lmmse \
    --ebno-db -4 \
    --chunk-idx "${c}" \
    --chunk-batches "${BATCHES}" \
    --receiver-microbatch-size "${MICRO}" \
    --stageb-prefix clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB \
    --optuna-dir "${ROOT}/optuna" \
    --output-root "${ROOT_OUT}"
done

python scripts/merge_isolated_eval_chunks.py \
  --input-root "${ROOT_OUT}" \
  --output-csv "${ROOT_OUT}/merged_upair_smoke.csv" \
  --variant main_d256_b4_r2 \
  --receiver upair5g_lmmse \
  --num-users 1 \
  --ebno-db -4

echo "[SMOKE] PASSED process-isolated UPAIR eval smoke test"
