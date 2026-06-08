#!/usr/bin/env bash
# Submit process-isolated evaluation chunks.
# Use env vars to select scope, e.g.:
#   UPAIR_EVAL_VARIANTS=main_d256_b4_r2 UPAIR_EVAL_RECEIVERS=upair5g_lmmse UPAIR_EVAL_USERS=1 UPAIR_EVAL_EBNOS=-4 UPAIR_EVAL_CHUNKS=0,1,2 bash upair_submit_eval_isolated_chunks.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_submit_lib.sh"
upair_ensure_venv

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

CONFIG="${UPAIR_CONFIG:-${ROOT}/configs/twc_comprehensive_mu32_base.yaml}"
STAGEB_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB}"
DMRS_CASE="${UPAIR_DMRS_CASE:-1dmrs}"
SEED="${UPAIR_SEED:-7}"
VARIANTS_RAW="${UPAIR_EVAL_VARIANTS:-main_d256_b4_r2}"
RECEIVERS_RAW="${UPAIR_EVAL_RECEIVERS:-baseline_ls_lmmse,baseline_ls_2dlmmse_lmmse,upair5g_lmmse,perfect_csi_lmmse}"
USERS_RAW="${UPAIR_EVAL_USERS:-1,2,3,4}"
EBNOS_RAW="${UPAIR_EVAL_EBNOS:--4,-3,-2,-1,0,1,2,3,4}"
CHUNKS_RAW="${UPAIR_EVAL_CHUNKS:-0}"
CHUNK_BATCHES="${UPAIR_EVAL_CHUNK_BATCHES:-20}"
MICRO="${UPAIR_EVAL_MICRO:-8}"
TIME_LIMIT="${UPAIR_TIME_EVAL_CHUNK:-02:00:00}"
OUT_ROOT="${UPAIR_EVAL_CHUNK_ROOT:-${ROOT}/_isolated_eval_chunks}"

mkdir -p "${ROOT}/logs/eval_iso" "${ROOT}/logs/submit" "${OUT_ROOT}"

split_csv() {
  local raw="${1//,/ }"
  # shellcheck disable=SC2206
  local arr=( ${raw} )
  printf '%s\n' "${arr[@]}"
}

submit_one() {
  local variant="$1" receiver="$2" users="$3" ebno="$4" chunk="$5"
  local safe_ebno="${ebno//- /m}"
  safe_ebno="${ebno//-/m}"
  safe_ebno="${safe_ebno//./p}"
  local job="uIso-$(upair_first_n_chars "${variant}" 8)-u${users}-${receiver:0:8}-e${safe_ebno}-c${chunk}"
  local log="${ROOT}/logs/eval_iso/eval_${variant}_u${users}_${receiver}_e${safe_ebno}_c${chunk}_%j.out"
  local jobfile="${ROOT}/logs/submit/eval_iso_${variant}_u${users}_${receiver}_e${safe_ebno}_c${chunk}.sbatch"
  upair_write_sbatch_header "${jobfile}" "${job}" "${TIME_LIMIT}" "${log}"
  cat >> "${jobfile}" <<SBATCH
set -euo pipefail
cd "${ROOT}"
source "${ROOT}/upair_portable_env.sh"
upair_activate
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR}"
python -u "${ROOT}/scripts/run_isolated_eval_chunk.py" \
  --config "${CONFIG}" \
  --variant "${variant}" \
  --dmrs-case "${DMRS_CASE}" \
  --seed "${SEED}" \
  --num-users "${users}" \
  --receiver "${receiver}" \
  --ebno-db "${ebno}" \
  --chunk-idx "${chunk}" \
  --chunk-batches "${CHUNK_BATCHES}" \
  --receiver-microbatch-size "${MICRO}" \
  --stageb-prefix "${STAGEB_PREFIX}" \
  --optuna-dir "${ROOT}/optuna" \
  --output-root "${OUT_ROOT}"
SBATCH
  echo "[ISO-SUBMIT] ${variant} u=${users} receiver=${receiver} ebno=${ebno} chunk=${chunk}"
  upair_submit_job_script "${jobfile}"
}

while IFS= read -r variant; do
  [[ -n "${variant}" ]] || continue
  while IFS= read -r receiver; do
    [[ -n "${receiver}" ]] || continue
    while IFS= read -r users; do
      [[ -n "${users}" ]] || continue
      while IFS= read -r ebno; do
        [[ -n "${ebno}" ]] || continue
        while IFS= read -r chunk; do
          [[ -n "${chunk}" ]] || continue
          submit_one "${variant}" "${receiver}" "${users}" "${ebno}" "${chunk}"
        done < <(split_csv "${CHUNKS_RAW}")
      done < <(split_csv "${EBNOS_RAW}")
    done < <(split_csv "${USERS_RAW}")
  done < <(split_csv "${RECEIVERS_RAW}")
done < <(split_csv "${VARIANTS_RAW}")
