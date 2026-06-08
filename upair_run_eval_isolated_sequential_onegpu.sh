#!/usr/bin/env bash
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

CONFIG="${UPAIR_CONFIG:-${ROOT}/configs/twc_comprehensive_mu32_base.yaml}"
VARIANTS_RAW="${UPAIR_EVAL_VARIANTS:-main_d256_b4_r2}"
RECEIVERS_RAW="${UPAIR_EVAL_RECEIVERS:-upair5g_lmmse}"
USERS_RAW="${UPAIR_EVAL_USERS:-1}"
EBNOS_RAW="${UPAIR_EVAL_EBNOS:--4}"
CHUNKS_RAW="${UPAIR_EVAL_CHUNKS:-0,1,2,3,4,5,6,7}"
CHUNK_BATCHES="${UPAIR_EVAL_CHUNK_BATCHES:-20}"
MICRO="${UPAIR_EVAL_MICRO:-8}"
DMRS_CASE="${UPAIR_DMRS_CASE:-1dmrs}"
SEED="${UPAIR_SEED:-7}"
STAGEB_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB}"
OUT_ROOT="${UPAIR_EVAL_CHUNK_ROOT:-${ROOT}/_isolated_eval_chunks}"

mkdir -p "${OUT_ROOT}"

split_csv() {
  local raw="${1//,/ }"
  # shellcheck disable=SC2206
  local arr=( ${raw} )
  printf '%s\n' "${arr[@]}"
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

          echo
          echo "================================================================================"
          echo "[SEQ-ISO] variant=${variant} receiver=${receiver} U=${users} Eb/N0=${ebno} chunk=${chunk}"
          echo "================================================================================"

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

        done < <(split_csv "${CHUNKS_RAW}")

        safe_ebno="${ebno//-/m}"
        safe_ebno="${safe_ebno//./p}"
        out_csv="${OUT_ROOT}/merged_${variant}_u${users}_${receiver}_e${safe_ebno}.csv"

        python "${ROOT}/scripts/merge_isolated_eval_chunks.py" \
          --input-root "${OUT_ROOT}" \
          --output-csv "${out_csv}" \
          --variant "${variant}" \
          --receiver "${receiver}" \
          --num-users "${users}" \
          --ebno-db "${ebno}"

      done < <(split_csv "${EBNOS_RAW}")
    done < <(split_csv "${USERS_RAW}")
  done < <(split_csv "${RECEIVERS_RAW}")
done < <(split_csv "${VARIANTS_RAW}")

echo "[SEQ-ISO] Done."
