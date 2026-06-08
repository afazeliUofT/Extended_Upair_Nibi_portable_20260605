#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
source "${ROOT}/upair_portable_env.sh"
upair_activate

INPUT_ROOT="${UPAIR_EVAL_CHUNK_ROOT:-${ROOT}/_isolated_eval_chunks}"
OUTPUT_CSV="${UPAIR_EVAL_MERGED_CSV:-${INPUT_ROOT}/merged_curves.csv}"

python "${ROOT}/scripts/merge_isolated_eval_chunks.py" \
  --input-root "${INPUT_ROOT}" \
  --output-csv "${OUTPUT_CSV}" \
  ${UPAIR_EVAL_MERGE_FILTERS:-}
