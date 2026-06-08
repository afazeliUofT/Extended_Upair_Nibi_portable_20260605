#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

fail=0
ok(){ echo "[OK] $*"; }
bad(){ echo "[FAIL] $*" >&2; fail=1; }

[[ -f scripts/run_isolated_eval_chunk.py ]] && ok "scripts/run_isolated_eval_chunk.py exists" || bad "missing run_isolated_eval_chunk.py"
[[ -f scripts/merge_isolated_eval_chunks.py ]] && ok "scripts/merge_isolated_eval_chunks.py exists" || bad "missing merge_isolated_eval_chunks.py"
[[ -x upair_submit_eval_isolated_chunks.sh ]] && ok "upair_submit_eval_isolated_chunks.sh executable" || bad "submit wrapper missing/not executable"
[[ -x upair_merge_eval_isolated_chunks.sh ]] && ok "upair_merge_eval_isolated_chunks.sh executable" || bad "merge wrapper missing/not executable"

grep -q "compiled_receiver_error_counts.*False" scripts/run_isolated_eval_chunk.py && ok "chunk runner forces compiled_receiver_error_counts=False" || bad "chunk runner does not force compiled_receiver_error_counts=False"
grep -q "nmse_receivers.*\[\]" scripts/run_isolated_eval_chunk.py && ok "chunk runner forces nmse_receivers=[]" || bad "chunk runner does not force nmse_receivers=[]"
grep -q "chunk_seed" scripts/run_isolated_eval_chunk.py && ok "chunk runner changes evaluation seed per chunk" || bad "chunk runner does not set per-chunk seed"
grep -q "TF_GPU_ALLOCATOR" upair_submit_eval_isolated_chunks.sh && ok "isolated submit wrapper exports TF_GPU_ALLOCATOR" || bad "isolated submit wrapper missing TF_GPU_ALLOCATOR"

python - <<'PY'
from pathlib import Path
import ast
for path in ["scripts/run_isolated_eval_chunk.py", "scripts/merge_isolated_eval_chunks.py"]:
    ast.parse(Path(path).read_text())
    print(f"[OK] syntax parse: {path}")
PY

source "${ROOT}/upair_portable_env.sh"
upair_ensure_venv

if [[ -s optuna/clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB_main_d256_b4_r2_best_params.json || -s optuna/clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB_main_d256_b4_r2.db ]]; then
  ok "Stage-B best exists for main_d256_b4_r2"
else
  bad "Stage-B best missing for main_d256_b4_r2"
fi

if [[ -s TWC_plots_comprehensive/runs_rx16/seed7/1dmrs/main_d256_b4_r2/checkpoints/best.weights.h5 ]]; then
  ok "main_d256_b4_r2 checkpoint exists"
else
  bad "main_d256_b4_r2 checkpoint missing"
fi

[[ "$fail" == "0" ]] || exit 1
echo "[PROBE] PASSED isolated eval readiness probe"
