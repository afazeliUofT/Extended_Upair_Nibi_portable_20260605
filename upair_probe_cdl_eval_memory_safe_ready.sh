#!/usr/bin/env bash
# Verify CDL-C evaluation is configured for BLER-only memory-safe final eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

echo "[PROBE] ROOT=${ROOT}"
echo "[PROBE] git head=$(git rev-parse --short HEAD 2>/dev/null || true)"

python - <<'PY'
from pathlib import Path
import sys, yaml, json

fail = False
def ok(msg): print("[OK]", msg)
def bad(msg):
    global fail
    print("[FAIL]", msg, file=sys.stderr)
    fail = True

cfg = yaml.safe_load(Path("configs/twc_comprehensive_mu32_base.yaml").read_text())
ev = cfg.get("evaluation", {})

expected = {
    "nmse_receivers": [],
    "save_example_batch": False,
    "stream_eval_microbatches": True,
    "compiled_receiver_error_counts": False,
    "receiver_call_jit_compile": False,
    "receiver_microbatch_size": 8,
    "memory_cleanup_every_batches": 1,
    "memory_cleanup_every_microbatch": True,
}
for k, v in expected.items():
    if ev.get(k) == v:
        ok(f"evaluation.{k} = {v!r}")
    else:
        bad(f"evaluation.{k} expected {v!r}, got {ev.get(k)!r}")

if int(ev.get("logical_batch_size", 0)) == 64:
    ok("evaluation.logical_batch_size remains 64")
else:
    bad(f"evaluation.logical_batch_size expected 64, got {ev.get('logical_batch_size')}")

submit_lib = Path("upair_submit_lib.sh").read_text() if Path("upair_submit_lib.sh").exists() else ""
train_wrapper = Path("upair_submit_train_eval_all.sh").read_text()
for token in ["TF_CPP_MIN_LOG_LEVEL", "TF_CPP_VMODULE", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_GPU_ALLOCATOR", "cuda_malloc_async"]:
    if token in submit_lib or token in train_wrapper:
        ok(f"wrapper env contains {token}")
    else:
        bad(f"wrapper env missing {token}")

if 'cat >> "${jobfile}" <<SBATCH' in train_wrapper and "Explicit TensorFlow evaluation-memory environment" in train_wrapper:
    ok("generated train/eval sbatch template includes explicit TF env block")
else:
    bad("generated train/eval sbatch template does not include explicit TF env block")

print("\n[TRAINING STATES]")
for p in sorted(Path("TWC_plots_comprehensive/runs_rx16/seed7/1dmrs").glob("*/metrics/train_state.json")):
    data = json.loads(p.read_text())
    print(f"{p.parent.parent.name}: complete={data.get('training_complete')} latest={data.get('latest_step')}/{data.get('total_steps')} reason={data.get('save_reason')} best_val={data.get('best_val')}")

print("\n[EVAL STATES]")
for p in sorted(Path("TWC_plots_comprehensive/eval_runs_rx16/seed7/1dmrs").glob("*/metrics/evaluation_state.json")):
    data = json.loads(p.read_text())
    print(f"{p.parent.parent.name}: complete={data.get('evaluation_complete')} current_ebno={data.get('current_ebno_db')} partial_batches={data.get('partial_batches_run')} completed={data.get('completed_ebno_db')}")
    print(f"  eval_params.nmse_receivers={data.get('evaluation_parameters', {}).get('nmse_receivers')}")
    print(f"  eval_params.compiled_receiver_error_counts={data.get('evaluation_parameters', {}).get('compiled_receiver_error_counts')}")
    print(f"  eval_params.receiver_microbatch_size={data.get('evaluation_parameters', {}).get('receiver_microbatch_size')}")

if fail:
    raise SystemExit("[PROBE] FAILED CDL eval memory-safe ready probe")
print("[PROBE] PASSED CDL eval memory-safe ready probe")
PY
