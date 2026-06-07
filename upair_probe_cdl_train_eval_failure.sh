#!/usr/bin/env bash
# Diagnose CDL-C train/eval failures, OOM symptoms, and memory controls.
# Run from /home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

echo "[PROBE] ROOT=${ROOT}"
echo "[PROBE] git head=$(git rev-parse --short HEAD 2>/dev/null || true)"

echo
echo "================================================================================"
echo "[1] Train/eval logs present"
find logs/train_eval -maxdepth 1 -type f -name 'train_eval_*.out' -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort || true

echo
echo "================================================================================"
echo "[2] Critical patterns in train/eval logs"
grep -RniE 'ResourceExhausted|OOM|out of memory|Allocator|BFC|bfc_allocator|failed|Traceback|Killed|CANCELLED|TIME LIMIT|signal|Evaluation stopped|Training stopped' logs/train_eval 2>/dev/null | \
  grep -v 'InUse at' | head -300 || true

echo
echo "================================================================================"
echo "[3] BFC allocator dump size"
for f in logs/train_eval/train_eval_*.out; do
  [[ -f "$f" ]] || continue
  n=$(grep -c 'bfc_allocator.cc:1095.*InUse at' "$f" || true)
  if [[ "$n" != "0" ]]; then
    echo "[BFC_DUMP] $f InUse-lines=$n"
    echo "  first:"
    grep -n 'bfc_allocator.cc:1095.*InUse at' "$f" | head -3 || true
    echo "  last:"
    grep -n 'bfc_allocator.cc:1095.*InUse at' "$f" | tail -3 || true
  fi
done

echo
echo "================================================================================"
echo "[4] Last non-BFC lines from each train/eval log"
for f in logs/train_eval/train_eval_*.out; do
  [[ -f "$f" ]] || continue
  echo
  echo "---- $f ----"
  grep -v 'bfc_allocator.cc:1095.*InUse at' "$f" | tail -120 || true
done

echo
echo "================================================================================"
echo "[5] Config memory/eval controls"
python - <<'PY'
from pathlib import Path
import json, yaml

cfg = yaml.safe_load(Path("configs/twc_comprehensive_mu32_base.yaml").read_text())
def get(path, default=None):
    node = cfg
    for p in path.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node

keys = [
    "system.batch_size_eval",
    "evaluation.logical_batch_size",
    "evaluation.receiver_microbatch_size",
    "evaluation.stream_eval_microbatches",
    "evaluation.compiled_receiver_error_counts",
    "evaluation.receiver_call_jit_compile",
    "evaluation.memory_cleanup_every_batches",
    "evaluation.memory_cleanup_every_microbatch",
    "evaluation.nmse_receivers",
    "evaluation.save_example_batch",
    "evaluation.per_receiver_stopping",
    "evaluation.target_bler_floor",
    "evaluation.max_num_batches_per_point",
    "evaluation.target_block_errors_per_receiver",
    "evaluation.stopping_receivers",
]
for k in keys:
    print(f"{k}: {get(k)}")

print("\n[training state summaries]")
for p in sorted(Path("TWC_plots_comprehensive/runs_rx16/seed7/1dmrs").glob("*/metrics/train_state.json")):
    data = json.loads(p.read_text())
    print(f"{p.parent.parent.name}: complete={data.get('training_complete')} latest={data.get('latest_step')}/{data.get('total_steps')} reason={data.get('save_reason')} best_val={data.get('best_val')}")

print("\n[evaluation state summaries]")
for p in sorted(Path("TWC_plots_comprehensive/eval_runs_rx16/seed7/1dmrs").glob("*/metrics/evaluation_state.json")):
    data = json.loads(p.read_text())
    print(f"{p.parent.parent.name}: complete={data.get('evaluation_complete')} current_ebno={data.get('current_ebno_db')} partial_batches={data.get('partial_batches_run')} active={data.get('active_receivers')} completed={data.get('completed_ebno_db')}")
    print(f"  current_frame_errors={data.get('current_frame_errors')}")
    print(f"  current_num_frames={data.get('current_num_frames')}")
PY

echo
echo "[PROBE] DONE"
