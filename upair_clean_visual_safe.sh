#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

echo "================================================================================"
echo "[CLEAN] Before cleanup: top-level"
ls -1

echo
echo "================================================================================"
echo "[CLEAN] Preserving required runtime folders:"
echo "  - optuna/                                  required for Stage-B best params"
echo "  - TWC_plots_comprehensive/runs_rx16/       required for checkpoints/resume"
echo "  - src/, scripts/, configs/, wrappers       required code"

echo
echo "================================================================================"
echo "[CLEAN] Removing transient evaluation/smoke/stress folders"
rm -rf \
  _isolated_eval_chunks \
  _isolated_eval_chunks_smoke \
  _stress_cdl_eval_memory \
  _stress_cdl_eval_one_receiver \
  _smoke_* \
  logs/eval_iso

echo
echo "================================================================================"
echo "[CLEAN] Removing old failed automatic-eval outputs only"
rm -rf TWC_plots_comprehensive/eval_runs_rx16
rm -rf TWC_plots_comprehensive/csv_rx16

echo
echo "================================================================================"
echo "[CLEAN] Removing old huge runtime logs; keeping submit scripts folder structure"
rm -rf logs/train_eval
mkdir -p logs/submit logs/train_eval logs/eval_iso

echo
echo "================================================================================"
echo "[CLEAN] Removing Python bytecode/cache"
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

echo
echo "================================================================================"
echo "[CLEAN] Removing obsolete patch/probe artifacts"
rm -f \
  upair_apply_cdl_eval_bler_safe_patch.sh \
  upair_apply_cdl_eval_compiled_off_patch.sh \
  upair_probe_cdl_train_eval_failure.sh \
  upair_stress_cdl_eval_main_u1_memsafe.sh \
  upair_stress_cdl_eval_one_receiver.sh \
  upair_overequalization_output.txt \
  upair_overequalization_probe.py \
  upair_overequalization_probe.sh \
  upair_slowpc_fix_validation_output.txt \
  upair_slowpc_fix_validation_probe.py \
  upair_slowpc_fix_validation_probe.sh \
  *.sha256 \
  *.before_freeze_fix

echo
echo "================================================================================"
echo "[CLEAN] Updating .gitignore"
python - <<'PY'
from pathlib import Path

p = Path(".gitignore")
s = p.read_text() if p.exists() else ""
items = [
    "TWC_plots_comprehensive/",
    "logs/",
    "optuna/",
    "_isolated_eval_chunks*/",
    "_stress_*/",
    "_smoke_*/",
    "__pycache__/",
    "*.py[cod]",
    "*.out",
    "*.err",
    "*.log",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.weights.h5",
    "*.data-*",
    "*.index",
    "checkpoint",
]
lines = s.splitlines()
for item in items:
    if item not in lines:
        lines.append(item)
p.write_text("\n".join(lines).rstrip() + "\n")
PY

echo
echo "================================================================================"
echo "[CLEAN] Untracking generated runtime artifacts from Git only; local files remain"
git rm -r --cached --ignore-unmatch \
  TWC_plots_comprehensive \
  logs \
  optuna \
  _isolated_eval_chunks \
  _isolated_eval_chunks_smoke \
  _stress_cdl_eval_memory \
  _stress_cdl_eval_one_receiver \
  >/dev/null 2>&1 || true

echo
echo "================================================================================"
echo "[CLEAN] Final top-level view"
ls -1

echo
echo "================================================================================"
echo "[CLEAN] Required files check"
required=(
  configs
  src
  scripts
  pyproject.toml
  requirements.txt
  requirements-narval.txt
  upair_portable_env.sh
  upair_submit_lib.sh
  upair_submit_train_eval_all.sh
  upair_submit_eval_isolated_chunks.sh
  upair_run_eval_isolated_sequential_onegpu.sh
  upair_merge_eval_isolated_chunks.sh
  upair_probe_isolated_eval_ready.sh
)
for x in "${required[@]}"; do
  if [[ -e "$x" ]]; then
    echo "[OK] $x"
  else
    echo "[FAIL] missing $x" >&2
    exit 1
  fi
done

echo
echo "================================================================================"
echo "[CLEAN] Runtime required for resume/eval"
if [[ -d optuna ]]; then
  echo "[OK] optuna/ exists locally"
else
  echo "[WARN] optuna/ missing locally"
fi

if [[ -d TWC_plots_comprehensive/runs_rx16 ]]; then
  echo "[OK] training/checkpoint folder exists locally"
else
  echo "[WARN] TWC_plots_comprehensive/runs_rx16 missing locally"
fi

echo
echo "[CLEAN] Done. Review:"
echo "  git status --short"
