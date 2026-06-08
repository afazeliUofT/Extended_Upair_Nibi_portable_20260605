#!/usr/bin/env bash
# Safely clean repository mess without deleting active training checkpoints from disk.
# It untracks generated artifacts from Git and removes transient stress/smoke folders.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

echo "[CLEAN] Updating .gitignore"
python - <<'PY'
from pathlib import Path
p = Path(".gitignore")
s = p.read_text() if p.exists() else ""
items = [
    "TWC_plots_comprehensive/",
    "logs/",
    "optuna/",
    "_stress_*/",
    "_smoke_*/",
    "_isolated_eval_chunks*/",
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

echo "[CLEAN] Removing generated artifacts from Git tracking only; local files remain."
git rm -r --cached --ignore-unmatch TWC_plots_comprehensive logs optuna _stress_cdl_eval_memory _stress_cdl_eval_one_receiver _isolated_eval_chunks _isolated_eval_chunks_smoke 2>/dev/null || true
git rm --cached --ignore-unmatch *.out *.err *.log 2>/dev/null || true

echo "[CLEAN] Removing local transient stress/smoke/chunk folders only."
rm -rf _stress_cdl_eval_memory _stress_cdl_eval_one_receiver _isolated_eval_chunks_smoke
if [[ "${UPAIR_DELETE_ISOLATED_CHUNKS:-0}" == "1" ]]; then
  rm -rf _isolated_eval_chunks
fi
if [[ "${UPAIR_DELETE_SMOKE:-1}" == "1" ]]; then
  rm -rf _smoke_*
fi

echo "[CLEAN] Removing Python bytecode caches."
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

if [[ "${UPAIR_REMOVE_OLD_DIAGNOSTIC_SCRIPTS:-0}" == "1" ]]; then
  echo "[CLEAN] Removing old diagnostic scripts."
  rm -f \
    upair_overequalization_output.txt \
    upair_overequalization_probe.py \
    upair_overequalization_probe.sh \
    upair_slowpc_fix_validation_output.txt \
    upair_slowpc_fix_validation_probe.py \
    upair_slowpc_fix_validation_probe.sh \
    upair_probe_sionna_slow_gain_formula.sh \
    upair_probe_sionna_umi_large_scale_attrs.sh \
    upair_probe_umi_pc_fast_fading_contamination.sh \
    upair_probe_umi_pc_power_integrity.sh \
    upair_fix_umi_pc_epsilon.sh
fi

echo "[CLEAN] Done. Review with:"
echo "  git status --short"
echo "  bash upair_repo_cleanup_dryrun.sh"
