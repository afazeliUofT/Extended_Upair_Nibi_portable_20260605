#!/usr/bin/env bash
# Show generated/redundant artifacts that make the repo messy.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

echo "================================================================================"
echo "[DRYRUN] Git-tracked generated artifacts"
git ls-files | grep -E '^(TWC_plots_comprehensive/|logs/|optuna/|_stress_|_smoke_|_isolated_eval_chunks|.*\.out$|.*\.err$|.*\.log$|.*\.weights\.h5$|.*\.data-[0-9]+-of-[0-9]+$|.*\.index$)' || true

echo
echo "================================================================================"
echo "[DRYRUN] Top-level diagnostic/obsolete scripts"
ls -1 upair_* 2>/dev/null | grep -E '(overequalization|slowpc|umi_pc|nearfar|stress|probe_sionna|fix_umi|before_|stageA_shape|feature_rbs|exact_dmrs)' || true

echo
echo "================================================================================"
echo "[DRYRUN] Local transient folders"
find . -maxdepth 2 -type d \( -name '_stress*' -o -name '_smoke*' -o -name '_isolated_eval_chunks*' -o -name '__pycache__' \) -print || true

echo
echo "[DRYRUN] No files changed. Run upair_cleanup_repo_generated_safe.sh to untrack generated artifacts and remove transient stress/smoke folders."
