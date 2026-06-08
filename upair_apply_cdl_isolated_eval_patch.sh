#!/usr/bin/env bash
# Add process-isolated CDL evaluation utilities.
# Purpose: avoid GPU memory accumulation in upair5g_lmmse final evaluation by
# running each receiver/EbN0/user/chunk in a fresh Python process.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

[[ -f configs/twc_comprehensive_mu32_base.yaml && -d src/upair5g && -f scripts/run_comprehensive_mu32_ablation.py ]] || {
  echo "[ISO-EVAL-PATCH] Run from the CDL-C repo root." >&2
  exit 1
}

mkdir -p scripts

cat > scripts/run_isolated_eval_chunk.py <<'PY'
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from upair5g.config import get_cfg, load_config, set_cfg  # noqa: E402
from upair5g.evaluation import evaluate_model  # noqa: E402
from scripts.run_comprehensive_mu32_ablation import (  # noqa: E402
    _apply_optuna_best_1dmrs,
    _eval_cfg,
    _variant_cfg,
)


def _safe_tag(x: Any) -> str:
    s = str(x)
    return s.replace("-", "m").replace("+", "p").replace(".", "p").replace(",", "_")


def _read_single_row(curves_path: Path, receiver: str, ebno: float, num_users: int) -> dict[str, Any]:
    df = pd.read_csv(curves_path)
    if "receiver" in df.columns:
        df = df[df["receiver"].astype(str) == str(receiver)]
    if "ebno_db" in df.columns:
        df = df[df["ebno_db"].astype(float) == float(ebno)]
    if "num_users" in df.columns:
        df = df[df["num_users"].astype(int) == int(num_users)]
    if len(df) != 1:
        raise RuntimeError(
            f"Expected one curve row for receiver={receiver}, ebno={ebno}, num_users={num_users}, "
            f"got {len(df)} rows from {curves_path}"
        )
    return df.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated evaluation chunk.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dmrs-case", default="1dmrs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-users", type=int, required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--ebno-db", type=float, required=True)
    parser.add_argument("--chunk-idx", type=int, required=True)
    parser.add_argument("--chunk-batches", type=int, default=20)
    parser.add_argument("--receiver-microbatch-size", type=int, default=8)
    parser.add_argument("--stageb-prefix", default="clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB")
    parser.add_argument("--optuna-dir", default=str(PROJECT_ROOT / "optuna"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "_isolated_eval_chunks"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_CPP_VMODULE", "bfc_allocator=0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    cfg = load_config(args.config)
    train_cfg = _variant_cfg(cfg, args.variant, args.dmrs_case, args.seed)
    _apply_optuna_best_1dmrs(
        train_cfg,
        args.variant,
        args.dmrs_case,
        storage_dir=args.optuna_dir,
        study_prefix=args.stageb_prefix,
        require_external=True,
    )

    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        PROJECT_ROOT
        / "TWC_plots_comprehensive"
        / "runs_rx16"
        / f"seed{args.seed}"
        / args.dmrs_case
        / args.variant
        / "checkpoints"
        / str(get_cfg(train_cfg, "training.checkpoint_name", "best.weights.h5"))
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for isolated eval: {checkpoint}")

    cfg_eval = _eval_cfg(train_cfg, args.variant, args.dmrs_case, args.num_users)

    # Use a distinct deterministic evaluation seed per chunk so chunks are independent
    # rather than replaying the same Monte Carlo samples.
    base_eval_seed = int(get_cfg(cfg_eval, "system.evaluation_seed", args.seed + 1000))
    ebno_offset = int(round((float(args.ebno_db) + 100.0) * 1000.0))
    chunk_seed = base_eval_seed + 100003 * int(args.chunk_idx) + 17 * ebno_offset + 1009 * int(args.num_users)
    set_cfg(cfg_eval, "system.evaluation_seed", int(chunk_seed))
    set_cfg(cfg_eval, "system.seed", int(chunk_seed))

    tag = (
        f"{args.variant}_u{args.num_users}_{args.receiver}_"
        f"ebno{_safe_tag(args.ebno_db)}_chunk{args.chunk_idx:04d}_"
        f"m{args.receiver_microbatch_size}_b{args.chunk_batches}"
    )
    out_root = Path(args.output_root)
    set_cfg(cfg_eval, "experiment.output_root", str(out_root))
    set_cfg(cfg_eval, "experiment.name", tag)

    set_cfg(cfg_eval, "system.ebno_db_eval", [float(args.ebno_db)])
    set_cfg(cfg_eval, "baselines.enabled_receivers", [str(args.receiver)])

    # BLER-only, memory-safe, no long-lived compiled receiver-count graph.
    set_cfg(cfg_eval, "evaluation.nmse_receivers", [])
    set_cfg(cfg_eval, "evaluation.save_example_batch", False)
    set_cfg(cfg_eval, "evaluation.compiled_receiver_error_counts", False)
    set_cfg(cfg_eval, "evaluation.receiver_call_jit_compile", False)
    set_cfg(cfg_eval, "evaluation.receiver_microbatch_size", int(args.receiver_microbatch_size))
    set_cfg(cfg_eval, "evaluation.stream_eval_microbatches", True)
    set_cfg(cfg_eval, "evaluation.memory_cleanup_every_batches", 1)
    set_cfg(cfg_eval, "evaluation.memory_cleanup_every_microbatch", True)
    set_cfg(cfg_eval, "evaluation.min_num_batches_per_point", int(args.chunk_batches))
    set_cfg(cfg_eval, "evaluation.max_num_batches_per_point", int(args.chunk_batches))
    set_cfg(cfg_eval, "evaluation.target_block_errors_per_receiver", 0)
    set_cfg(cfg_eval, "evaluation.per_receiver_stopping", False)
    set_cfg(cfg_eval, "evaluation.force", True)
    set_cfg(cfg_eval, "evaluation.progress_every_batches", max(1, min(10, int(args.chunk_batches))))
    set_cfg(cfg_eval, "baselines.covariance_estimation.reuse_cache", True)

    print("[ISO-CHUNK] variant:", args.variant)
    print("[ISO-CHUNK] receiver:", args.receiver)
    print("[ISO-CHUNK] num_users:", args.num_users)
    print("[ISO-CHUNK] ebno_db:", args.ebno_db)
    print("[ISO-CHUNK] chunk_idx:", args.chunk_idx)
    print("[ISO-CHUNK] chunk_batches:", args.chunk_batches)
    print("[ISO-CHUNK] receiver_microbatch_size:", args.receiver_microbatch_size)
    print("[ISO-CHUNK] chunk_seed:", chunk_seed)
    print("[ISO-CHUNK] checkpoint:", checkpoint)
    print("[ISO-CHUNK] TF_GPU_ALLOCATOR:", os.environ.get("TF_GPU_ALLOCATOR"))
    print("[ISO-CHUNK] output tag:", tag)

    result = evaluate_model(cfg_eval, checkpoint_path=str(checkpoint), num_users=int(args.num_users))
    curves_path = Path(result["curves_path"])
    row = _read_single_row(curves_path, args.receiver, float(args.ebno_db), int(args.num_users))
    row.update(
        {
            "variant": args.variant,
            "dmrs_case": args.dmrs_case,
            "seed": int(args.seed),
            "training_seed": int(args.seed),
            "evaluation_seed": int(chunk_seed),
            "receiver": args.receiver,
            "num_users": int(args.num_users),
            "ebno_db": float(args.ebno_db),
            "chunk_idx": int(args.chunk_idx),
            "chunk_batches_requested": int(args.chunk_batches),
            "receiver_microbatch_size": int(args.receiver_microbatch_size),
            "checkpoint_path": str(checkpoint),
            "chunk_output_dir": str(result["output_dir"]),
            "chunk_curves_path": str(curves_path),
        }
    )

    out_dir = Path(result["output_dir"])
    pd.DataFrame([row]).to_csv(out_dir / "chunk_result.csv", index=False)
    with open(out_dir / "chunk_result.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, sort_keys=True)
    print("[ISO-CHUNK] wrote:", out_dir / "chunk_result.csv")
    print("[ISO-CHUNK] DONE")


if __name__ == "__main__":
    main()
PY

cat > scripts/merge_isolated_eval_chunks.py <<'PY'
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUM_COLS = [
    "bit_errors",
    "num_bits",
    "block_errors",
    "num_blocks",
    "num_batches_run",
    "point_elapsed_s",
    "data_elapsed_s",
    "receiver_elapsed_s",
]


def _load_rows(input_root: Path) -> pd.DataFrame:
    paths = sorted(input_root.rglob("chunk_result.csv"))
    if not paths:
        raise FileNotFoundError(f"No chunk_result.csv files found under {input_root}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["chunk_result_path"] = str(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _merge_group(group: pd.DataFrame) -> dict[str, Any]:
    first = group.iloc[0].to_dict()
    out: dict[str, Any] = {
        "receiver": str(first["receiver"]),
        "variant": str(first.get("variant", "")),
        "dmrs_case": str(first.get("dmrs_case", "")),
        "seed": int(first.get("seed", 7)),
        "training_seed": int(first.get("training_seed", first.get("seed", 7))),
        "num_users": int(first["num_users"]),
        "ebno_db": float(first["ebno_db"]),
        "num_chunks": int(len(group)),
        "chunk_indices": ",".join(str(int(x)) for x in sorted(group["chunk_idx"].astype(int).tolist())),
    }
    for col in SUM_COLS:
        if col in group.columns:
            out[col] = float(group[col].sum())
    # Count columns should be ints
    for col in ["bit_errors", "num_bits", "block_errors", "num_blocks", "num_batches_run"]:
        if col in out:
            out[col] = int(round(float(out[col])))

    bit_errors = int(out.get("bit_errors", 0))
    num_bits = int(out.get("num_bits", 0))
    block_errors = int(out.get("block_errors", 0))
    num_blocks = int(out.get("num_blocks", 0))
    rx_time = float(out.get("receiver_elapsed_s", 0.0))
    point_time = float(out.get("point_elapsed_s", 0.0))
    batches = int(out.get("num_batches_run", 0))

    out["ber"] = float(bit_errors / num_bits) if num_bits else np.nan
    out["bler"] = float(block_errors / num_blocks) if num_blocks else np.nan
    out["nmse"] = np.nan
    out["mc_stop_reason"] = "isolated_chunk_merge"
    out["target_bler_floor"] = np.nan
    out["bler_zero_error_upper_bound"] = float(3.0 / num_blocks) if num_blocks and block_errors == 0 else np.nan
    out["ber_zero_error_upper_bound"] = float(3.0 / num_bits) if num_bits and bit_errors == 0 else np.nan
    out["receiver_ms_per_batch"] = float(1000.0 * rx_time / max(batches, 1))
    out["receiver_ms_per_frame"] = float(1000.0 * rx_time / max(num_blocks, 1))
    out["reliable_ber"] = bool(bit_errors >= 1000)
    out["reliable_bler"] = bool(block_errors >= 100)
    # Peak memory is max over chunks; current memory is not meaningful after merge.
    for col in ["gpu_mem_current_gib", "gpu_mem_peak_gib", "gpu_mem", "peak"]:
        if col in group.columns:
            try:
                out[col] = float(group[col].max())
            except Exception:
                pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge process-isolated evaluation chunks.")
    parser.add_argument("--input-root", default="_isolated_eval_chunks")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--receiver", default=None)
    parser.add_argument("--num-users", type=int, default=None)
    parser.add_argument("--ebno-db", type=float, default=None)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    df = _load_rows(input_root)

    if args.variant:
        df = df[df["variant"].astype(str) == args.variant]
    if args.receiver:
        df = df[df["receiver"].astype(str) == args.receiver]
    if args.num_users is not None:
        df = df[df["num_users"].astype(int) == int(args.num_users)]
    if args.ebno_db is not None:
        df = df[np.isclose(df["ebno_db"].astype(float), float(args.ebno_db))]

    if df.empty:
        raise RuntimeError("No chunk rows remain after filters.")

    group_cols = ["variant", "dmrs_case", "seed", "receiver", "num_users", "ebno_db"]
    rows = [_merge_group(g) for _, g in df.groupby(group_cols, dropna=False)]
    out_df = pd.DataFrame(rows).sort_values(["variant", "num_users", "ebno_db", "receiver"])

    if args.output_csv:
        out_csv = Path(args.output_csv)
    else:
        out_csv = input_root / "merged_curves.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    summary = {
        "input_root": str(input_root),
        "output_csv": str(out_csv),
        "num_input_chunks": int(len(df)),
        "num_merged_rows": int(len(out_df)),
        "rows": rows,
    }
    summary_json = Path(args.summary_json) if args.summary_json else out_csv.with_suffix(".summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("[ISO-MERGE] chunks:", len(df))
    print("[ISO-MERGE] rows:", len(out_df))
    print("[ISO-MERGE] wrote:", out_csv)
    print("[ISO-MERGE] wrote:", summary_json)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
PY

cat > upair_submit_eval_isolated_chunks.sh <<'BASH'
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
BASH
chmod +x upair_submit_eval_isolated_chunks.sh

cat > upair_merge_eval_isolated_chunks.sh <<'BASH'
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
BASH
chmod +x upair_merge_eval_isolated_chunks.sh

cat > upair_probe_isolated_eval_ready.sh <<'BASH'
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
BASH
chmod +x upair_probe_isolated_eval_ready.sh

# Keep generated eval chunks out of git.
python - <<'PY'
from pathlib import Path
p = Path(".gitignore")
s = p.read_text() if p.exists() else ""
items = [
    "_isolated_eval_chunks/",
    "logs/eval_iso/",
    "_stress_cdl_eval_memory/",
    "_stress_cdl_eval_one_receiver/",
    "_smoke_*/",
    "__pycache__/",
    "*.py[cod]",
]
lines = s.splitlines()
for item in items:
    if item not in lines:
        lines.append(item)
p.write_text("\n".join(lines).rstrip() + "\n")
print("[ISO-EVAL-PATCH] .gitignore updated")
PY

echo "[ISO-EVAL-PATCH] Done. Run:"
echo "  bash upair_probe_isolated_eval_ready.sh"
echo "  bash upair_smoke_isolated_upair_eval.sh"
