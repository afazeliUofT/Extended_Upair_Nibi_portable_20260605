#!/usr/bin/env bash
# Strengthen CDL-C final evaluation memory safety:
# - BLER-only eval (NMSE disabled)
# - disable compiled receiver error-count wrapper (prevents Sionna 2D-LMMSE graph/memory accumulation)
# - smaller receiver microbatch
# - explicit TF allocator/log env in generated Slurm job scripts
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ ! -f configs/twc_comprehensive_mu32_base.yaml || ! -f upair_submit_train_eval_all.sh ]]; then
  echo "[PATCH] Run from the CDL-C repo root." >&2
  exit 1
fi

python - <<'PY'
from pathlib import Path
import yaml

cfg_path = Path("configs/twc_comprehensive_mu32_base.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
ev = cfg.setdefault("evaluation", {})

changes = {}
def setv(k, v):
    old = ev.get(k, None)
    ev[k] = v
    changes[k] = (old, v)

setv("nmse_receivers", [])
setv("save_example_batch", False)
setv("stream_eval_microbatches", True)
setv("compiled_receiver_error_counts", False)
setv("receiver_call_jit_compile", False)
setv("receiver_microbatch_size", 8)
setv("memory_cleanup_every_batches", 1)
setv("memory_cleanup_every_microbatch", True)
setv("log_gpu_memory", True)
setv("log_latency", True)

# Keep logical batch at 64 for Monte Carlo throughput. _eval_cfg will set system.batch_size_eval to this.
ev["logical_batch_size"] = int(ev.get("logical_batch_size", 64))

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("[PATCH] Updated evaluation memory-safety settings:")
for k, (old, new) in changes.items():
    print(f"  {k}: {old!r} -> {new!r}")
PY

python - <<'PY'
from pathlib import Path

env_block = '''# Explicit TensorFlow evaluation-memory environment.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
'''

# Top-level wrapper env, so Slurm export=ALL also carries it.
for path in ["upair_submit_train_eval_all.sh", "upair_submit_lib.sh"]:
    p = Path(path)
    if not p.exists():
        continue
    s = p.read_text()
    if "TF_GPU_ALLOCATOR" not in s:
        if "set -euo pipefail\n" in s:
            s = s.replace("set -euo pipefail\n", "set -euo pipefail\n" + env_block, 1)
        else:
            s = env_block + "\n" + s
        p.write_text(s)
        print(f"[PATCH] Added TF env block to {path}")
    else:
        print(f"[PATCH] {path} already contains TF_GPU_ALLOCATOR")

# Also inject env into generated train/eval sbatch scripts explicitly.
p = Path("upair_submit_train_eval_all.sh")
s = p.read_text()
needle = 'cat >> "${jobfile}" <<SBATCH\nset -euo pipefail\n'
replacement = 'cat >> "${jobfile}" <<SBATCH\nset -euo pipefail\n' + env_block
if "Explicit TensorFlow evaluation-memory environment." not in s[s.find('cat >> "${jobfile}"'):]:
    if needle not in s:
        raise SystemExit("[PATCH] Could not locate jobfile template in upair_submit_train_eval_all.sh")
    s = s.replace(needle, replacement, 1)
    p.write_text(s)
    print("[PATCH] Injected TF env block inside generated train/eval sbatch files")
else:
    print("[PATCH] generated train/eval sbatch template already has explicit TF env block")
PY

echo "[PATCH] Done. Run:"
echo "  bash upair_probe_cdl_eval_memory_safe_ready.sh"
