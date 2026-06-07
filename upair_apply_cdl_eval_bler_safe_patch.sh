#!/usr/bin/env bash
# Patch CDL-C config/wrappers for BLER-only, lower-noise, memory-safer evaluation.
# Run from /home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

python - <<'PY'
from pathlib import Path
import yaml

cfg_path = Path("configs/twc_comprehensive_mu32_base.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
ev = cfg.setdefault("evaluation", {})

old_nmse = ev.get("nmse_receivers", None)
ev["nmse_receivers"] = []                 # BLER/BER only; avoids explicit estimator NMSE calls in eval.
ev["save_example_batch"] = False          # avoid storing example tensors in final eval.
ev["stream_eval_microbatches"] = True
ev["receiver_microbatch_size"] = min(int(ev.get("receiver_microbatch_size", 16)), 16)
ev["compiled_receiver_error_counts"] = True
ev["receiver_call_jit_compile"] = False
ev["memory_cleanup_every_batches"] = 1
ev["memory_cleanup_every_microbatch"] = True
ev["log_gpu_memory"] = True
ev["log_latency"] = True

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[PATCH] evaluation.nmse_receivers: {old_nmse} -> []")
print("[PATCH] evaluation.save_example_batch=false; streaming/microbatch cleanup retained.")
PY

python - <<'PY'
from pathlib import Path

for path in ["upair_submit_train_eval_all.sh", "upair_submit_lib.sh"]:
    p = Path(path)
    if not p.exists():
        continue
    s = p.read_text()
    marker = "export TF_CPP_MIN_LOG_LEVEL"
    insert = '''
# Keep TensorFlow allocator dumps from flooding Slurm logs.
# Real errors still surface as Python exceptions/Tracebacks.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
'''
    if marker not in s:
        # Put after set -euo pipefail when possible.
        if "set -euo pipefail\n" in s:
            s = s.replace("set -euo pipefail\n", "set -euo pipefail\n" + insert, 1)
        else:
            s = insert + "\n" + s
        p.write_text(s)
        print(f"[PATCH] added TensorFlow log/memory env defaults to {path}")
    else:
        print(f"[PATCH] {path} already has TensorFlow log env defaults")
PY

echo "[PATCH] Done. Run:"
echo "  bash upair_probe_cdl_train_eval_failure.sh"
