import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import json
import re
from pathlib import Path
from typing import Iterable, Dict, Any, Union, List
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"You are working on {device} device")
# GPU utilities
def print_peak_gpu_memory():
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
    print(f"Peak GPU memory used: {peak_memory:.2f} MB.")
    torch.cuda.reset_peak_memory_stats()

def print_gpu_utilization():
    nvmlInit()  # Initialize NVML
    handle = nvmlDeviceGetHandleByIndex(0)  # Assuming we're using GPU 0
    info = nvmlDeviceGetMemoryInfo(handle)  # Get memory info
    print(f"GPU memory occupied: {info.used // 1024**2} MB.")

def metrics_to_str(metrics):
    out = []
    for year, vals in metrics.items():
        rmse = vals["rmse"].item()
        mae  = vals["mae"].item()
        bias = vals["bias"].item()
        std  = vals["std"].item()
        n    = int(vals["area"].item())
        out.append(f"{year}: rmse={rmse:.2f}, mae={mae:.2f}, bias={bias:.2f}, std={std:.2f}, area in Km^2={n}")
    return " | ".join(out)




ITER_RE = re.compile(
    r"""^
        Iter\s*(?P<iter>\d+):\s*
        loss=(?P<loss>[-\d.]+),\s*
        metrics\s*=\s*(?P<metrics>.+)
        $
    """,
    re.VERBOSE,
)


GROUP_RE = re.compile(
    r"""
        ^\s*
        (?P<id>\d+):\s*
        rmse=(?P<rmse>[-\d.]+),\s*
        mae=(?P<mae>[-\d.]+),\s*
        bias=(?P<bias>[-\d.]+),\s*
        std=(?P<std>[-\d.]+),\s*
        area\sin\sKm\^2=(?P<area>[-\d.]+)
        \s*$
    """,
    re.VERBOSE,
)

def iterations_to_jsonl(
    lines: Iterable[str],
    args
):

    out_path = Path(os.path.join(args.outdir,"training_metrics.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parsed: List[Dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8") as f:
        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            m = ITER_RE.match(line)
            if not m:
                # Skip lines that don't match the expected pattern
                continue

            iter_num = int(m.group("iter"))
            loss_val = float(m.group("loss"))
            metrics_blob = m.group("metrics")

            # Split groups on ' | '
            groups = [g.strip() for g in metrics_blob.split("|") if g.strip()]

            metrics: Dict[str, Dict[str, float]] = {}
            for g in groups:
                gm = GROUP_RE.match(g)
                if not gm:
                    # Skip any malformed group without failing the whole line
                    continue
                gid = gm.group("id")  # keep as string to preserve leading zeros like "09"
                metrics[gid] = {
                    "rmse": float(gm.group("rmse")),
                    "mae": float(gm.group("mae")),
                    "bias": float(gm.group("bias")),
                    "std": float(gm.group("std")),
                    "area_km2": float(gm.group("area")),
                }

            record = {"iter": iter_num, "loss": loss_val, "metrics": metrics}
            parsed.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



