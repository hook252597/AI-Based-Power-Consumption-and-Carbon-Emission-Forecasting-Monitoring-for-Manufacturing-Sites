"""AI-only comparison with explicit input/output paths; leaves V1 untouched."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
for key, folder in (("TMP", ROOT / "tmp"), ("TEMP", ROOT / "tmp"), ("MPLCONFIGDIR", ROOT / ".cache/matplotlib"), ("PYTHONPYCACHEPREFIX", ROOT / ".cache/pycache")):
    folder.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(folder)
sys.pycache_prefix = str(ROOT / ".cache/pycache")

from manufacturing_ai.forecast_v2 import run


def within_project(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise argparse.ArgumentTypeError("Generated output must remain inside this AI_project/package root")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare four AI forecast families using March only; April/May are retrospective.")
    parser.add_argument("--hourly", type=Path, required=True, help="Read-only complete hourly observed CSV")
    parser.add_argument("--reference", type=Path, required=True, help="Read-only May505-hour reference CSV; opened after model freeze")
    parser.add_argument("--output-dir", type=within_project, default=ROOT / "outputs/forecast_v2")
    parser.add_argument("--model-dir", type=within_project, default=ROOT / "models/forecast_v2")
    args = parser.parse_args()
    run(args.hourly, args.reference, args.output_dir, args.model_dir)
