"""Prepare local sensor cache only; no training, API requests or UI startup."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def _chunk_size(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chunksize must be an integer >= 1000") from exc
    if number < 1000:
        raise argparse.ArgumentTypeError("chunksize must be >= 1000")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare April 1–29 observations and frozen AI scores under an existing ML artifact root.")
    parser.add_argument("--project-root", type=Path, required=True, help="ML root containing src/, models/, and outputs/.")
    parser.add_argument("--source", type=Path, help="Read-only rtu_data_full.csv; recommended when moving to a new computer.")
    parser.add_argument("--chunksize", type=_chunk_size, default=250_000)
    parser.add_argument("--rebuild", action="store_true", help="Regenerate the cache after a deliberate source/model change.")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    cache = root / ".cache/agent_data"
    temp = cache / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ.update(TMP=str(temp), TEMP=str(temp), MPLCONFIGDIR=str(cache / "matplotlib"),
                      PYTHONPYCACHEPREFIX=str(cache / "pycache"), PYTHONIOENCODING="utf-8")
    tempfile.tempdir = str(temp)
    sys.pycache_prefix = str(cache / "pycache")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from .agent_data import prepare_data_cache
    manifest = prepare_data_cache(root, chunksize=args.chunksize, rebuild=args.rebuild,
                                  source_path=args.source)
    print(json.dumps({"complete": bool(manifest.get("complete")),
                      "available_dates": len(manifest.get("dates", {})),
                      "manifest": str(root / "outputs/agent_data/manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
