"""Build portable exact alert intervals from a verified five-second cache.

No model training or inference is performed. Every cached day is compared with
current dashboard hourly counts, sensor minima and frozen AI scores. Example:
    python tools/prepare_alert_events.py --source-cache ../outputs/agent_data
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from monitoring.alert_service import (DEFAULT_THRESHOLDS, ITEMS, MODULE, SAMPLE_SECONDS,
                                      build_events, sha256)


def _inside(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("캐시 파일 경로가 지정된 캐시 폴더 밖을 가리킵니다.")
    return result


def _check_day(frame: pd.DataFrame, hourly: pd.DataFrame, day: str) -> dict:
    raw = frame.copy()
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    if not raw["datetime"].dt.strftime("%Y-%m-%d").eq(day).all():
        raise ValueError(f"{day}: 다른 날짜의 기록이 섞여 있습니다.")
    raw["channel"] = raw[MODULE].astype(str).str.split("(", n=1).str[0]
    raw["hour"] = raw["datetime"].dt.floor("h")
    specs = {"n": ("ai_flag", "size"), "ai_flags": ("ai_flag", "sum"),
             "ai_score_max": ("ai_score", "max")}
    specs.update({col: (col, "min") for col, *_ in ITEMS.values()})
    observed = raw.groupby(["channel", "hour"]).agg(**specs).sort_index()
    target = hourly[hourly["datetime"].dt.strftime("%Y-%m-%d").eq(day)].copy()
    target["channel"] = target[MODULE].astype(str).str.split("(", n=1).str[0]
    target = target.set_index(["channel", "datetime"])[list(specs)].sort_index()
    target.index.names = observed.index.names
    if not observed.index.equals(target.index):
        raise ValueError(f"{day}: 설비·시간 목록이 현재 대시보드와 다릅니다.")
    for col in ("n", "ai_flags"):
        if not np.array_equal(observed[col].to_numpy(), target[col].to_numpy()):
            raise ValueError(f"{day}: {col} 값이 현재 대시보드와 다릅니다.")
    for col in list(specs)[2:]:
        if not np.allclose(observed[col], target[col], rtol=0, atol=1e-10, equal_nan=True):
            raise ValueError(f"{day}: {col} 값이 현재 대시보드와 다릅니다.")
    return {"rows": len(raw), "equipment_count": int(raw[MODULE].nunique()),
            "hourly_rows_verified": len(observed), "ai_flags": int(observed["ai_flags"].sum()),
            "max_ai_score_difference": float((observed["ai_score_max"] - target["ai_score_max"]).abs().max()),
            "per_equipment": {str(k): int(v) for k, v in raw.groupby(MODULE).size().items()}}


def build(source_cache: Path, output_dir: Path | None = None, hourly_path: Path | None = None) -> dict:
    source_cache = Path(source_cache).resolve()
    output_dir = Path(output_dir or REPO / "ml/outputs/agent_data").resolve()
    hourly_path = Path(hourly_path or output_dir / "dashboard_hourly.parquet").resolve()
    manifest_path = source_cache / "manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    signature = manifest.get("signature") or {}
    if not manifest.get("complete") or signature.get("sample_seconds") != SAMPLE_SECONDS:
        raise ValueError("완료·5초 표본 검증이 없는 캐시는 사용할 수 없습니다.")
    project = source_cache.parents[1]
    source = (project / manifest["source_file"]).resolve()
    model = (project / manifest["model_file"]).resolve()
    source_stat = source.stat()
    source_meta = {"size": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns}
    if source_stat.st_size != signature["source_metadata"]["size"]:
        raise ValueError("원본 파일 크기가 검증 캐시와 다릅니다.")
    # Original cache hashed the source in the same pass as its data. Re-hash only
    # when the source metadata changed; model and partition hashes are always checked.
    if source_meta != signature["source_metadata"] and sha256(source) != manifest["source_sha256"]:
        raise ValueError("원본 파일 SHA256이 검증 캐시와 다릅니다.")
    if sha256(model) != signature["model_sha256"]:
        raise ValueError("5초 캐시를 만든 모델 파일이 달라졌습니다.")
    hourly_hash = sha256(hourly_path)
    hourly = pd.read_parquet(hourly_path)
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    hourly_meta_path = hourly_path.with_suffix(".json")
    hourly_meta = json.loads(hourly_meta_path.read_text(encoding="utf-8-sig"))
    if manifest.get("ai_thresholds") != hourly_meta.get("ai_thresholds"):
        raise ValueError("5초 캐시와 대시보드의 AI 경고 기준이 다릅니다.")
    event_rows, dates = [], {}
    for day, info in sorted(manifest["dates"].items()):
        path = _inside(source_cache, info["file"])
        if not info.get("complete") or path.stat().st_size != info["size"] or sha256(path) != info["sha256"]:
            raise ValueError(f"{day}: 원본 5초 캐시의 파일 검증이 실패했습니다.")
        frame = pd.read_parquet(path)
        if len(frame) != info["rows"]:
            raise ValueError(f"{day}: 검증한 5초 기록 수와 다릅니다.")
        check = _check_day(frame, hourly, day)
        rows = build_events(frame, ai_thresholds=manifest["ai_thresholds"])
        if sum(e["flagged_records"] for e in rows if e["kind"] == "ai") != check["ai_flags"]:
            raise ValueError(f"{day}: AI 이벤트에 기록이 빠지거나 중복되었습니다.")
        event_rows.extend(rows)
        dates[day] = {**check, "source_partition_sha256": info["sha256"], "event_rows": len(rows)}
        print(f"verified {day}: {check['rows']:,} samples, {len(rows):,} intervals", flush=True)
    if sha256(manifest_path) != manifest_hash or sha256(hourly_path) != hourly_hash:
        raise ValueError("생성 중 원본 캐시 메타데이터 또는 시간별 요약이 변경되었습니다.")
    if source.stat().st_size != source_stat.st_size or source.stat().st_mtime_ns != source_stat.st_mtime_ns:
        raise ValueError("생성 중 원본 파일이 변경되었습니다.")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "alert_events.parquet"
    temporary = output_dir / "alert_events.parquet.partial"
    result = pd.DataFrame(event_rows)
    result.to_parquet(temporary, index=False)
    temporary.replace(target)
    metadata = {
        "format_version": 1, "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_seconds": SAMPLE_SECONDS, "event_rows": len(result), "events_sha256": sha256(target),
        "hourly_sha256": hourly_hash, "source_manifest_sha256": manifest_hash,
        "source_sha256": manifest["source_sha256"], "source_bytes": source_stat.st_size,
        "source_model_sha256": signature["model_sha256"], "dashboard_model_sha256": hourly_meta["model_sha256"],
        "model_candidate": manifest.get("candidate"), "ai_thresholds": manifest["ai_thresholds"],
        "rule_thresholds": DEFAULT_THRESHOLDS, "dates": dates,
        "model_comparison": "No inference or training performed. Every day was verified against current dashboard hourly AI counts, max scores, sample counts and all nine sensor minima.",
        "counting_note": "Intervals are per equipment and sensor, split at normal samples, missing 5-second samples and day boundaries. End timestamp is exclusive. Affected hourly bins are never used as duration.",
    }
    tmp_meta = output_dir / "alert_events.json.partial"
    tmp_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_meta.replace(output_dir / "alert_events.json")
    print(f"READY: {len(dates)} days, {len(result):,} verified intervals, {target.stat().st_size:,} bytes", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-cache", type=Path, required=True,
                        help="검증된 manifest.json과 daily 폴더를 포함한 경로")
    parser.add_argument("--output-dir", type=Path, default=REPO / "ml/outputs/agent_data")
    args = parser.parse_args()
    build(args.source_cache, args.output_dir)


if __name__ == "__main__":
    main()
