"""Read-only local April observations and separately labelled May predictions.

The cache stores fixed model scores. User thresholds affect rule queries only.
No credential or API-key configuration is read by this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import hashlib
import io
import json
import os
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

MODULE = "module(equipment)"
RAW_COLUMNS = [MODULE, "timestamp", "localtime", "operation", "voltageR", "voltageS", "voltageT", "voltageRS", "voltageST", "voltageTR", "currentR", "currentS", "currentT", "activePower", "powerFactorR", "powerFactorS", "powerFactorT", "reactivePowerLagging", "accumActiveEnergy"]
PHASE_ITEMS = ("voltageR", "voltageS", "voltageT")
LINE_ITEMS = ("voltageRS", "voltageST", "voltageTR")
PF_ITEMS = ("powerFactorR", "powerFactorS", "powerFactorT")
RULE_ITEMS = PHASE_ITEMS + LINE_ITEMS + PF_ITEMS
DEFAULT_THRESHOLDS = {"phase": 210.0, "line": 360.0, "pf": 85.0}
EXPECTED_PER_DAY = 17280
FIVE_SECONDS_NS = 5_000_000_000


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024*1024), b""):
            result.update(part)
    return result.hexdigest()


def _source_meta(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _source_reference(source, root):
    """Prefer a movable relative reference; different Windows drives stay absolute."""
    try:
        return Path(os.path.relpath(source, root)).as_posix()
    except ValueError:
        return Path(source).as_posix()


def _verify_source(source, manifest, force_hash=False):
    """Accept changed timestamps only after checking the original byte digest."""
    before = _source_meta(source)
    expected = manifest["signature"]["source_metadata"]
    if before["size"] != expected["size"]:
        raise ValueError("원본 데이터 크기가 변경되어 실측 캐시를 다시 생성해야 합니다.")
    if force_hash or before != expected:
        expected_hash = manifest.get("source_sha256")
        if not expected_hash or _sha256(source) != expected_hash:
            raise ValueError("원본 데이터 SHA256이 일치하지 않아 실측 캐시를 사용할 수 없습니다.")
        if _source_meta(source) != before:
            raise ValueError("원본 데이터가 무결성 확인 중 변경되었습니다.")
    return before


def _resolve_source(root, source_path, manifest=None):
    if source_path is not None:
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"원본 CSV 경로를 확인하세요: {source}")
        return source
    candidates = []
    if manifest and manifest.get("source_file"):
        candidates.append(root / manifest["source_file"])
    candidates.extend((root / "data/rtu_data_full.csv", root.parent / "data/rtu_data_full.csv",
                       root.parent / "프로젝트/data/rtu_data_full.csv"))
    for source in candidates:
        if source.is_file():
            return source.resolve()
    raise FileNotFoundError("rtu_data_full.csv를 찾지 못했습니다. --source 옵션으로 원본 경로를 지정하세요.")


def _write_manifest(path, manifest):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class _HashingRawReader(io.RawIOBase):
    """Hash source bytes while pandas consumes the same single file pass."""
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.handle.read(len(buffer))
        count = len(data)
        buffer[:count] = data
        self.digest.update(data)
        self.bytes_read += count
        return count


def prepare_data_cache(project_root, chunksize=250_000, rebuild=False, source_path=None):
    """Prepare April observations without training. Explicit sources may live elsewhere.

    Existing caches can be relinked to an identical CSV copy without rescoring.
    A changed timestamp or source path requires a full SHA256 byte comparison.
    """
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize < 1000:
        raise ValueError("chunksize는 1000 이상의 정수여야 합니다.")
    root = Path(project_root).resolve()
    model_path = root / "models/anomaly_v2/selected_model.joblib"
    destination = root / "outputs/agent_data"
    manifest_path = destination / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    source = _resolve_source(root, source_path, existing)
    signature = {"format_version": 1, "source_metadata": _source_meta(source), "model_sha256": _sha256(model_path), "period_start": "2025-04-01", "period_end": "2025-04-29", "sample_seconds": 5}
    if manifest_path.exists() and not rebuild:
        previous = existing.get("signature", {})
        if not existing.get("complete") or any(previous.get(key) != value for key, value in signature.items() if key != "source_metadata") or not existing.get("dates"):
            raise ValueError("Existing cache identity differs or is incomplete; rerun with --rebuild")
        changed_path = (root / existing.get("source_file", "")).resolve() != source
        metadata = _verify_source(source, existing, force_hash=changed_path)
        for item in existing["dates"].values():
            partition = (destination / item["file"]).resolve()
            if not partition.is_relative_to(destination) or not item.get("complete") or not partition.is_file() or partition.stat().st_size != item["size"] or _sha256(partition) != item["sha256"]:
                raise ValueError("Existing cache partition integrity differs; rerun with --rebuild")
        existing["source_file"] = _source_reference(source, root)
        existing["signature"]["source_metadata"] = metadata
        _write_manifest(manifest_path, existing)
        print(f"Verified cache identity; {len(existing['dates'])} April partitions already available", flush=True)
        return existing
    import pyarrow as pa
    import pyarrow.parquet as pq
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from manufacturing_ai.anomaly_v2 import load_selected, predict_observations
    destination.mkdir(parents=True, exist_ok=True)
    bundle = load_selected(model_path)
    modules = list(bundle["equipment"])
    directory_name = "daily" if not (destination / "daily").exists() else "daily_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    daily_dir = destination / directory_name
    daily_dir.mkdir()
    writers, counts = {}, {}
    numeric_types = {c: "float32" for c in RAW_COLUMNS if c not in (MODULE, "timestamp", "localtime", "operation", "activePower", "accumActiveEnergy")}
    dtypes = {MODULE: "string", "timestamp": "int64", "localtime": "int64", "operation": "int8", "activePower": "float64", "accumActiveEnergy": "float64", **numeric_types}
    seen, scored, start = 0, 0, time.monotonic()
    source_reader = None
    try:
        with source.open("rb") as original:
            source_reader = _HashingRawReader(original)
            with io.BufferedReader(source_reader, buffer_size=1024*1024) as stream:
                for index, raw in enumerate(pd.read_csv(stream, usecols=RAW_COLUMNS, dtype=dtypes, chunksize=chunksize)):
                    seen += len(raw)
                    selected = raw.loc[(raw.localtime >= 20250401000000) & (raw.localtime < 20250430000000)].copy()
                    if len(selected):
                        selected["datetime"] = pd.to_datetime(selected.localtime.astype(str), format="%Y%m%d%H%M%S")
                        inference = predict_observations(selected, bundle)
                        selected["ai_score"] = inference.anomaly_score.to_numpy()
                        selected["ai_flag"] = inference.is_anomaly.to_numpy(dtype=bool)
                        for date, frame in selected.groupby(selected.datetime.dt.strftime("%Y-%m-%d"), sort=False):
                            table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
                            if date not in writers:
                                writers[date] = pq.ParquetWriter(daily_dir / f"{date}.parquet.partial", table.schema, compression="zstd")
                                counts[date] = {}
                            writers[date].write_table(table)
                            for module, count in frame[MODULE].value_counts().items():
                                counts[date][str(module)] = counts[date].get(str(module), 0) + int(count)
                        scored += len(selected)
                    if index % 8 == 0:
                        print(f"Cache scanned {seen:,} source rows; scored {scored:,} April rows ({time.monotonic()-start:.1f}s)", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    if source_reader.bytes_read != signature["source_metadata"]["size"]:
        raise ValueError("Source stream did not cover every byte")
    if _source_meta(source) != signature["source_metadata"] or _sha256(model_path) != signature["model_sha256"]:
        raise ValueError("Source or model changed during cache construction")
    expected_dates = pd.date_range("2025-04-01", "2025-04-29", freq="D").strftime("%Y-%m-%d").tolist()
    if sorted(counts) != expected_dates or scored != 29*EXPECTED_PER_DAY*13:
        raise ValueError("April cache does not contain the required29 complete dates/all13 equipment")
    dates = {}
    for date in expected_dates:
        if set(counts[date]) != set(modules) or any(n != EXPECTED_PER_DAY for n in counts[date].values()):
            raise ValueError(f"Incomplete equipment/day: {date}")
        partial, final = daily_dir / f"{date}.parquet.partial", daily_dir / f"{date}.parquet"
        # Validate actual timestamp continuity before accepting an apparently
        # complete row count (duplicates/missing samples must not cancel out).
        keys = pd.read_parquet(partial, columns=[MODULE, "datetime"]).sort_values([MODULE, "datetime"])
        for module, frame in keys.groupby(MODULE):
            expected = pd.date_range(date, periods=EXPECTED_PER_DAY, freq="5s")
            if not pd.DatetimeIndex(frame.datetime).equals(expected):
                raise ValueError(f"Noncontinuous five-second day: {date}/{module}")
        partial.replace(final)
        dates[date] = {"file": f"{directory_name}/{date}.parquet", "rows": sum(counts[date].values()), "per_equipment": counts[date], "size": final.stat().st_size, "sha256": _sha256(final), "complete": True}
    manifest = {"complete": True, "signature": signature, "source_file": _source_reference(source, root), "source_sha256": source_reader.digest.hexdigest(), "source_bytes_hashed_in_same_pass": source_reader.bytes_read, "source_rows_scanned": seen, "model_file": "models/anomaly_v2/selected_model.joblib", "model": "IsolationForest", "candidate": bundle["candidate"], "equipment": modules, "ai_thresholds": {m: float(item["threshold"]) for m,item in bundle["equipment"].items()}, "dates": dates, "rows": scored, "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-start, "time_basis": "localtime; Asia/Seoul; source timestamp is retained but not used for grouping", "power_unit": "W", "energy_formula": "activePower(W)*5/3600000=kWh", "note": "AI scores and thresholds are frozen. User rule settings never modify them. Model limitations include missed unfamiliar single-sensor combinations."}
    _write_manifest(manifest_path, manifest)
    print(f"READY: {len(dates)} days, {scored:,} observed rows with fixed AI flags", flush=True)
    return manifest


def _token(value):
    return re.sub(r"[\s_()（）\-]+", "", unicodedata.normalize("NFKC", str(value))).lower()


def normalize_item(item):
    aliases = {_token(c): c for c in RAW_COLUMNS[4:]+["ai_score"]}
    for phase in "RST":
        aliases.update({_token(f"phase_voltage_{phase}"): f"voltage{phase}", _token(f"voltage_phase_{phase}"): f"voltage{phase}", _token(f"{phase}상전압"): f"voltage{phase}", _token(f"power_factor_{phase}"): f"powerFactor{phase}", _token(f"pf_{phase}"): f"powerFactor{phase}", _token(f"{phase}상역률"): f"powerFactor{phase}", _token(f"current_{phase}"): f"current{phase}"})
    for line in ("RS", "ST", "TR"):
        aliases.update({_token(f"line_voltage_{line}"): f"voltage{line}", _token(f"voltage_line_{line}"): f"voltage{line}", _token(f"{line}선간전압"): f"voltage{line}"})
    aliases.update({"power": "activePower", "activepowerw": "activePower", "유효전력": "activePower", "aiscore": "ai_score", "ai": "ai_score"})
    if _token(item) not in aliases:
        raise ValueError(f"지원하지 않는 측정 항목: {item}")
    return aliases[_token(item)]


def resolve_thresholds(thresholds=None):
    output = dict(DEFAULT_THRESHOLDS)
    aliases = {"phase": "phase", "phasevoltage": "phase", "phasevoltagemin": "phase", "voltagephase": "phase", "상전압": "phase", "line": "line", "linevoltage": "line", "linevoltagemin": "line", "voltageline": "line", "선간전압": "line", "pf": "pf", "powerfactor": "pf", "powerfactormin": "pf", "역률": "pf"}
    for key, value in (thresholds or {}).items():
        token = _token(key)
        if token in aliases:
            name = aliases[token]
        else:
            name = normalize_item(key)
            if name not in RULE_ITEMS:
                raise ValueError("규칙 임계값은 전압·역률 항목만 설정할 수 있습니다")
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise ValueError("임계값은 0보다 큰 유한한 수여야 합니다")
        output[name] = number
    return output


def _threshold(item, config):
    if item in config:
        return config[item]
    family = "phase" if item in PHASE_ITEMS else "line" if item in LINE_ITEMS else "pf" if item in PF_ITEMS else None
    return config[family] if family else None


def _date(value):
    # A month-only request must not silently become the first day's readings.
    if isinstance(value, str) and not re.match(r"^\d{4}-\d{1,2}-\d{1,2}(?:$|[ T])", value.strip()):
        return None
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        return stamp.strftime("%Y-%m-%d")
    except (ValueError, TypeError, OverflowError):
        return None


def _events(frame, equipment, item, flags, threshold, ai=False):
    """Group only adjacent observed 5-second anomaly rows, never bridge normals."""
    indices = np.flatnonzero(np.asarray(flags, dtype=bool))
    if not len(indices):
        return []
    times = frame.datetime.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    cuts = np.flatnonzero((np.diff(indices) != 1) | (np.diff(times[indices]) != FIVE_SECONDS_NS)) + 1
    values = frame["ai_score" if ai else item].to_numpy(dtype=float)
    result = []
    for group in np.split(indices, cuts):
        first, last = int(group[0]), int(group[-1])
        value = float(np.max(values[group]) if ai else np.min(values[group]))
        record = {"equipment": equipment, "item": "AI" if ai else item, "timestamp": pd.Timestamp(times[first]).isoformat(), "end_timestamp": (pd.Timestamp(times[last])+pd.Timedelta(seconds=5)).isoformat(), "last_observed_at": pd.Timestamp(times[last]).isoformat(), "value": value, "threshold": float(threshold), "deviation_pct": None if ai else float((threshold-value)/threshold*100), "duration_min": len(group)*5/60.0, "duration_seconds": len(group)*5, "record_count": len(group), "source": "AI" if ai else "rule", "unit": "anomaly_score" if ai else "%" if item in PF_ITEMS else "V"}
        if ai:
            record["note"] = "고정된 다변량 AI 점수의 이상 구간입니다. 특정 센서나 설비 고장이 원인이라고 단정하지 않습니다."
        result.append(record)
    return result


class DataService:
    def __init__(self, project_root: Path):
        self.root = Path(project_root).resolve()
        self.cache_dir = self.root / "outputs/agent_data"
        self.manifest = {}
        self._error = "4월 실측 캐시가 준비되지 않았습니다."
        path = self.cache_dir / "manifest.json"
        if path.exists():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if not manifest.get("complete"):
                    raise ValueError("캐시 생성이 아직 완료되지 않았습니다.")
                source = self.root / manifest["source_file"]
                model = self.root / manifest["model_file"]
                _verify_source(source, manifest)
                if _sha256(model) != manifest["signature"]["model_sha256"]:
                    raise ValueError("AI 모델이 변경되어 점수 캐시를 다시 생성해야 합니다.")
                self.manifest, self._error = manifest, None
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self._error = str(exc)
        self.modules = self.manifest.get("equipment", [])
        # Equipment registration can remain visible while observation cache is
        # absent, but availability/counts must remain false/null.
        if not self.modules:
            schema = self.root / "outputs/forecast_v2/model_schema.json"
            if schema.exists():
                try:
                    self.modules = json.loads(schema.read_text(encoding="utf-8")).get("selected", {}).get("modules", [])
                except (OSError, ValueError, TypeError):
                    self.modules = []

    @property
    def available_dates(self):
        return sorted(self.manifest.get("dates", {})) if self._error is None else []

    def get_available_dates(self):
        return self.available_dates

    def _provenance(self, date):
        date = _date(date)
        partition = self.manifest.get("dates", {}).get(date, {})
        identifier = f"observed:{date}:{partition['sha256'][:12]}" if partition else None
        return {"data_id": identifier, "source_id": identifier, "source": "RTU 5초 실측 캐시", "source_period": {"start": date+"T00:00:00" if date else None, "end": date+"T23:59:55" if date else None}, "timezone": "Asia/Seoul", "sample_seconds": 5}

    def normalize_equipment(self, equipment):
        if equipment is None or _token(equipment) in ("", "all", "전체", "모든설비", "공장전체", "factory"):
            return None
        value = _token(equipment)
        for module in self.modules:
            label = module.split("(", 1)[1][:-1] if "(" in module else module
            if value in (_token(module), _token(label)):
                return module
        channel = re.fullmatch(r"(?:ch|channel|채널|설비)?(\d+)(?:번|번설비|채널|번채널)?", value)
        if channel:
            for module in self.modules:
                if module.split("(", 1)[0] == channel.group(1):
                    return module
        raise ValueError(f"등록되지 않은 설비: {equipment}")

    @lru_cache(maxsize=2)
    def _read_day(self, date):
        if self._error:
            frame = pd.DataFrame()
            frame.attrs["reason"] = self._error
            return frame
        item = self.manifest.get("dates", {}).get(date)
        if item is None:
            frame = pd.DataFrame()
            frame.attrs["reason"] = "해당 날짜의 실측 데이터가 없습니다. 이용 가능 기간은 2025-04-01~2025-04-29입니다."
            return frame
        path = (self.cache_dir / item["file"]).resolve()
        try:
            if not path.is_relative_to(self.cache_dir) or path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
                raise ValueError("날짜 캐시 파일의 무결성 확인에 실패했습니다.")
            frame = pd.read_parquet(path).sort_values([MODULE, "datetime"]).reset_index(drop=True)
            if len(frame) != item["rows"] or not frame.datetime.dt.strftime("%Y-%m-%d").eq(date).all():
                raise ValueError("날짜 캐시의 레코드/날짜 검증에 실패했습니다.")
            frame.attrs["data_available"] = True
            return frame
        except (OSError, ValueError, KeyError) as exc:
            frame = pd.DataFrame()
            frame.attrs["reason"] = str(exc)
            return frame

    def daily_frame(self, date):
        valid_date = _date(date)
        if valid_date is None:
            frame = pd.DataFrame(columns=RAW_COLUMNS+["datetime", "ai_score", "ai_flag"])
            frame.attrs.update(data_available=False, reason="날짜 형식을 확인하세요.")
            return frame
        frame = self._read_day(valid_date).copy()
        frame.attrs["data_available"] = not frame.empty
        return frame

    def _select(self, date, equipment=None):
        frame = self.daily_frame(date)
        if frame.empty:
            return frame, None, frame.attrs.get("reason", "실측 데이터가 없습니다.")
        try:
            module = self.normalize_equipment(equipment)
        except ValueError as exc:
            return frame.iloc[:0], None, str(exc)
        if module is not None:
            frame = frame.loc[frame[MODULE] == module].copy()
        if frame.empty:
            return frame, module, "해당 설비의 실측 데이터가 없습니다."
        return frame, module, None

    def get_anomalies(self, date, equipment=None, thresholds=None):
        config = resolve_thresholds(thresholds)
        frame, module, reason = self._select(date, equipment)
        if reason:
            return {"date": _date(date), "equipment": module or equipment, "count": None, "items": [], "ai_count": None, "ai_items": [], "record_count": 0, "note": reason, "data_available": False, "thresholds": config}
        items, ai_items = [], []
        rule_flags, ai_flags = 0, 0
        for current, subset in frame.groupby(MODULE, sort=False):
            subset = subset.sort_values("datetime").reset_index(drop=True)
            for item in RULE_ITEMS:
                threshold = _threshold(item, config)
                flags = subset[item].notna().to_numpy() & (subset[item].to_numpy() < threshold)
                rule_flags += int(flags.sum())
                items.extend(_events(subset, str(current), item, flags, threshold))
            fixed_flags = subset.ai_flag.to_numpy(bool)
            ai_flags += int(fixed_flags.sum())
            ai_items.extend(_events(subset, str(current), "AI", fixed_flags, self.manifest["ai_thresholds"][str(current)], ai=True))
        items.sort(key=lambda e: (e["timestamp"], e["equipment"], e["item"]))
        ai_items.sort(key=lambda e: (e["timestamp"], e["equipment"]))
        expected = EXPECTED_PER_DAY * (1 if module else len(self.modules))
        return {**self._provenance(date), "date": _date(date), "equipment": module, "count": len(items), "items": items, "ai_count": len(ai_items), "ai_items": ai_items, "record_count": len(frame), "rule_flagged_sensor_records": rule_flags, "ai_flagged_records": ai_flags, "coverage": len(frame)/expected, "data_complete": len(frame) == expected, "data_available": True, "thresholds": config, "note": "연속된 5초 이상 기록을 설비·항목별 한 구간으로 묶었습니다. 정상 기록이나 누락 구간에서 분리하며, 규칙 임계값 변경은 AI 판정에 영향을 주지 않습니다. AI 결과는 실제 고장 원인을 확정하지 않습니다."}

    def get_equipment_list(self, date="2025-04-29", thresholds=None):
        anomalies = self.get_anomalies(date, thresholds=thresholds)
        day = self.daily_frame(date)
        counts = day[MODULE].value_counts().to_dict() if not day.empty else {}
        return [{"equipment": module, "label": module, "data_available": bool(anomalies["data_available"] and counts.get(module, 0)), "record_count": int(counts.get(module, 0)), "rule_event_count": sum(e["equipment"] == module for e in anomalies["items"]) if anomalies["data_available"] else None, "ai_event_count": sum(e["equipment"] == module for e in anomalies["ai_items"]) if anomalies["data_available"] else None, "note": None if anomalies["data_available"] else anomalies["note"]} for module in self.modules]

    def get_measurements(self, date, equipment, item, thresholds=None):
        frame, module, reason = self._select(date, equipment)
        try:
            canonical = normalize_item(item)
            config = resolve_thresholds(thresholds)
        except ValueError as exc:
            reason, canonical, config = str(exc), str(item), DEFAULT_THRESHOLDS
        if reason:
            return {"date": _date(date), "equipment": module or equipment, "item": canonical, "hours": [], "mean": [], "min": [], "max": [], "series": [], "coverage": 0.0, "data_available": False, "note": reason}
        count_equipment = 1 if module else len(self.modules)
        grouped = frame.groupby(frame.datetime.dt.floor("h"))[canonical].agg(["mean", "min", "max", "count"])
        hours = pd.date_range(_date(date), periods=24, freq="h")
        series = []
        for hour in hours:
            row = grouped.loc[hour] if hour in grouped.index else None
            count = int(row["count"]) if row is not None else 0
            series.append({"timestamp": hour.isoformat(), "mean": float(row["mean"]) if count else None, "min": float(row["min"]) if count else None, "max": float(row["max"]) if count else None, "record_count": count, "coverage": count/(720*count_equipment)})
        unit = "%" if canonical in PF_ITEMS else "V" if canonical.startswith("voltage") else "A" if canonical.startswith("current") else "W" if canonical == "activePower" else "anomaly_score" if canonical == "ai_score" else "raw_source_unit"
        threshold = self.manifest["ai_thresholds"].get(module) if canonical == "ai_score" else _threshold(canonical, config)
        raw_min_index = frame[canonical].idxmin()
        min_hour = grouped["mean"].idxmin()
        representative = {"minimum_observed": float(frame.loc[raw_min_index, canonical]), "minimum_observed_timestamp": frame.loc[raw_min_index, "datetime"].isoformat(), "minimum_hourly_mean": float(grouped.loc[min_hour, "mean"]), "minimum_hour": min_hour.isoformat(), "maximum_observed": float(frame[canonical].max()), "basis": "minimum_observed is a five-second measurement; minimum_hourly_mean is an hourly mean. These are not interchangeable."}
        return {**self._provenance(date), "date": _date(date), "equipment": module, "item": canonical, "unit": unit, "threshold": threshold, "hours": [s["timestamp"] for s in series], "mean": [s["mean"] for s in series], "min": [s["min"] for s in series], "max": [s["max"] for s in series], "series": series, "summary": representative, "representative_value": representative["minimum_observed"], "representative_timestamp": representative["minimum_observed_timestamp"], "representative_basis": "관측된 5초 측정값 중 일 최저값", "record_count": len(frame), "coverage": sum(s["record_count"] for s in series)/(EXPECTED_PER_DAY*count_equipment), "sample_seconds": 5, "aggregation": "hourly mean/min/max of observed five-second samples", "data_available": True, "note": "시간별 평균·최솟값·최댓값입니다. 관측이 없는 시간은 0으로 채우지 않습니다."}

    def get_measurements_group(self, date, equipment=None, item="phase_voltage", thresholds=None):
        """Read one day once, preserving separate extrema for each machine/sensor."""
        groups = {"phase_voltage": PHASE_ITEMS, "line_voltage": LINE_ITEMS, "power_factor": PF_ITEMS}
        api_names = {**{c: f"phase_voltage_{c[-1]}" for c in PHASE_ITEMS},
                     **{c: f"line_voltage_{c[-2:]}" for c in LINE_ITEMS},
                     **{c: f"power_factor_{c[-1]}" for c in PF_ITEMS}}
        latest = self.available_dates[-1] if self.available_dates else None
        result = {"date": _date(date), "equipment": equipment, "item": item, "grouped": True,
                  "items": [], "data_available": False, "latest_available_date": latest,
                  "equipment_count": 0, "record_count": 0}
        try:
            columns = groups[item] if item in groups else (normalize_item(item),)
            if any(column not in RULE_ITEMS for column in columns):
                raise ValueError("묶음 조회는 전압·역률 항목만 지원합니다.")
            module = self.normalize_equipment(equipment)
            config = resolve_thresholds(thresholds)
        except ValueError as exc:
            return {**result, "note": str(exc)}
        selected = [module] if module else self.modules
        result.update(equipment=module, equipment_count=len(selected))
        frame = self.daily_frame(date)
        if frame.empty:
            return {**result, "note": frame.attrs.get("reason", "해당 날짜의 실측 데이터가 없습니다.")}
        per_equipment = {str(name): subset for name, subset in frame.groupby(MODULE, sort=False)}
        rows = []
        for name in selected:
            subset = per_equipment.get(name, frame.iloc[:0])
            readings = []
            for column in columns:
                numeric = pd.to_numeric(subset[column], errors="coerce")
                valid = numeric.loc[np.isfinite(numeric)]
                count = len(valid)
                minimum_index = valid.idxmin() if count else None
                minimum = float(valid.loc[minimum_index]) if count else None
                threshold = _threshold(column, config)
                readings.append({"item": api_names[column], "unit": "%" if column in PF_ITEMS else "V",
                                 "threshold": threshold, "minimum_observed": minimum,
                                 "minimum_observed_timestamp": subset.loc[minimum_index, "datetime"].isoformat() if count else None,
                                 "maximum_observed": float(valid.max()) if count else None,
                                 "below_threshold": minimum < threshold if count else None,
                                 "data_available": bool(count), "record_count": count,
                                 "coverage": count / EXPECTED_PER_DAY})
            rows.append({"equipment": name, "measurements": readings})
        observed = sum(len(per_equipment.get(name, ())) for name in selected)
        return {**self._provenance(date), **result, "items": rows, "record_count": observed,
                "data_available": any(r["data_available"] for row in rows for r in row["measurements"]),
                "note": "설비·항목별 관측된 5초 측정값의 일 최저·최고값입니다. 일 최저값은 실시간 상태나 하루 전체의 상태를 뜻하지 않으며, 관측이 없는 값은 비워 둡니다."}

    def get_power(self, date, equipment=None, granularity="hour"):
        frame, module, reason = self._select(date, equipment)
        if granularity not in ("hour", "day", "hourly", "daily"):
            reason = "지원하는 집계 단위는 hour 또는 day입니다."
        if reason:
            return {"date": _date(date), "equipment": module or equipment, "series": [], "total_kwh": None, "cost": None, "co2": None, "record_count": 0, "data_available": False, "note": reason}
        frequency = "h" if granularity in ("hour", "hourly") else "D"
        units = 1 if module else len(self.modules)
        grouped = frame.groupby(frame.datetime.dt.floor(frequency)).activePower.agg(["sum", "count"])
        dates = pd.date_range(_date(date), periods=24 if frequency == "h" else 1, freq=frequency)
        expected = (720 if frequency == "h" else EXPECTED_PER_DAY)*units
        series = []
        for stamp in dates:
            count = int(grouped.loc[stamp, "count"]) if stamp in grouped.index else 0
            energy = float(grouped.loc[stamp, "sum"])/720000.0 if count else None
            series.append({"timestamp": stamp.isoformat(), "kwh": energy, "energy_kwh": energy, "cost": energy*180 if energy is not None else None, "co2": energy*.424 if energy is not None else None, "record_count": count, "coverage": count/expected})
        total = float(frame.activePower.sum())/720000.0
        instantaneous = frame.groupby("datetime").activePower.sum()/1000.0
        peak_time = instantaneous.idxmax()
        return {**self._provenance(date), "date": _date(date), "equipment": module, "granularity": "hour" if frequency == "h" else "day", "series": series, "total_kwh": total, "cost": total*180, "co2": total*.424, "peak_kw": float(instantaneous.loc[peak_time]), "peak_timestamp": peak_time.isoformat(), "peak_basis": "five-second mean active power, summed across selected equipment", "record_count": len(frame), "coverage": len(frame)/(EXPECTED_PER_DAY*units), "data_available": True, "power_unit": "W", "energy_unit": "kWh", "cost_unit": "KRW estimate", "co2_unit": "kgCO2 estimate", "note": "실측 유효전력(W)에 5초/3,600,000을 곱해 kWh로 합산했습니다. 요금은 180원/kWh, 탄소는 0.424kgCO2/kWh를 적용한 추정치입니다."}

    def get_power_ranking(self, date):
        frame, _, reason = self._select(date)
        if reason:
            return {"date": _date(date), "ranks": [], "items": [], "data_available": False, "note": reason}
        grouped = frame.groupby(MODULE).activePower.agg(["sum", "count"])
        rows = []
        for module in self.modules:
            count = int(grouped.loc[module, "count"]) if module in grouped.index else 0
            energy = float(grouped.loc[module, "sum"])/720000.0 if count else None
            rows.append({"equipment": module, "kwh": energy, "total_kwh": energy, "cost": energy*180 if energy is not None else None, "co2": energy*.424 if energy is not None else None, "record_count": count, "coverage": count/EXPECTED_PER_DAY, "data_available": bool(count)})
        rows.sort(key=lambda row: -(row["kwh"] if row["kwh"] is not None else -1))
        for index, row in enumerate(rows, 1):
            row["rank"] = index if row["data_available"] else None
        return {**self._provenance(date), "date": _date(date), "ranks": rows, "items": rows, "count": len(rows), "total_kwh": sum(row["kwh"] or 0 for row in rows), "data_available": True, "note": "해당 날짜의 실측 전력량 합계 기준 순위입니다."}

    def get_forecast(self, month="2025-05", equipment=None):
        unavailable = {"month": str(month), "equipment": equipment, "series": [], "total_kwh": None, "cost": None, "co2": None, "period_start": None, "period_end": None, "data_available": False}
        if str(month) != "2025-05":
            return {**unavailable, "note": "제공된 예측은 2025년 5월10일~31일00시의 505시간뿐입니다."}
        try:
            module = self.normalize_equipment(equipment)
            folder = self.root / "outputs/forecast_v2"
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            freeze = json.loads((folder / "model_freeze.json").read_text(encoding="utf-8"))
            if _sha256(self.root / "models/forecast_v2/selected_model.joblib") != freeze["model_sha256"]["selected"]:
                raise ValueError("예측 모델이 변경되어 저장된 예측과 모델의 일치를 확인할 수 없습니다.")
            values = pd.read_csv(folder / "selected/may_equipment_physical.csv", parse_dates=["datetime"])
            expected = pd.date_range("2025-05-10", "2025-05-31", freq="h")
            if len(values) != 505*13 or set(values[MODULE]) != set(self.modules) or values.duplicated([MODULE, "datetime"]).any() or not np.isfinite(values.prediction_kWh).all() or (values.prediction_kWh < 0).any():
                raise ValueError("저장된 예측의 구조를 확인할 수 없습니다.")
            for _, current in values.groupby(MODULE):
                if not pd.DatetimeIndex(current.datetime).equals(expected):
                    raise ValueError("저장된 예측 시각이 505시간 기준과 일치하지 않습니다.")
            if module is not None:
                values = values.loc[values[MODULE] == module]
            if values.empty:
                raise ValueError("해당 설비의 예측값이 없습니다.")
            hourly = values.groupby("datetime").prediction_kWh.sum()
            total = float(hourly.sum())
            score = summary["selected_may"]
            family = summary["selected_family"]
            model_name = {"extra_trees": "ExtraTrees", "random_forest": "RandomForest", "xgboost": "XGBoost", "hist_gradient_boosting": "HistGradientBoosting"}.get(family, family)
            mae = float(score["mae_kWh_if_reference_is_Wh"])
            return {"month": "2025-05", "equipment": module, "period_start": expected[0].isoformat(), "period_end": expected[-1].isoformat(), "period_end_inclusive": True, "hours": 505, "full_month": False, "series": [{"timestamp": timestamp.isoformat(), "kwh": float(value), "cost": float(value)*180, "co2": float(value)*.424} for timestamp, value in hourly.items()], "total_kwh": total, "cost": total*180, "co2": total*.424, "model": model_name, "model_candidate": summary["selected_candidate"], "mae": mae if module is None else None, "factory_mae_kwh": mae, "equipment_mae_kwh": None, "mae_scope": "factory_only", "mae_unit_confirmation_pending": True, "mae_course_native": float(score["mae_course_native"]["hourly_pow"]), "previous_month_diff_pct": None, "prevmonthdiff": None, "prevmonthdiff_note": "4월 실측29일과 5월 예측505시간은 기간이 달라 전월 대비율을 산출하지 않습니다.", "data_available": True, "data_kind": "forecast", "note": "5월 전체가 아닌 5월10일00시~31일00시(포함) 505시간 예측입니다. MAE는 공장 전체에만 해당하며 물리 kWh 해석은 제공자 단위 확인이 필요합니다. 5월 평가는 이미 확인한 데이터에 대한 재평가입니다."}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {**unavailable, "note": str(exc)}
