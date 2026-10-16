"""Reproducible contiguous one-minute aggregation of the December-April raw CSV.

Never reads May answers or outlier labels. localtime is the source's declared
Korean wall-clock time; timestamp disagreement is audited rather than hidden.
Partial moments are combined across CSV chunks before population std is computed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

MODULE = "module(equipment)"
SENSORS = ["activePower", "voltageR", "voltageS", "voltageT",
           "currentR", "currentS", "currentT", "powerFactorR",
           "powerFactorS", "powerFactorT"]
LINE_SENSORS = ["voltageRS", "voltageST", "voltageTR"]
EXPECTED_MASK = sum(1 << s for s in range(0, 60, 5))
PROJECT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT.parent / "프로젝트" / "data" / "rtu_data_full.csv"


def parse_localtime(values):
    """Parse fixed-format declared local wall clock; never infer a timezone."""
    return pd.to_datetime(values.astype(str), format="%Y%m%d%H%M%S", errors="raise")


def chunk_moments(raw: pd.DataFrame) -> pd.DataFrame:
    """Return associative sufficient statistics, including exact second masks."""
    raw = raw.copy()
    local = parse_localtime(raw["localtime"])
    raw["minute"] = local.dt.floor("min")
    keys = [MODULE, "minute"]
    grp = raw.groupby(keys, sort=False, observed=True)
    out = grp.size().rename("n").to_frame()
    out["operation_sum"] = grp["operation"].sum()
    out["operation_count"] = grp["operation"].count()
    for sensor in SENSORS:
        values = grp[sensor].agg(["sum", "count", "min", "max"])
        for stat in values:
            out[f"{sensor}__{stat}"] = values[stat]
        squared = raw[sensor].astype(np.float64).pow(2)
        out[f"{sensor}__sumsq"] = squared.groupby(
            [raw[MODULE], raw["minute"]], sort=False, observed=True).sum()
    for sensor in LINE_SENSORS:
        out[f"{sensor}__min"] = grp[sensor].min()
    group_ids = grp.ngroup().to_numpy()
    masks = np.zeros(len(out), dtype=np.uint64)
    second_mask = np.left_shift(np.uint64(1), local.dt.second.to_numpy(dtype=np.uint64))
    np.bitwise_or.at(masks, group_ids, second_mask)
    out["second_mask"] = masks
    raw["off_grid"] = (local.dt.second % 5 != 0).astype(np.int32)
    raw["missing_sensor_row"] = raw[SENSORS + LINE_SENSORS].isna().any(axis=1).astype(np.int32)
    out["off_grid_records"] = raw.groupby(keys, sort=False, observed=True)["off_grid"].sum()
    out["missing_sensor_rows"] = raw.groupby(keys, sort=False, observed=True)["missing_sensor_row"].sum()
    return out.reset_index().rename(columns={"minute": "timestamp"})


def merge_moments(partials: pd.DataFrame) -> pd.DataFrame:
    """Combine fragments exactly; a chunk boundary never becomes a time gap."""
    keys = [MODULE, "timestamp"]
    agg = {}
    for c in partials.columns:
        if c in keys:
            continue
        if c == "second_mask":
            agg[c] = lambda s: np.bitwise_or.reduce(s.to_numpy(dtype=np.uint64))
        elif c.endswith("__min"):
            agg[c] = "min"
        elif c.endswith("__max"):
            agg[c] = "max"
        else:
            agg[c] = "sum"
    # Only boundary-overlap minutes need a second aggregation.
    repeated = partials.duplicated(keys, keep=False)
    unique = partials.loc[~repeated]
    if repeated.any():
        combined = partials.loc[repeated].groupby(keys, as_index=False, sort=False).agg(agg)
        totals = pd.concat([unique, combined], ignore_index=True)
    else:
        totals = unique.copy()
    totals = totals.sort_values(keys).reset_index(drop=True)
    out = totals[keys].copy()
    out["n"] = totals["n"].astype(np.int32)
    out["operation_mean"] = (totals.operation_sum / totals.operation_count.replace(0, np.nan)).astype(np.float32)
    for sensor in SENSORS:
        count = totals[f"{sensor}__count"].replace(0, np.nan)
        mean = totals[f"{sensor}__sum"] / count
        variance = totals[f"{sensor}__sumsq"] / count - mean.pow(2)
        out[f"{sensor}_mean"] = mean.astype(np.float32)
        out[f"{sensor}_std"] = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
        out[f"{sensor}_min"] = totals[f"{sensor}__min"].astype(np.float32)
        out[f"{sensor}_max"] = totals[f"{sensor}__max"].astype(np.float32)
    for sensor in LINE_SENSORS:
        out[f"{sensor}_min"] = totals[f"{sensor}__min"].astype(np.float32)
    masks = totals["second_mask"].astype(np.uint64)
    unique_n = masks.map(lambda value: int(value).bit_count())
    out["unique_sample_timestamps"] = unique_n.astype(np.int16)
    out["duplicate_records"] = (out.n - unique_n).astype(np.int32)
    out["off_grid_records"] = totals["off_grid_records"].astype(np.int32)
    out["missing_sample_slots"] = masks.map(lambda value: (EXPECTED_MASK & ~int(value)).bit_count()).astype(np.int16)
    out["has_missing_sensor"] = totals["missing_sensor_rows"] > 0
    out["complete_minute"] = ((out.n == 12) & (masks == EXPECTED_MASK) &
                              (out.duplicate_records == 0) & ~out.has_missing_sensor)
    return out


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build(source: Path, output_dir: Path, chunksize=500_000):
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", {"complete": False, "phase": "building"})
    parts_dir = output_dir / "_minute_parts"
    parts_dir.mkdir(exist_ok=True)
    source_stat = source.stat()
    prior_path = PROJECT / "outputs" / "agent_data" / "manifest.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
    metadata = prior.get("signature", {}).get("source_metadata", {})
    same_source = (metadata.get("size") == source_stat.st_size and
                   metadata.get("mtime_ns") == source_stat.st_mtime_ns and
                   prior.get("source_rows_scanned") and prior.get("source_sha256"))
    source_hash = prior.get("source_sha256") if same_source else None
    if source_hash is None:
        h = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        source_hash = h.hexdigest()
    usecols = [MODULE, "timestamp", "localtime", "operation"] + SENSORS + LINE_SENSORS + ["accumActiveEnergy"]
    dtypes = {c: np.float64 for c in SENSORS + LINE_SENSORS + ["accumActiveEnergy", "operation"]}
    dtypes.update({MODULE: str, "timestamp": np.int64, "localtime": str})
    rows = 0
    last_times = {}
    monotonic_violations = Counter()
    timestamp_offsets = Counter()
    operation_values = Counter()
    equipment_rows = Counter()
    files_by_equipment = {}
    observed_min, observed_max = None, None
    unit_ratios = []
    max_memory = 0
    progress = output_dir / "build_progress.json"
    for index, chunk in enumerate(pd.read_csv(source, usecols=usecols, dtype=dtypes, chunksize=chunksize)):
        local = parse_localtime(chunk.localtime)
        operation_values.update({str(k): int(v) for k, v in chunk.operation.value_counts(dropna=False).items()})
        in_scope = (local >= pd.Timestamp("2024-12-01")) & (local < pd.Timestamp("2025-05-01"))
        if not bool(in_scope.all()):
            raise ValueError("Source includes records outside the explicitly allowed Dec-Apr period.")
        minute_min, minute_max = local.min(), local.max()
        observed_min = minute_min if observed_min is None else min(observed_min, minute_min)
        observed_max = minute_max if observed_max is None else max(observed_max, minute_max)
        offsets = chunk["timestamp"].to_numpy() - local.astype("int64").to_numpy() // 1_000_000
        timestamp_offsets.update({str(int(k)): int(v) for k, v in pd.Series(offsets).value_counts().items()})
        for equipment, positions in chunk.groupby(MODULE, sort=False).indices.items():
            times = local.iloc[positions]
            time_values = times.to_numpy()
            monotonic_violations[equipment] += int((np.diff(time_values) < np.timedelta64(0, "ns")).sum())
            if equipment in last_times and times.iloc[0] < last_times[equipment]:
                monotonic_violations[equipment] += 1
            last_times[equipment] = times.iloc[-1]
            equipment_rows[equipment] += len(positions)
            # Small physical-unit check, independent of anomaly labels.
            if len(unit_ratios) < 50000:
                selected = chunk.iloc[positions[: min(5000, len(positions))]]
                seconds = times.iloc[: len(selected)].diff().dt.total_seconds()
                delta_wh = selected.accumActiveEnergy.diff()
                expected_wh = selected.activePower.shift(1) * seconds / 3600
                valid = (seconds == 5) & (delta_wh > 0) & (expected_wh > 0)
                unit_ratios.extend((delta_wh[valid] / expected_wh[valid]).tolist())
        partial = chunk_moments(chunk)
        for equipment, group in partial.groupby(MODULE, sort=False):
            code = hashlib.sha256(equipment.encode("utf-8")).hexdigest()[:12]
            folder = parts_dir / code
            folder.mkdir(exist_ok=True)
            path = folder / f"part_{index:05d}.parquet"
            group.to_parquet(path, index=False)
            files_by_equipment.setdefault(equipment, []).append(path)
        rows += len(chunk)
        rss = psutil.Process().memory_info().rss
        max_memory = max(max_memory, rss)
        record = {"phase": "scan", "rows": rows, "chunk": index + 1,
                  "elapsed_seconds": round(time.perf_counter() - started, 2),
                  "rss_mb": round(rss / 1024**2, 2)}
        _write_json(progress, record)
        if index % 5 == 0:
            print(json.dumps(record), flush=True)
        del chunk, partial
        gc.collect()
    if source.stat().st_size != source_stat.st_size or source.stat().st_mtime_ns != source_stat.st_mtime_ns:
        raise RuntimeError("Source changed during reading.")
    if same_source and rows != prior["source_rows_scanned"]:
        raise RuntimeError("Source row count differs from the matching prior manifest.")
    output_file = output_dir / "minute.parquet"
    temporary_output = output_dir / "minute.building.parquet"
    writer = None
    equipment_stats = {}
    total_minutes = 0
    for equipment in sorted(files_by_equipment):
        partials = pd.concat([pd.read_parquet(p) for p in files_by_equipment[equipment]], ignore_index=True)
        data = merge_moments(partials)
        table = pa.Table.from_pandas(data, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(temporary_output, table.schema, compression="zstd", compression_level=3)
        writer.write_table(table, row_group_size=50_000)
        expected_minutes = int((data.timestamp.max() - data.timestamp.min()).total_seconds() / 60) + 1
        equipment_stats[equipment] = {
            "raw_rows": equipment_rows[equipment], "minutes": len(data),
            "first_minute": data.timestamp.min().isoformat(),
            "last_minute": data.timestamp.max().isoformat(),
            "complete_minutes": int(data.complete_minute.sum()),
            "incomplete_minutes": int((~data.complete_minute).sum()),
            "entire_missing_minutes_between_first_last": expected_minutes - len(data),
            "missing_sample_slots_in_observed_minutes": int(data.missing_sample_slots.sum()),
            "duplicate_records": int(data.duplicate_records.sum()),
            "off_grid_records": int(data.off_grid_records.sum()),
            "minutes_with_missing_sensor": int(data.has_missing_sensor.sum()),
            "source_time_order_violations": monotonic_violations[equipment],
            "operation_values": [float(x) for x in data.operation_mean.dropna().unique()],
        }
        total_minutes += len(data)
        max_memory = max(max_memory, psutil.Process().memory_info().rss)
        print(f"finalized {equipment}: {len(data):,} minutes", flush=True)
        del partials, data, table
        gc.collect()
    if writer:
        writer.close()
        temporary_output.replace(output_file)
    ratios = np.array(unit_ratios, dtype=np.float64)
    manifest = {
        "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "bytes": source_stat.st_size,
                   "mtime_ns": source_stat.st_mtime_ns, "sha256": source_hash,
                   "sha256_method": "prior manifest SHA reused after exact size/mtime_ns and row count match" if same_source else "full file SHA256",
                   "prior_manifest": str(prior_path) if same_source else None, "rows": rows},
        "output": {"path": str(output_file), "bytes": output_file.stat().st_size,
                   "rows": total_minutes, "sort_order": [MODULE, "timestamp"],
                   "float_storage": "float32; sufficient-statistic accumulation is float64",
                   "population_standard_deviation": True},
        "time": {"first_localtime": observed_min.isoformat(), "last_localtime": observed_max.isoformat(),
                 "canonical": "localtime parsed as declared Korean local wall-clock, timezone-naive",
                 "raw_unix_ms_minus_local_naive_epoch_ms_distribution": dict(timestamp_offsets),
                 "warning": "Source Unix timestamp and declared localtime disagree. Do not reinterpret localtime through source Unix timestamps."},
        "units": {"activePower": "W (working assumption; source unit confirmation remains pending)",
                  "activePower_evidence": "accumActiveEnergy successive Wh increments approximately match activePower*5/3600, not activePower*1000*5/3600",
                  "accumulator_delta_over_expected_wh_from_W": {
                      "samples": len(ratios), "median": float(np.median(ratios)),
                      "p05": float(np.quantile(ratios, .05)), "p95": float(np.quantile(ratios, .95))},
                  "voltage": "V", "current": "A", "powerFactor": "%"},
        "quality": {"expected_samples_per_minute": 12,
                    "complete_minute": "exactly 12 records at unique seconds 0,5,...55, all required sensor values present",
                    "missing_minutes": "not padded; downstream rolling features must break at gaps",
                    "duplicate_policy": "retained in moments, explicitly flagged, complete_minute=False; exclude incomplete minutes in model experiments",
                    "partial_final_minute": "retained and explicitly marked incomplete",
                    "source_sorted_within_equipment": not any(monotonic_violations.values()),
                    "raw_operation_value_counts": dict(operation_values)},
        "equipment": equipment_stats,
        "label_policy": "No is_outlier or flag labels read, and no May ground-truth file read.",
        "runtime": {"elapsed_seconds": round(time.perf_counter() - started, 2),
                    "peak_observed_rss_mb": round(max_memory / 1024**2, 2), "chunksize": chunksize}
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(progress, {"phase": "complete", "rows": rows, "minutes": total_minutes,
                           "elapsed_seconds": manifest["runtime"]["elapsed_seconds"]})
    print(json.dumps({"complete": True, "rows": rows, "minutes": total_minutes, "runtime": manifest["runtime"]}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT / "data")
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    build(args.source, args.output_dir, args.chunksize)
