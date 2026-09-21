"""Frozen Binance Spot kline ingestion and replay resource catalogs.

Acquisition is intentionally separate from evaluation.  Benchmark runs only
read content-addressed local artifacts and never fall back to a live endpoint.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import statistics
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


SNAPSHOT_SCHEMA = "evolens-binance-spot-klines-v1"
REPLAY_SCHEMA = "evolens-market-binance-replay-v1"
BINANCE_ARCHIVE_ROOT = "https://data.binance.vision/data/spot/monthly/klines"
KLINE_COLUMNS = 12
_ARCHIVE_RE = re.compile(
    r"^(?P<symbol>[A-Z0-9]+)-(?P<interval>[0-9]+[smhdwM])-(?P<period>[0-9]{4}-[0-9]{2}(?:-[0-9]{2})?)\.zip$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _timestamp_to_us(raw: str, period: str) -> Tuple[int, str]:
    value = int(raw)
    # Binance Spot bulk archives changed from milliseconds to microseconds on
    # 2025-01-01.  Use the archive period as the schema contract rather than a
    # magnitude heuristic, and retain the original unit in provenance.
    unit = "us" if period >= "2025-01" else "ms"
    return (value if unit == "us" else value * 1000), unit


def _interval_us(interval: str) -> int:
    match = re.fullmatch(r"([0-9]+)([smhdw])", interval)
    if not match:
        raise ValueError(f"unsupported fixed Binance interval: {interval}")
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[match.group(2)]
    return int(match.group(1)) * multiplier * 1_000_000


@dataclass(frozen=True)
class FrozenBar:
    symbol: str
    interval: str
    open_time_us: int
    close_time_us: int
    open: str
    high: str
    low: str
    close: str
    base_volume: str
    quote_volume: str
    trade_count: int
    taker_buy_base_volume: str
    taker_buy_quote_volume: str
    source_object: str
    source_line: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenBar":
        return cls(**dict(value))


@dataclass(frozen=True)
class ReplayWindow:
    resource_id: str
    source_snapshot_sha256: str
    symbol: str
    partition: str
    start_time_us: int
    decision_time_us: List[int]
    execution_time_us: List[int]
    decision_prices: List[float]
    execution_prices: List[float]
    valuation_prices: List[float]
    reference_values: List[float]
    quote_volume_relative: List[float]
    trade_count_relative: List[float]
    taker_buy_ratio: List[float]
    normalization_factor: float
    feature_scores: Dict[str, float]
    field_contract: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayWindow":
        return cls(**dict(value))


def _parse_checksum(path: Path) -> str:
    token = path.read_text(encoding="utf-8").strip().split()[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ValueError(f"invalid Binance checksum file: {path}")
    return token


def download_binance_monthly_klines(
    symbols: Sequence[str], months: Sequence[str], interval: str, output_dir: Path, workers: int = 4
) -> Dict[str, Any]:
    """Download immutable monthly bulk objects, never a live market stream."""
    if not symbols or not months:
        raise ValueError("symbols and months must be non-empty")
    _interval_us(interval)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    normalized_symbols = sorted({item.upper() for item in symbols})
    normalized_months = sorted(set(months))
    jobs: List[Tuple[str, str]] = []
    for symbol in normalized_symbols:
        if not re.fullmatch(r"[A-Z0-9]+", symbol):
            raise ValueError(f"invalid Binance symbol: {symbol}")
        for month in normalized_months:
            if not re.fullmatch(r"[0-9]{4}-[0-9]{2}", month):
                raise ValueError(f"invalid month: {month}")
            jobs.append((symbol, month))

    def fetch(url: str) -> Tuple[bytes, Dict[str, Any]]:
        request = urllib.request.Request(url, headers={"User-Agent": "EvoPathBench/1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
            headers = {
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "content_length": response.headers.get("Content-Length"),
            }
        return payload, headers

    def download_one(job: Tuple[str, str]) -> Dict[str, Any]:
        symbol, month = job
        filename = f"{symbol}-{interval}-{month}.zip"
        url = f"{BINANCE_ARCHIVE_ROOT}/{symbol}/{interval}/{filename}"
        target = output_dir / filename
        checksum_target = output_dir / f"{filename}.CHECKSUM"
        sidecar_target = output_dir / f"{filename}.SOURCE.json"
        if target.exists() and checksum_target.exists() and sidecar_target.exists():
            official = _parse_checksum(checksum_target)
            actual = sha256_file(target)
            sidecar = json.loads(sidecar_target.read_text(encoding="utf-8"))
            if official != actual or sidecar.get("raw_zip_sha256") != actual or sidecar.get("exact_url") != url:
                raise ValueError(f"existing local object failed provenance verification: {filename}")
            return sidecar
        checksum_before, checksum_headers_before = fetch(url + ".CHECKSUM")
        payload, zip_headers = fetch(url)
        checksum_after, checksum_headers_after = fetch(url + ".CHECKSUM")
        if checksum_before != checksum_after:
            raise ValueError(f"official checksum changed during acquisition: {filename}")
        checksum_target.write_bytes(checksum_after)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite changed source object: {target}")
        target.write_bytes(payload)
        official = _parse_checksum(checksum_target)
        actual = sha256_file(target)
        if official != actual:
            raise ValueError(f"official checksum mismatch after download: {filename}")
        headers = {
            filename: zip_headers,
            checksum_target.name + ":before": checksum_headers_before,
            checksum_target.name + ":after": checksum_headers_after,
        }
        sidecar = {
            "exact_url": url,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "headers": headers,
            "raw_zip_sha256": actual,
            "official_checksum": official,
            "checksum_file_sha256": sha256_file(checksum_target),
            "checksum_double_read_consistent": True,
        }
        sidecar_target.write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return sidecar

    objects: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, job): job for job in jobs}
        for future in as_completed(futures):
            objects.append(future.result())
    objects.sort(key=lambda item: item["exact_url"])
    manifest = {
        "kind": "binance_monthly_bulk_acquisition",
        "market": "spot",
        "data_type": "klines",
        "symbols": normalized_symbols,
        "months": normalized_months,
        "interval": interval,
        "workers": workers,
        "objects": objects,
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_decimal(raw: str, label: str) -> Decimal:
    try:
        value = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"invalid decimal {label}={raw!r}") from error
    if not value.is_finite():
        raise ValueError(f"non-finite decimal {label}={raw!r}")
    return value


def _parse_kline_row(
    row: Sequence[str], symbol: str, interval: str, period: str, source_object: str, line_no: int,
    allow_truncated_interval: bool = False,
) -> FrozenBar:
    if len(row) != KLINE_COLUMNS:
        raise ValueError(f"{source_object}:{line_no}: expected 12 columns, found {len(row)}")
    open_time_us, unit = _timestamp_to_us(row[0], period)
    close_time_us, close_unit = _timestamp_to_us(row[6], period)
    if unit != close_unit or close_time_us < open_time_us:
        raise ValueError(f"{source_object}:{line_no}: invalid kline time interval")
    duration = _interval_us(interval)
    source_tick_us = 1 if unit == "us" else 1000
    alignment_origin = 4 * 86_400_000_000 if interval.endswith("w") else 0
    expected_close_delta = duration - source_tick_us
    irregular_ok = allow_truncated_interval and close_time_us >= open_time_us
    if (open_time_us - alignment_origin) % duration != 0 or (
        close_time_us - open_time_us != expected_close_delta and not irregular_ok
    ):
        raise ValueError(f"{source_object}:{line_no}: kline is not aligned to declared interval")
    open_price = _parse_decimal(row[1], "open")
    high = _parse_decimal(row[2], "high")
    low = _parse_decimal(row[3], "low")
    close = _parse_decimal(row[4], "close")
    values = [open_price, high, low, close]
    if any(value <= 0 for value in values) or low > min(open_price, close) or high < max(open_price, close) or low > high:
        raise ValueError(f"{source_object}:{line_no}: invalid OHLC relation")
    base_volume = _parse_decimal(row[5], "base_volume")
    quote_volume = _parse_decimal(row[7], "quote_volume")
    taker_base = _parse_decimal(row[9], "taker_buy_base_volume")
    taker_quote = _parse_decimal(row[10], "taker_buy_quote_volume")
    trade_count = int(row[8])
    if min(base_volume, quote_volume, taker_base, taker_quote) < 0 or trade_count < 0:
        raise ValueError(f"{source_object}:{line_no}: negative volume or trade count")
    if taker_base > base_volume or taker_quote > quote_volume:
        raise ValueError(f"{source_object}:{line_no}: taker-buy volume exceeds total volume")
    tolerance = Decimal("0.001")
    if (base_volume == 0) != (quote_volume == 0):
        raise ValueError(f"{source_object}:{line_no}: inconsistent zero base/quote volume")
    if base_volume > 0:
        vwap = quote_volume / base_volume
        if vwap < low * (1 - tolerance) or vwap > high * (1 + tolerance):
            raise ValueError(f"{source_object}:{line_no}: quote/base VWAP is outside OHLC")
    if (taker_base == 0) != (taker_quote == 0):
        raise ValueError(f"{source_object}:{line_no}: inconsistent zero taker base/quote volume")
    if taker_base > 0:
        taker_vwap = taker_quote / taker_base
        if taker_vwap < low * (1 - tolerance) or taker_vwap > high * (1 + tolerance):
            raise ValueError(f"{source_object}:{line_no}: taker VWAP is outside OHLC")
    return FrozenBar(
        symbol=symbol,
        interval=interval,
        open_time_us=open_time_us,
        close_time_us=close_time_us,
        open=str(open_price),
        high=str(high),
        low=str(low),
        close=str(close),
        base_volume=str(base_volume),
        quote_volume=str(quote_volume),
        trade_count=trade_count,
        taker_buy_base_volume=str(taker_base),
        taker_buy_quote_volume=str(taker_quote),
        source_object=source_object,
        source_line=line_no,
    )


def _iter_archive(
    path: Path, allow_truncated_intervals: bool = False
) -> Tuple[Iterator[FrozenBar], Dict[str, Any]]:
    match = _ARCHIVE_RE.match(path.name)
    if not match:
        raise ValueError(f"unsupported Binance archive name: {path.name}")
    metadata = match.groupdict()
    archive_sha = sha256_file(path)
    checksum_path = path.with_name(path.name + ".CHECKSUM")
    official_checksum: Optional[str] = None
    checksum_status = "missing"
    if checksum_path.exists():
        official_checksum = _parse_checksum(checksum_path)
        if official_checksum != archive_sha:
            raise ValueError(f"official checksum mismatch for {path.name}")
        checksum_status = "verified"

    archive = zipfile.ZipFile(path, "r")
    members = sorted(name for name in archive.namelist() if not name.endswith("/"))
    if len(members) != 1:
        archive.close()
        raise ValueError(f"{path.name}: expected one CSV member, found {members}")
    member = members[0]

    def rows() -> Iterator[FrozenBar]:
        previous_time: Optional[int] = None
        try:
            with archive.open(member, "r") as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                for line_no, row in enumerate(csv.reader(text), start=1):
                    if line_no == 1 and row and not row[0].strip().lstrip("-").isdigit():
                        continue
                    bar = _parse_kline_row(
                        row,
                        metadata["symbol"],
                        metadata["interval"],
                        metadata["period"],
                        path.name,
                        line_no,
                        allow_truncated_interval=allow_truncated_intervals,
                    )
                    if previous_time is not None and bar.open_time_us <= previous_time:
                        raise ValueError(f"{path.name}:{line_no}: duplicate or non-monotonic open time")
                    previous_time = bar.open_time_us
                    yield bar
        finally:
            archive.close()

    provenance = {
        "market": "binance_spot",
        "data_type": "klines",
        "symbol": metadata["symbol"],
        "interval": metadata["interval"],
        "period": metadata["period"],
        "exact_url": f"{BINANCE_ARCHIVE_ROOT}/{metadata['symbol']}/{metadata['interval']}/{path.name}",
        "source_object": path.name,
        "raw_zip_sha256": archive_sha,
        "raw_size": path.stat().st_size,
        "official_checksum": official_checksum,
        "checksum_status": checksum_status,
        "checksum_file_sha256": sha256_file(checksum_path) if checksum_path.exists() else None,
        "timestamp_unit": "us" if metadata["period"] >= "2025-01" else "ms",
        "archive_member": member,
    }
    acquisition_sidecar = path.with_name(path.name + ".SOURCE.json")
    if acquisition_sidecar.exists():
        acquisition = json.loads(acquisition_sidecar.read_text(encoding="utf-8"))
        if acquisition.get("raw_zip_sha256") != archive_sha:
            raise ValueError(f"acquisition sidecar hash mismatch for {path.name}")
        provenance["acquisition"] = acquisition
    return rows(), provenance


def freeze_binance_klines(
    archive_paths: Sequence[Path],
    output_dir: Path,
    require_official_checksums: bool = True,
    require_acquisition_metadata: bool = False,
) -> Dict[str, Any]:
    """Canonicalize official Binance archive objects into an immutable snapshot."""
    if not archive_paths:
        raise ValueError("at least one Binance archive is required")
    bars: List[FrozenBar] = []
    sources: List[Dict[str, Any]] = []
    for path in sorted((Path(item) for item in archive_paths), key=lambda item: item.name):
        iterator, source = _iter_archive(path)
        if require_official_checksums and source["checksum_status"] != "verified":
            raise ValueError(f"missing official checksum for {path.name}")
        if require_acquisition_metadata and "acquisition" not in source:
            raise ValueError(f"missing acquisition sidecar for {path.name}")
        source_bars = list(iterator)
        if not source_bars:
            raise ValueError(f"empty Binance archive: {path.name}")
        source["row_count"] = len(source_bars)
        source["min_open_time_us"] = source_bars[0].open_time_us
        source["max_close_time_us"] = source_bars[-1].close_time_us
        bars.extend(source_bars)
        sources.append(source)
    bars.sort(key=lambda item: (item.symbol, item.open_time_us, item.source_object, item.source_line))
    seen = set()
    for bar in bars:
        key = (bar.symbol, bar.interval, bar.open_time_us)
        if key in seen:
            raise ValueError(f"duplicate canonical bar: {key}")
        seen.add(key)

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical_json(bar.to_dict()) for bar in bars)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as compressed:
        compressed.write(payload)
    compressed_payload = buffer.getvalue()
    bars_path = output_dir / "bars.jsonl.gz"
    bars_path.write_bytes(compressed_payload)
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA,
        "source_provider": "Binance public data",
        "source_policy": "official monthly Spot kline archives; evaluation has no network fallback",
        "bar_count": len(bars),
        "symbols": sorted({bar.symbol for bar in bars}),
        "intervals": sorted({bar.interval for bar in bars}),
        "sources": sources,
        "sha256": {"bars.jsonl.gz": hashlib.sha256(compressed_payload).hexdigest()},
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_frozen_bars(snapshot_dir: Path) -> Tuple[List[FrozenBar], Dict[str, Any]]:
    manifest = json.loads((snapshot_dir / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError(f"unsupported Binance snapshot schema: {manifest.get('schema_version')}")
    path = snapshot_dir / "bars.jsonl.gz"
    expected = manifest["sha256"]["bars.jsonl.gz"]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"frozen bar hash mismatch: expected={expected}, actual={actual}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        bars = [FrozenBar.from_dict(json.loads(line)) for line in handle if line.strip()]
    if len(bars) != manifest["bar_count"]:
        raise ValueError("frozen bar count mismatch")
    return bars, manifest


def _corr(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    scale_left = math.sqrt(sum((x - mean_left) ** 2 for x in left))
    scale_right = math.sqrt(sum((y - mean_right) ** 2 for y in right))
    return numerator / max(scale_left * scale_right, 1e-15)


def _feature_scores(bars: Sequence[FrozenBar]) -> Dict[str, float]:
    closes = [float(item.close) for item in bars]
    returns = [math.log(current / previous) for previous, current in zip(closes, closes[1:])]
    volatility = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    scale = max(volatility, 1e-10)
    net = math.log(closes[-1] / closes[0])
    lag_corr = _corr(returns[:-1], returns[1:])
    quote_volumes = [max(float(item.quote_volume), 1e-12) for item in bars]
    imbalances = []
    for item in bars:
        quote = float(item.quote_volume)
        taker = float(item.taker_buy_quote_volume)
        imbalances.append(2.0 * taker / quote - 1.0 if quote > 0 else 0.0)
    reflexivity = _corr(imbalances[:-1], returns[1:])
    peak = closes[0]
    max_drawdown = 0.0
    for price in closes:
        peak = max(peak, price)
        max_drawdown = max(max_drawdown, 1.0 - price / peak)
    ordered = sorted(returns)
    tail_count = max(1, int(math.ceil(0.10 * len(ordered))))
    downside_cvar = -statistics.fmean(ordered[:tail_count]) if ordered else 0.0
    positive_jump = max(returns, default=0.0) / scale
    negative_jump = -min(returns, default=0.0) / scale
    return {
        "trend_positive": net / (scale * math.sqrt(max(1, len(returns)))),
        "trend_negative": -net / (scale * math.sqrt(max(1, len(returns)))),
        "mean_reversion": -lag_corr,
        "momentum_persistence": lag_corr,
        "liquidity_low": -math.log(statistics.median(quote_volumes)),
        "liquidity_high": math.log(statistics.median(quote_volumes)),
        "event_positive": positive_jump,
        "event_negative": negative_jump,
        "reflexivity_positive": reflexivity,
        "reflexivity_negative": -reflexivity,
        "risk_stress": max_drawdown + downside_cvar,
        "risk_calm": -(max_drawdown + downside_cvar),
        "realized_volatility": volatility,
        "max_drawdown": max_drawdown,
    }


def _aggregate_chunk(items: Sequence[FrozenBar]) -> FrozenBar:
    if not items:
        raise ValueError("cannot aggregate an empty bar chunk")
    return FrozenBar(
        symbol=items[0].symbol,
        interval=f"{len(items)}x{items[0].interval}",
        open_time_us=items[0].open_time_us,
        close_time_us=items[-1].close_time_us,
        open=items[0].open,
        high=str(max(Decimal(item.high) for item in items)),
        low=str(min(Decimal(item.low) for item in items)),
        close=items[-1].close,
        base_volume=str(sum(Decimal(item.base_volume) for item in items)),
        quote_volume=str(sum(Decimal(item.quote_volume) for item in items)),
        trade_count=sum(item.trade_count for item in items),
        taker_buy_base_volume=str(sum(Decimal(item.taker_buy_base_volume) for item in items)),
        taker_buy_quote_volume=str(sum(Decimal(item.taker_buy_quote_volume) for item in items)),
        source_object=items[0].source_object,
        source_line=items[0].source_line,
    )


def build_replay_windows(
    bars: Sequence[FrozenBar],
    source_snapshot_sha256: str,
    horizon: int,
    stride: Optional[int] = None,
    fit_fraction: float = 0.20,
    calibration_fraction: float = 0.40,
    development_fraction: float = 0.20,
    bar_size: int = 1,
) -> List[ReplayWindow]:
    """Build causal h+1-bar windows with chronological partitions."""
    if horizon < 5:
        raise ValueError("horizon must be at least 5")
    if bar_size < 1:
        raise ValueError("bar_size must be positive")
    if not 0 < fit_fraction < 1 or not 0 < calibration_fraction < 1 or not 0 < development_fraction < 1:
        raise ValueError("split fractions must be inside (0, 1)")
    if fit_fraction + calibration_fraction + development_fraction >= 1:
        raise ValueError("fit, calibration, and development fractions leave no test partition")
    stride = stride or horizon + 1
    if stride < horizon + 1:
        raise ValueError("stride must be at least horizon+1 to prevent overlapping windows")
    grouped: Dict[Tuple[str, str], List[FrozenBar]] = {}
    for bar in bars:
        grouped.setdefault((bar.symbol, bar.interval), []).append(bar)
    all_times = sorted({bar.open_time_us for bar in bars})
    if len(all_times) < horizon + 1:
        raise ValueError("not enough bars for a replay window")
    fit_cut = all_times[min(len(all_times) - 1, int(len(all_times) * fit_fraction))]
    calibration_cut = all_times[
        min(len(all_times) - 1, int(len(all_times) * (fit_fraction + calibration_fraction)))
    ]
    development_cut = all_times[
        min(
            len(all_times) - 1,
            int(len(all_times) * (fit_fraction + calibration_fraction + development_fraction)),
        )
    ]

    windows: List[ReplayWindow] = []
    calibration_scales: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for (symbol, source_interval), values in grouped.items():
        fit_values = [item for item in values if item.close_time_us < fit_cut]
        if not fit_values:
            raise ValueError(f"no predeployment fit bars available for {symbol}/{source_interval}")
        calibration_scales[(symbol, source_interval)] = (
            max(statistics.median(float(item.quote_volume) for item in fit_values), 1e-12),
            max(statistics.median(item.trade_count for item in fit_values), 1.0),
        )
    for (symbol, interval), values in sorted(grouped.items()):
        values.sort(key=lambda item: item.open_time_us)
        canonical_delta = _interval_us(interval)
        raw_window_size = (horizon + 1) * bar_size
        for start in range(0, len(values) - raw_window_size + 1, stride * bar_size):
            raw_chunk = values[start : start + raw_window_size]
            if canonical_delta <= 0 or any(
                right.open_time_us - left.open_time_us != canonical_delta
                for left, right in zip(raw_chunk, raw_chunk[1:])
            ):
                continue
            chunk = [
                _aggregate_chunk(raw_chunk[offset : offset + bar_size])
                for offset in range(0, raw_window_size, bar_size)
            ]
            start_time = chunk[0].open_time_us
            outcome_end = chunk[-1].close_time_us
            if start_time >= fit_cut and outcome_end < calibration_cut:
                partition = "calibration"
            elif start_time >= calibration_cut and outcome_end < development_cut:
                partition = "development"
            elif start_time >= development_cut:
                partition = "test"
            else:
                # Purge windows crossing a split boundary.
                continue
            normalization = 100.0 / float(chunk[0].close)
            decision = [float(item.close) * normalization for item in chunk[:-1]]
            execution = [float(item.open) * normalization for item in chunk[1:]]
            valuation = [float(item.close) * normalization for item in chunk[1:]]
            references = []
            for item in chunk[:-1]:
                base = float(item.base_volume)
                vwap = float(item.quote_volume) / base if base > 0 else float(item.close)
                references.append(vwap * normalization)
            median_quote, median_trades = calibration_scales[(symbol, interval)]
            quote_relative = [float(item.quote_volume) / median_quote for item in chunk[:-1]]
            trade_relative = [item.trade_count / median_trades for item in chunk[:-1]]
            taker_ratios = [
                float(item.taker_buy_quote_volume) / float(item.quote_volume)
                if float(item.quote_volume) > 0
                else 0.5
                for item in chunk[:-1]
            ]
            identity = f"{source_snapshot_sha256}|{symbol}|{interval}|{start_time}|{outcome_end}|{horizon}"
            resource_id = "bw-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            windows.append(
                ReplayWindow(
                    resource_id=resource_id,
                    source_snapshot_sha256=source_snapshot_sha256,
                    symbol=symbol,
                    partition=partition,
                    start_time_us=start_time,
                    decision_time_us=[item.close_time_us for item in chunk[:-1]],
                    execution_time_us=[item.open_time_us for item in chunk[1:]],
                    decision_prices=decision,
                    execution_prices=execution,
                    valuation_prices=valuation,
                    reference_values=references,
                    quote_volume_relative=quote_relative,
                    trade_count_relative=trade_relative,
                    taker_buy_ratio=taker_ratios,
                    normalization_factor=normalization,
                    feature_scores=_feature_scores(chunk),
                    field_contract={
                        "price_path": "observed_normalized",
                        "reference_value": "causal_derived_closed_bar_vwap",
                        "volume_and_orderflow": "observed_normalized",
                        "fee_spread_impact": "simulated_execution_contract",
                    },
                )
            )
    if not windows:
        raise ValueError("no gap-free replay windows were produced")
    return windows


class FrozenMarketCatalog:
    def __init__(self, windows: Iterable[ReplayWindow]) -> None:
        self._windows = {item.resource_id: item for item in windows}
        if len(self._windows) == 0:
            raise ValueError("empty replay catalog")

    def get(self, resource_id: str) -> ReplayWindow:
        try:
            return self._windows[resource_id]
        except KeyError as error:
            raise ValueError(f"unknown replay resource: {resource_id}") from error

    def __len__(self) -> int:
        return len(self._windows)


def write_replay_catalog(path: Path, windows: Sequence[ReplayWindow]) -> str:
    payload = b"".join(_canonical_json(item.to_dict()) for item in sorted(windows, key=lambda item: item.resource_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_replay_catalog(dataset_dir: Path, manifest: Mapping[str, Any]) -> Optional[FrozenMarketCatalog]:
    track = manifest.get("track", {})
    if track.get("kind") != "binance_replay":
        return None
    relative = track.get("resource_catalog", "resources/replay_windows.jsonl")
    path = (dataset_dir / relative).resolve()
    root = dataset_dir.resolve()
    if root not in path.parents:
        raise ValueError(f"resource path escapes dataset directory: {relative}")
    expected = manifest["sha256"].get(relative)
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"replay resource hash mismatch: {relative}")
    windows = [ReplayWindow.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(windows) != track.get("resource_count"):
        raise ValueError("replay resource count mismatch")
    return FrozenMarketCatalog(windows)
