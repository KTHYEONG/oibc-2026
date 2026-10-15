"""제출 파일 바이트 검사."""

from __future__ import annotations

import codecs
import logging
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from src.oibc.config import CompetitionSpec, DataPaths

logger = logging.getLogger(__name__)

_COMPOSITION_HEADER = ("site_id", "group")
_FORECAST_HEADER = ("group", "ts", "forecast_kwh")
_KST_OFFSET = "+09:00"
_KST = "Asia/Seoul"
_MAX_EXAMPLES = 5


class SubmissionError(ValueError):
    """제출 계약 위반."""


@dataclass(frozen=True)
class SubmissionReference:
    """채점 기준 정보: 발전소 id·용량(정확한 십진수)과 채점 시각."""

    site_capacity: Mapping[str, Decimal]
    score_ts: tuple[str, ...]
    score_index: pd.DatetimeIndex


@dataclass(frozen=True)
class ValidationReport:
    """제출 검증 결과. errors가 비어 있어야 제출 가능."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    group_capacity_kw: Mapping[str, float]
    group_forecast_total_kwh: Mapping[str, float]
    n_composition_rows: int
    n_forecast_rows: int

    @property
    def ok(self) -> bool:
        """오류가 없으면 True."""
        return len(self.errors) == 0


def _resolve_spec(spec: CompetitionSpec | None) -> CompetitionSpec:
    return spec if spec is not None else CompetitionSpec()


def load_reference(
    paths: DataPaths | None = None, spec: CompetitionSpec | None = None
) -> SubmissionReference:
    """원본 sites.csv와 smp.csv에서 기준 정보를 읽는다. 기준 파일 계약 위반 시 SubmissionError."""
    cfg = _resolve_spec(spec)
    base = (paths if paths is not None else DataPaths()).raw_dir
    try:
        sites = pd.read_csv(base / "sites.csv", dtype=str, keep_default_na=False)
    except Exception as e:
        raise SubmissionError(f"reference: cannot read sites.csv ({e})") from e
    try:
        smp = pd.read_csv(base / "smp.csv", dtype=str, keep_default_na=False)
    except Exception as e:
        raise SubmissionError(f"reference: cannot read smp.csv ({e})") from e
    if "site_id" not in sites.columns or "capacity_kw" not in sites.columns:
        raise SubmissionError("reference: sites.csv missing site_id/capacity_kw")
    ids = [str(v) for v in sites["site_id"].tolist()]
    if len(ids) != cfg.n_sites:
        raise SubmissionError(f"reference: sites count={len(ids)} expected={cfg.n_sites}")
    if len(set(ids)) != len(ids):
        raise SubmissionError("reference: duplicate site_id")
    caps: dict[str, Decimal] = {}
    for sid, raw in zip(ids, [str(v) for v in sites["capacity_kw"].tolist()], strict=True):
        try:
            cap = Decimal(raw.strip())
        except (InvalidOperation, ValueError) as e:
            raise SubmissionError(f"reference: invalid capacity_kw site={sid}") from e
        if cap <= 0:
            raise SubmissionError(f"reference: non-positive capacity site={sid}")
        caps[sid] = cap
    if "ts" not in smp.columns:
        raise SubmissionError("reference: smp.csv missing ts")
    raw_ts = [str(v) for v in smp["ts"].tolist()]
    if len(raw_ts) != cfg.n_eval_hours:
        raise SubmissionError(f"reference: smp count={len(raw_ts)} expected={cfg.n_eval_hours}")
    try:
        parsed = pd.DatetimeIndex(pd.to_datetime(raw_ts, format="mixed"))
    except Exception as e:
        raise SubmissionError(f"reference: invalid ts ({e})") from e
    if parsed.tz is None:
        raise SubmissionError("reference: smp ts tz-naive")
    index = parsed.tz_convert(_KST)
    if len(set(index)) != len(index):
        raise SubmissionError("reference: duplicate ts")
    if any(index[i] > index[i + 1] for i in range(len(index) - 1)):
        raise SubmissionError("reference: ts not ascending")
    if index[0] != cfg.eval_start or index[-1] != cfg.eval_end:
        raise SubmissionError("reference: eval window mismatch")
    return SubmissionReference(site_capacity=caps, score_ts=tuple(raw_ts), score_index=index)


def _read_text_lines(path: Path, kind: str) -> tuple[list[str] | None, list[str], str | None]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, [f"{kind}: file not found"], None
    except OSError as e:
        return None, [f"{kind}: cannot read ({e})"], None
    if raw.startswith(codecs.BOM_UTF8):
        return None, [f"{kind}: bom not allowed"], None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, [f"{kind}: not utf-8"], None
    lines = text.splitlines()
    return lines, [], text


def _blank_line_errors(lines: list[str], kind: str) -> list[str]:
    bad = [i + 1 for i, line in enumerate(lines) if line.strip() == ""]
    if not bad:
        return []
    shown = ",".join(str(n) for n in bad[:_MAX_EXAMPLES])
    return [f"{kind}: blank line count={len(bad)} rows={shown}"]


def _field_errors(
    lines: list[str], kind: str, n_fields: int, header: tuple[str, ...]
) -> tuple[list[str], list[list[str]]]:
    errors: list[str] = []
    rows: list[list[str]] = []
    if not lines:
        errors.append(f"{kind}: missing header expected={','.join(header)}")
        return errors, rows
    if lines[0] != ",".join(header):
        errors.append(f"{kind}: bad header got={lines[0][:80]}")
    bad: dict[str, list[int]] = {"bad field count": [], "surrounding whitespace": []}
    for i, line in enumerate(lines[1:], start=2):
        if line.strip() == "":
            continue
        parts = line.split(",")
        if len(parts) != n_fields:
            bad["bad field count"].append(i)
            continue
        if any(p != p.strip() for p in parts):
            bad["surrounding whitespace"].append(i)
            continue
        rows.append(parts)
    errors.extend(_row_errors(kind, bad))
    return errors, rows


def _summarize(items: list[str]) -> str:
    return ",".join(items[:_MAX_EXAMPLES])


def _row_errors(kind: str, by_key: Mapping[str, list[int]]) -> list[str]:
    return [
        f"{kind}: {key} count={len(rows)} rows={_summarize([str(r) for r in rows])}"
        for key, rows in by_key.items()
        if rows
    ]


def validate_composition_file(
    path: Path, ref: SubmissionReference, spec: CompetitionSpec | None = None
) -> tuple[list[str], dict[str, float], int]:
    """composition.csv를 검사해 (오류 목록, 그룹별 용량, 데이터 행 수)를 반환한다."""
    cfg = _resolve_spec(spec)
    kind = "composition"
    lines, early, _ = _read_text_lines(path, kind)
    if lines is None:
        return early, {}, 0
    errors = list(early)
    errors.extend(_blank_line_errors(lines, kind))
    field_errors, rows = _field_errors(lines, kind, 2, _COMPOSITION_HEADER)
    errors.extend(field_errors)
    expected = cfg.n_sites + 1
    if len(lines) != expected:
        errors.append(f"{kind}: row count={len(lines)} expected={expected}")
    counts = Counter(r[0] for r in rows)
    found = set(counts)
    expected_ids = set(ref.site_capacity)
    missing = sorted(expected_ids - found)
    unknown = sorted(found - expected_ids)
    dupes = sorted(s for s, c in counts.items() if c > 1)
    if missing:
        errors.append(f"{kind}: missing site count={len(missing)} first={_summarize(missing)}")
    if unknown:
        errors.append(f"{kind}: unknown site count={len(unknown)} first={_summarize(unknown)}")
    if dupes:
        errors.append(f"{kind}: duplicate site count={len(dupes)} first={_summarize(dupes)}")
    bad_groups = sorted({r[1] for r in rows if r[1] not in cfg.groups})
    if bad_groups:
        errors.append(f"{kind}: invalid group first={_summarize(bad_groups)}")
    cap_by_group: dict[str, Decimal] = {g: Decimal(0) for g in cfg.groups}
    for sid, grp in rows:
        if sid in ref.site_capacity and grp in cap_by_group:
            cap_by_group[grp] += ref.site_capacity[sid]
    threshold = Decimal(str(cfg.min_group_capacity_kw))
    errors.extend(
        f"{kind}: under capacity group={grp} cap={cap_by_group[grp]}"
        for grp in cfg.groups
        if cap_by_group[grp] < threshold
    )
    caps = {g: float(v) for g, v in cap_by_group.items()}
    n_rows = sum(1 for line in lines[1:] if line.strip() != "")
    return errors, caps, n_rows


def _parse_forecast_ts(raw: str) -> pd.Timestamp | None:
    if not raw.endswith(_KST_OFFSET):
        return None
    try:
        return pd.Timestamp(raw).tz_convert(_KST)
    except (ValueError, TypeError):
        return None


def validate_forecast_file(
    path: Path, ref: SubmissionReference, spec: CompetitionSpec | None = None
) -> tuple[list[str], list[str], dict[str, float], int]:
    """forecast.csv를 검사해 (오류, 경고, 그룹별 예측 합계, 데이터 행 수)를 반환한다."""
    cfg = _resolve_spec(spec)
    kind = "forecast"
    lines, early, _ = _read_text_lines(path, kind)
    if lines is None:
        return early, [], {}, 0
    errors = list(early)
    warnings: list[str] = []
    errors.extend(_blank_line_errors(lines, kind))
    field_errors, rows = _field_errors(lines, kind, 3, _FORECAST_HEADER)
    errors.extend(field_errors)
    expected = len(cfg.groups) * cfg.n_eval_hours + 1
    if len(lines) != expected:
        errors.append(f"{kind}: row count={len(lines)} expected={expected}")
    groups = cfg.groups
    ref_set = set(ref.score_index)
    ref_strings = set(ref.score_ts)
    per_group_ts: dict[str, set[pd.Timestamp]] = {g: set() for g in groups}
    bad: dict[str, list[int]] = {
        key: []
        for key in (
            "invalid group",
            "bad ts",
            "non-hourly ts",
            "duplicate pair",
            "bad forecast",
            "non-finite forecast",
            "negative forecast",
        )
    }
    format_diff = 0
    totals: dict[str, float] = dict.fromkeys(groups, 0.0)
    all_zero: dict[str, bool] = dict.fromkeys(groups, True)
    for i, parts in enumerate(rows, start=2):
        grp, raw_ts, raw_val = parts[0], parts[1], parts[2]
        if grp not in groups:
            bad["invalid group"].append(i)
            continue
        ts = _parse_forecast_ts(raw_ts)
        if ts is None:
            bad["bad ts"].append(i)
            continue
        if ts != ts.floor("h"):
            bad["non-hourly ts"].append(i)
            continue
        # 같은 순간을 다른 문자열 형식으로 쓴 행도 중복이므로 파싱된 순간으로 판정한다.
        if ts in per_group_ts[grp]:
            bad["duplicate pair"].append(i)
            continue
        per_group_ts[grp].add(ts)
        if raw_ts not in ref_strings and ts in ref_set:
            format_diff += 1
        # float()는 "1_000" 같은 파이썬 리터럴 표기도 받으므로 CSV 수치로 보지 않는다.
        try:
            val = float(raw_val)
        except ValueError:
            val = None
        if val is None or "_" in raw_val:
            bad["bad forecast"].append(i)
            continue
        if not math.isfinite(val):
            bad["non-finite forecast"].append(i)
            continue
        if val < 0:
            bad["negative forecast"].append(i)
            continue
        totals[grp] += val
        if val != 0:
            all_zero[grp] = False
    errors.extend(_row_errors(kind, bad))
    for grp in groups:
        want = ref_set
        got = per_group_ts[grp]
        miss = sorted(want - got)
        extra = sorted(got - want)
        if miss:
            first = miss[0].strftime("%Y-%m-%d %H:%M:%S+09:00")
            errors.append(f"{kind}: missing ts group={grp} count={len(miss)} first={first}")
        if extra:
            first = extra[0].strftime("%Y-%m-%d %H:%M:%S+09:00")
            errors.append(f"{kind}: extra ts group={grp} count={len(extra)} first={first}")
    if format_diff:
        warnings.append(f"{kind}: non-canonical ts format count={format_diff}")
    warnings.extend(
        f"{kind}: all-zero forecast group={grp}"
        for grp in groups
        if rows and all_zero[grp] and per_group_ts[grp]
    )
    n_rows = sum(1 for line in lines[1:] if line.strip() != "")
    return errors, warnings, totals, n_rows


def _exceed_warnings(
    path: Path, caps: Mapping[str, float], groups: tuple[str, ...]
) -> list[str]:
    text = path.read_text(encoding="utf-8")
    warnings: list[str] = []
    for grp in groups:
        cap = caps.get(grp, 0.0)
        over = sum(
            1
            for line in text.splitlines()[1:]
            if line.split(",")[0] == grp and float(line.split(",")[2]) > cap
        )
        if over:
            warnings.append(f"forecast: over capacity group={grp} count={over}")
    return warnings


def validate_submission(
    composition_path: Path,
    forecast_path: Path,
    ref: SubmissionReference | None = None,
    spec: CompetitionSpec | None = None,
) -> ValidationReport:
    """두 제출 파일을 검사해 리포트를 반환한다. 예외 대신 오류를 수집한다(기준 파일 오류만 예외)."""
    cfg = _resolve_spec(spec)
    reference = ref if ref is not None else load_reference(spec=cfg)
    comp_errors, caps, n_comp = validate_composition_file(composition_path, reference, cfg)
    fc_errors, fc_warnings, totals, n_fc = validate_forecast_file(forecast_path, reference, cfg)
    warnings = list(fc_warnings)
    if not comp_errors and not fc_errors:
        warnings.extend(_exceed_warnings(forecast_path, caps, cfg.groups))
    report = ValidationReport(
        errors=tuple(comp_errors + fc_errors),
        warnings=tuple(warnings),
        group_capacity_kw=dict(caps),
        group_forecast_total_kwh=dict(totals),
        n_composition_rows=n_comp,
        n_forecast_rows=n_fc,
    )
    cap_str = ",".join(f"{g}:{caps.get(g, 0.0):.1f}" for g in cfg.groups)
    logger.info(
        "[SUBMIT] validate ok=%s errors=%d warnings=%d cap=%s",
        report.ok,
        len(report.errors),
        len(report.warnings),
        cap_str,
    )
    for err in report.errors:
        logger.warning("[SUBMIT] %s", err)
    return report


def assert_valid(report: ValidationReport) -> None:
    """리포트에 오류가 있으면 모든 오류를 담아 SubmissionError를 던진다."""
    if not report.ok:
        raise SubmissionError("; ".join(report.errors))
