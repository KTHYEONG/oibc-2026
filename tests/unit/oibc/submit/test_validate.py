import codecs
from pathlib import Path

import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec, DataPaths
from src.oibc.submit import validate as vmod
from src.oibc.submit.validate import (
    SubmissionError,
    assert_valid,
    load_reference,
    validate_composition_file,
    validate_forecast_file,
    validate_submission,
)

_TS = (
    "2025-06-01 01:00:00+09:00",
    "2025-06-01 02:00:00+09:00",
    "2025-06-01 03:00:00+09:00",
)
_CAPS = {"PV001": "100", "PV002": "100", "PV003": "99.99", "PV004": "100.01"}


def _spec() -> CompetitionSpec:
    return CompetitionSpec(
        n_sites=4,
        min_group_capacity_kw=200.0,
        train_end=pd.Timestamp("2025-06-01 00:00", tz="Asia/Seoul"),
        eval_start=pd.Timestamp("2025-06-01 01:00", tz="Asia/Seoul"),
        eval_end=pd.Timestamp("2025-06-01 03:00", tz="Asia/Seoul"),
        n_eval_hours=3,
    )


@pytest.fixture
def ref(tmp_path: Path) -> vmod.SubmissionReference:
    spec = _spec()
    (tmp_path / "sites.csv").write_text(
        "site_id,current_group,capacity_kw\n"
        + "".join(f"{sid},A,{cap}\n" for sid, cap in _CAPS.items()),
        encoding="utf-8",
    )
    (tmp_path / "smp.csv").write_text(
        "ts,smp,k\n" + "".join(f"{t},100.0,1\n" for t in _TS), encoding="utf-8"
    )
    return load_reference(DataPaths(raw_dir=tmp_path), spec)


@pytest.fixture
def spec() -> CompetitionSpec:
    return _spec()


def _comp(group_of: dict[str, str]) -> str:
    lines = ["site_id,group"]
    lines.extend(f"{sid},{group_of[sid]}" for sid in sorted(group_of))
    return "\n".join(lines) + "\n"


def _forecast(value: float = 10.0, groups: tuple[str, str] = ("G1", "G2")) -> str:
    lines = ["group,ts,forecast_kwh"]
    lines.extend(f"{g},{t},{value}" for g in groups for t in _TS)
    return "\n".join(lines) + "\n"


def _write(tmp_path: Path, comp: str, fc: str) -> tuple[Path, Path]:
    cp, fp = tmp_path / "composition.csv", tmp_path / "forecast.csv"
    cp.write_text(comp, encoding="utf-8")
    fp.write_text(fc, encoding="utf-8")
    return cp, fp


_VALID_GROUPS = {"PV001": "G1", "PV002": "G1", "PV003": "G2", "PV004": "G2"}


def test_valid_submission_passes(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), _forecast())
    report = validate_submission(cp, fp, ref, spec)
    assert report.ok and not report.errors and not report.warnings
    assert pytest.approx(sum(report.group_capacity_kw.values())) == 400.0


def test_forecast_header_strict(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), _forecast())
    for bad in ("group,ts,forecast", "Group,ts,forecast_kwh"):
        text = fp.read_text(encoding="utf-8").replace("group,ts,forecast_kwh", bad)
        bad_fp = tmp_path / "bad.csv"
        bad_fp.write_text(text, encoding="utf-8")
        report = validate_submission(cp, bad_fp, ref, spec)
        assert not report.ok and any("bad header" in e for e in report.errors)


def test_composition_bom_rejected(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), _forecast())
    cp.write_bytes(codecs.BOM_UTF8 + cp.read_bytes())
    report = validate_submission(cp, fp, ref, spec)
    assert not report.ok and any("bom" in e for e in report.errors)


def test_site_missing_duplicate_unknown(
    tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec
) -> None:
    fc = _forecast()
    rows = _comp(_VALID_GROUPS).splitlines()
    missing = "\n".join([rows[0], *rows[1:3]]) + "\n"
    dupe = "\n".join([*rows, rows[1]]) + "\n"
    unknown = "\n".join([*rows, "PV999,G1"]) + "\n"
    for text, key in ((missing, "missing site"), (dupe, "duplicate site"), (unknown, "unknown site")):
        cp = tmp_path / f"c_{key.split()[0]}.csv"
        cp.write_text(text, encoding="utf-8")
        fp = tmp_path / "f.csv"
        fp.write_text(fc, encoding="utf-8")
        report = validate_submission(cp, fp, ref, spec)
        assert any(key in e for e in report.errors), key
        assert any("row count" in e for e in report.errors), key


def test_capacity_boundary_exact(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    exact = {"PV001": "G1", "PV002": "G1", "PV003": "G2", "PV004": "G2"}
    under = {"PV001": "G1", "PV003": "G1", "PV002": "G2", "PV004": "G2"}
    cp_ok, fp = _write(tmp_path, _comp(exact), _forecast())
    assert validate_submission(cp_ok, fp, ref, spec).ok
    cp_bad = tmp_path / "under.csv"
    cp_bad.write_text(_comp(under), encoding="utf-8")
    report = validate_submission(cp_bad, fp, ref, spec)
    assert not report.ok and any("under capacity" in e for e in report.errors)


def test_invalid_group_name(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    groups = dict(_VALID_GROUPS)
    groups["PV001"] = "A"
    cp, fp = _write(tmp_path, _comp(groups), _forecast())
    report = validate_submission(cp, fp, ref, spec)
    assert not report.ok and any("invalid group" in e for e in report.errors)


def test_naive_ts_rejected(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), _forecast())
    text = fp.read_text(encoding="utf-8").replace(_TS[0], "2025-06-01 01:00:00", 1)
    fp.write_text(text, encoding="utf-8")
    report = validate_submission(cp, fp, ref, spec)
    assert not report.ok and any("bad ts" in e for e in report.errors)


def test_other_offset_rejected(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), _forecast())
    text = fp.read_text(encoding="utf-8").replace(_TS[0], "2025-06-01 01:00:00+00:00", 1)
    fp.write_text(text, encoding="utf-8")
    report = validate_submission(cp, fp, ref, spec)
    assert not report.ok and any("bad ts" in e for e in report.errors)


def test_ts_missing_extra_duplicate(
    tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec
) -> None:
    cp = tmp_path / "c.csv"
    cp.write_text(_comp(_VALID_GROUPS), encoding="utf-8")
    base = _forecast().splitlines()
    missing = "\n".join([base[0], *[ln for ln in base[1:] if ln != f"G1,{_TS[0]},10.0"]]) + "\n"
    extra = "\n".join([*base, "G1,2025-06-01 04:00:00+09:00,10.0"]) + "\n"
    dupe = "\n".join([*base, "G2,2025-06-01 01:00:00+09:00,10.0"]) + "\n"
    for text, key in ((missing, "missing ts"), (extra, "extra ts"), (dupe, "duplicate pair")):
        fp = tmp_path / f"f_{key.split()[0]}.csv"
        fp.write_text(text, encoding="utf-8")
        report = validate_submission(cp, fp, ref, spec)
        assert any(key in e for e in report.errors), key


def test_invalid_forecast_values(
    tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec
) -> None:
    cp = tmp_path / "c.csv"
    cp.write_text(_comp(_VALID_GROUPS), encoding="utf-8")
    for bad in ("nan", "inf", "-1", ""):
        text = _forecast().replace("G1,2025-06-01 01:00:00+09:00,10.0", f"G1,{_TS[0]},{bad}")
        fp = tmp_path / f"v_{bad or 'empty'}.csv"
        fp.write_text(text, encoding="utf-8")
        report = validate_submission(cp, fp, ref, spec)
        assert not report.ok, bad


def test_negative_zero_allowed(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), _forecast())
    fp.write_text(fp.read_text(encoding="utf-8").replace(",10.0\n", ",-0.0\n", 1), encoding="utf-8")
    report = validate_submission(cp, fp, ref, spec)
    assert report.ok


def test_format_difference_warns(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), _forecast())
    fp.write_text(fp.read_text(encoding="utf-8").replace("2025-06-01 ", "2025-06-01T"), encoding="utf-8")
    report = validate_submission(cp, fp, ref, spec)
    assert report.ok and any("non-canonical" in w for w in report.warnings)


def test_over_capacity_warns(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), _forecast())
    fp.write_text(fp.read_text(encoding="utf-8").replace(",10.0\n", ",500.0\n", 1), encoding="utf-8")
    report = validate_submission(cp, fp, ref, spec)
    assert report.ok and any("over capacity" in w for w in report.warnings)


def test_row_order_irrelevant(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), _forecast())
    rows = fp.read_text(encoding="utf-8").splitlines()
    fp.write_text("\n".join([rows[0], *reversed(rows[1:])]) + "\n", encoding="utf-8")
    assert validate_submission(cp, fp, ref, spec).ok


def test_all_errors_collected(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp = tmp_path / "c.csv"
    cp.write_text(_comp(_VALID_GROUPS), encoding="utf-8")
    fp = tmp_path / "f.csv"
    fp.write_text("group,ts,forecast\nG1,2025-06-01 01:00:00+09:00,-1\n", encoding="utf-8")
    report = validate_submission(cp, fp, ref, spec)
    assert sum(1 for e in report.errors if "forecast" in e) >= 2


def test_assert_valid_raises_with_all_errors(
    tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec
) -> None:
    cp = tmp_path / "c.csv"
    cp.write_text("site_id,group\n", encoding="utf-8")
    fp = tmp_path / "f.csv"
    fp.write_text("group,ts,forecast_kwh\n", encoding="utf-8")
    report = validate_submission(cp, fp, ref, spec)
    assert not report.ok
    with pytest.raises(SubmissionError) as exc:
        assert_valid(report)
    for err in report.errors:
        assert err in str(exc.value)


def test_reference_violation(tmp_path: Path, spec: CompetitionSpec) -> None:
    (tmp_path / "sites.csv").write_text(
        "site_id,capacity_kw\nPV001,100\nPV002,100\nPV003,100\nPV004,100\n", encoding="utf-8"
    )
    (tmp_path / "smp.csv").write_text("ts,smp,k\n2025-06-01 01:00:00+09:00,1,1\n", encoding="utf-8")
    with pytest.raises(SubmissionError):
        load_reference(DataPaths(raw_dir=tmp_path), spec)


_GOOD_SITES = "site_id,capacity_kw\nPV001,100\nPV002,100\nPV003,99.99\nPV004,100.01\n"
_GOOD_SMP = "ts,smp,k\n" + "".join(f"{t},100.0,1\n" for t in _TS)


def _raw_dir(tmp_path: Path, sites: str | None, smp: str | None) -> Path:
    raw = tmp_path / "raw_case"
    raw.mkdir(exist_ok=True)
    if sites is not None:
        (raw / "sites.csv").write_text(sites, encoding="utf-8")
    if smp is not None:
        (raw / "smp.csv").write_text(smp, encoding="utf-8")
    return raw


@pytest.mark.parametrize(
    ("sites", "smp"),
    [
        (None, _GOOD_SMP),
        (_GOOD_SITES, None),
        ("site_id\nPV001\n", _GOOD_SMP),
        ("site_id,capacity_kw\nPV001,100\nPV002,100\n", _GOOD_SMP),
        ("site_id,capacity_kw\nPV001,100\nPV001,100\nPV003,100\nPV004,100\n", _GOOD_SMP),
        ("site_id,capacity_kw\nPV001,abc\nPV002,100\nPV003,100\nPV004,100\n", _GOOD_SMP),
        ("site_id,capacity_kw\nPV001,0\nPV002,100\nPV003,100\nPV004,100\n", _GOOD_SMP),
        (_GOOD_SITES, "smp,k\n2025-06-01 01:00:00+09:00,1\n"),
        (_GOOD_SITES, "ts,smp,k\nnot-a-time,1,1\n2025-06-01 02:00:00+09:00,1,1\n2025-06-01 03:00:00+09:00,1,1\n"),
        (_GOOD_SITES, "ts,smp,k\n2025-06-01 01:00:00,1,1\n2025-06-01 02:00:00,1,1\n2025-06-01 03:00:00,1,1\n"),
        (_GOOD_SITES, f"ts,smp,k\n{_TS[0]},1,1\n{_TS[0]},1,1\n{_TS[1]},1,1\n"),
        (_GOOD_SITES, f"ts,smp,k\n{_TS[1]},1,1\n{_TS[0]},1,1\n{_TS[2]},1,1\n"),
        (_GOOD_SITES, "ts,smp,k\n2025-06-01 02:00:00+09:00,1,1\n2025-06-01 03:00:00+09:00,1,1\n2025-06-01 04:00:00+09:00,1,1\n"),
    ],
)
def test_reference_contracts_rejected(
    tmp_path: Path, spec: CompetitionSpec, sites: str | None, smp: str | None
) -> None:
    raw = _raw_dir(tmp_path, sites, smp)
    with pytest.raises(SubmissionError):
        load_reference(DataPaths(raw_dir=raw), spec)


def test_missing_files_reported(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    comp_errors, _, _ = validate_composition_file(tmp_path / "nope.csv", ref, spec)
    assert any("file not found" in e for e in comp_errors)
    fc_errors, _, _, _ = validate_forecast_file(tmp_path / "nope.csv", ref, spec)
    assert any("file not found" in e for e in fc_errors)


def test_directory_path_rejected(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    comp_errors, _, _ = validate_composition_file(tmp_path, ref, spec)
    assert any("cannot read" in e for e in comp_errors)


def test_non_utf8_rejected(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp = tmp_path / "c.csv"
    cp.write_bytes(b"site_id,group\nPV001,G1\xff\n")
    comp_errors, _, _ = validate_composition_file(cp, ref, spec)
    assert any("not utf-8" in e for e in comp_errors)


def test_empty_file_reports_header(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp = tmp_path / "c.csv"
    cp.write_text("", encoding="utf-8")
    report = validate_submission(cp, tmp_path / "nope.csv", ref, spec)
    assert any("missing header" in e for e in report.errors)


def test_blank_line_rejected(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), _forecast())
    cp.write_text(cp.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    report = validate_submission(cp, fp, ref, spec)
    assert any("blank line" in e for e in report.errors)


def test_bad_field_count_and_whitespace(
    tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec
) -> None:
    cp = tmp_path / "c.csv"
    cp.write_text("site_id,group\nPV001\nPV002 ,G1\nPV003,G2\nPV004,G2\n", encoding="utf-8")
    fp = tmp_path / "f.csv"
    fp.write_text(_forecast(), encoding="utf-8")
    report = validate_submission(cp, fp, ref, spec)
    assert any("bad field count" in e for e in report.errors)
    assert any("whitespace" in e for e in report.errors)


def test_forecast_invalid_group_and_shapes(
    tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec
) -> None:
    cp = tmp_path / "c.csv"
    cp.write_text(_comp(_VALID_GROUPS), encoding="utf-8")
    base = _forecast().splitlines()
    bad_group = "\n".join([base[0], "A,2025-06-01 01:00:00+09:00,10.0", *base[2:]]) + "\n"
    fp = tmp_path / "f.csv"
    fp.write_text(bad_group, encoding="utf-8")
    assert any("invalid group" in e for e in validate_submission(cp, fp, ref, spec).errors)
    half_hour = "\n".join([base[0], "G1,2025-06-01 01:30:00+09:00,10.0", *base[2:]]) + "\n"
    fp.write_text(half_hour, encoding="utf-8")
    assert any("non-hourly" in e for e in validate_submission(cp, fp, ref, spec).errors)
    garbage = "\n".join([base[0], "G1,not-a-time+09:00,10.0", *base[2:]]) + "\n"
    fp.write_text(garbage, encoding="utf-8")
    assert any("bad ts" in e for e in validate_submission(cp, fp, ref, spec).errors)


def test_all_zero_forecast_warns(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), _forecast())
    lines = fp.read_text(encoding="utf-8").splitlines()
    zeroed = [lines[0], *[ln.rsplit(",", 1)[0] + ",0.0" if ln.startswith("G1,") else ln for ln in lines[1:]]]
    fp.write_text("\n".join(zeroed) + "\n", encoding="utf-8")
    report = validate_submission(cp, fp, ref, spec)
    assert report.ok and any("all-zero" in w for w in report.warnings)


def test_same_instant_other_format_is_duplicate(
    tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec
) -> None:
    cp = tmp_path / "c.csv"
    cp.write_text(_comp(_VALID_GROUPS), encoding="utf-8")
    fp = tmp_path / "f.csv"
    fp.write_text(_forecast() + "G1,2025-06-01T01:00:00+09:00,10.0\n", encoding="utf-8")
    report = validate_submission(cp, fp, ref, spec)
    assert any("duplicate pair" in e for e in report.errors)


def test_underscore_numeral_rejected(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    text = _forecast().replace(f"G1,{_TS[0]},10.0", f"G1,{_TS[0]},1_0")
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), text)
    report = validate_submission(cp, fp, ref, spec)
    assert any("bad forecast" in e for e in report.errors)


def test_row_errors_aggregated(tmp_path: Path, ref: vmod.SubmissionReference, spec: CompetitionSpec) -> None:
    lines = ["group,ts,forecast_kwh"]
    lines.extend(f"G1,{t},-1" for t in _TS)
    lines.extend(f"G2,{t},-1" for t in _TS)
    cp, fp = _write(tmp_path, _comp(_VALID_GROUPS), "\n".join(lines) + "\n")
    report = validate_submission(cp, fp, ref, spec)
    negatives = [e for e in report.errors if "negative forecast" in e]
    assert negatives == ["forecast: negative forecast count=6 rows=2,3,4,5,6"]
