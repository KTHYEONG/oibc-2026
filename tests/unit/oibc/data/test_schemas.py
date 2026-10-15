import pandas as pd
import pytest

from src.oibc.data.schemas import GENERATION, SchemaError, validate_frame

KST = "Asia/Seoul"
DT = "datetime64[us, Asia/Seoul]"


def _valid() -> pd.DataFrame:
    ts = pd.Series(
        pd.date_range("2025-06-01 01:00", periods=2, freq="h", tz=KST), dtype=DT
    )
    return pd.DataFrame(
        {
            "site_id": pd.Series(["PV001", "PV001"], dtype="string"),
            "ts": ts,
            "op_date": pd.Series(
                [pd.Timestamp("2025-06-01", tz=KST)] * 2, dtype=DT
            ),
            "energy": pd.Series([1.0, 2.0], dtype="float32"),
            "dq_negative": pd.Series([False, False], dtype="bool"),
            "dq_over_cap": pd.Series([False, False], dtype="bool"),
            "dq_night_gen": pd.Series([False, False], dtype="bool"),
            "dq_zero_day": pd.Series([False, False], dtype="bool"),
        }
    )


def test_valid_frame_passes() -> None:
    validate_frame(_valid(), GENERATION)


def test_missing_column_rejected() -> None:
    df = _valid().drop(columns=["energy"])
    with pytest.raises(SchemaError, match="energy"):
        validate_frame(df, GENERATION)


def test_wrong_dtype_rejected() -> None:
    df = _valid()
    df["energy"] = df["energy"].astype("float64")
    with pytest.raises(SchemaError, match="energy"):
        validate_frame(df, GENERATION)


def test_duplicate_key_rejected() -> None:
    df = pd.concat([_valid().iloc[[0]], _valid().iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="generation"):
        validate_frame(df, GENERATION)


def test_unsorted_key_rejected() -> None:
    df = _valid().iloc[::-1].reset_index(drop=True)
    with pytest.raises(SchemaError, match="generation"):
        validate_frame(df, GENERATION)


def test_missing_key_rejected() -> None:
    df = _valid()
    df.loc[0, "site_id"] = pd.NA
    with pytest.raises(SchemaError, match="site_id"):
        validate_frame(df, GENERATION)


@pytest.mark.parametrize("period", ["test", None])
def test_invalid_period_rejected(period: str | None) -> None:
    from src.oibc.data.schemas import SMP

    frame = _valid()[["ts", "op_date"]].copy()
    frame["smp"] = pd.Series([1.0, 2.0], dtype="float32")
    frame["k"] = pd.Series([1, 1], dtype="int8")
    frame["period"] = pd.Series(["train", period], dtype="string")
    with pytest.raises(SchemaError, match="period"):
        validate_frame(frame, SMP)
