
import numpy as np
import pandas as pd

from core import (
    prepare_input_data,
    make_feature_row,
    fit_gpr,
    predict_gpr_residual,
    sequential_gpr_correction,
    calculate_metrics,
)


def make_quarterly_df(n=48):
    rows = []
    y = 2014
    q = 1
    for i in range(n):
        value = 4_000_000 + (q - 2.5) * 150_000 + i * 5_000
        rows.append({"year": y, "quarter": q, "production_mt": value})
        q += 1
        if q == 5:
            q = 1
            y += 1
    return pd.DataFrame(rows)


def test_prepare_data():
    p = prepare_input_data(make_quarterly_df())
    assert len(p) == 48
    assert p.iloc[0]["quarter"] == 1
    assert p.iloc[-1]["time_index"] == 47


def test_feature_row():
    history = np.arange(1, 21, dtype=float) * 100.0
    r = make_feature_row(history, raw_forecast=2200, target_quarter=1, horizon=1)
    assert "seasonal_lag4" in r
    assert r["last_value"] == 2000.0


def test_gpr_learns_simple_residual_pattern():
    rows = []
    for i in range(30):
        h = (i % 4) + 1
        q = (i % 4) + 1
        hist = np.arange(1, 21, dtype=float) * 10 + i
        raw = 1000 + i * 5
        feat = make_feature_row(hist, raw, q, h)
        rows.append(
            {
                "origin_index": 40 + i,
                "target_index": 40 + i,
                "origin_period": "x",
                "target_period": "x",
                "target_quarter": q,
                "horizon": h,
                "actual": raw + 100,
                "raw_forecast": raw,
                "residual": 100.0,
                **feat,
            }
        )
    df = pd.DataFrame(rows)
    model = fit_gpr(df.iloc[:24])
    mean, std = predict_gpr_residual(model, df.iloc[24:25])
    assert np.isfinite(mean[0])
    assert np.isfinite(std[0])


def test_sequential_correction_and_metrics():
    rows = []
    for origin in range(20, 36):
        q = (origin % 4) + 1
        raw = 1000 + origin
        actual = raw + 50
        hist = np.arange(1, 21, dtype=float) + origin
        feat = make_feature_row(hist, raw, q, 1)
        rows.append(
            {
                "origin_index": origin,
                "target_index": origin,
                "origin_period": str(origin),
                "target_period": str(origin),
                "target_quarter": q,
                "horizon": 1,
                "actual": actual,
                "raw_forecast": raw,
                "residual": actual - raw,
                **feat,
            }
        )
    df = pd.DataFrame(rows)
    out = sequential_gpr_correction(df, min_train_rows=8)
    assert out["corrected_forecast"].notna().sum() > 0

    series = np.arange(1, 80, dtype=float) * 100
    metrics = calculate_metrics(out, series)
    assert set(metrics["Method"]) == {"Raw Chronos", "Chronos + GPR"}


if __name__ == "__main__":
    test_prepare_data()
    test_feature_row()
    test_gpr_learns_simple_residual_pattern()
    test_sequential_correction_and_metrics()
    print("All lightweight core tests passed.")
