
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "quarter_sin",
    "quarter_cos",
    "horizon",
    "raw_forecast",
    "last_value",
    "seasonal_lag4",
    "mean_last4",
    "std_last4",
    "recent_change",
]


def prepare_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize a quarterly palay dataset.

    Required columns:
      year          integer year, e.g. 2024
      quarter       1-4 or Q1-Q4
      production_mt numeric production value in metric tons
    """
    required = {"year", "quarter", "production_mt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required column(s): " + ", ".join(sorted(missing))
            + ". Required columns are: year, quarter, production_mt."
        )

    out = df.loc[:, ["year", "quarter", "production_mt"]].copy()

    def parse_quarter(v):
        s = str(v).strip().upper()
        if s.startswith("Q"):
            s = s[1:]
        try:
            q = int(float(s))
        except Exception as exc:
            raise ValueError(f"Invalid quarter value: {v!r}") from exc
        if q not in (1, 2, 3, 4):
            raise ValueError(f"Quarter must be 1, 2, 3, or 4. Got {v!r}.")
        return q

    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    out["quarter"] = out["quarter"].map(parse_quarter).astype(int)
    out["production_mt"] = pd.to_numeric(out["production_mt"], errors="raise").astype(float)

    if out["production_mt"].isna().any():
        raise ValueError("production_mt contains missing values.")
    if (out["production_mt"] <= 0).any():
        raise ValueError("production_mt must contain positive values.")

    out = out.sort_values(["year", "quarter"]).reset_index(drop=True)

    if out.duplicated(["year", "quarter"]).any():
        dup = out.loc[out.duplicated(["year", "quarter"], keep=False), ["year", "quarter"]]
        raise ValueError(f"Duplicate year-quarter rows found:\n{dup.to_string(index=False)}")

    serial = out["year"] * 4 + (out["quarter"] - 1)
    gaps = np.diff(serial.to_numpy())
    if len(gaps) and not np.all(gaps == 1):
        raise ValueError(
            "The dataset must contain consecutive quarterly observations with no missing quarter. "
            "Fill or resolve missing quarters before running the POC."
        )

    out["period"] = out.apply(lambda r: f"{int(r.year)} Q{int(r.quarter)}", axis=1)
    out["time_index"] = np.arange(len(out))
    return out


def load_chronos_pipeline(model_name: str = "amazon/chronos-t5-small"):
    """
    Load the official Chronos-T5 pipeline.
    The import is kept inside this function so the rest of the project can
    still be tested without chronos-forecasting installed.
    """
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Chronos is not installed. Run: pip install -r requirements.txt"
        ) from exc

    use_cuda = bool(torch.cuda.is_available())
    device_map = "cuda" if use_cuda else "cpu"
    torch_dtype = torch.bfloat16 if use_cuda else torch.float32

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return pipeline, device_map


def chronos_point_forecast(pipeline, history: np.ndarray, prediction_length: int) -> np.ndarray:
    """Return the median Chronos forecast for prediction_length future quarters."""
    import torch

    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 1:
        raise ValueError("history must be a one-dimensional array.")
    if len(history) < 8:
        raise ValueError("At least 8 historical quarters are required for the POC.")

    context = torch.tensor(history, dtype=torch.float32)

    with torch.no_grad():
        forecast = pipeline.predict(
            context,
            int(prediction_length),
        )

    # Official Chronos-T5 output shape:
    # [num_series, num_samples, prediction_length]
    if hasattr(forecast, "detach"):
        samples = forecast[0].detach().cpu().numpy()
    else:
        samples = np.asarray(forecast)[0]

    return np.median(samples, axis=0).astype(float)


def _quarter_encoding(q: int) -> Tuple[float, float]:
    angle = 2.0 * np.pi * (int(q) - 1) / 4.0
    return float(np.sin(angle)), float(np.cos(angle))


def make_feature_row(
    history: np.ndarray,
    raw_forecast: float,
    target_quarter: int,
    horizon: int,
) -> dict:
    history = np.asarray(history, dtype=float)
    if len(history) < 4:
        raise ValueError("At least four quarters are required to build residual features.")

    qsin, qcos = _quarter_encoding(target_quarter)
    last4 = history[-4:]
    return {
        "quarter_sin": qsin,
        "quarter_cos": qcos,
        "horizon": float(horizon),
        "raw_forecast": float(raw_forecast),
        "last_value": float(history[-1]),
        "seasonal_lag4": float(history[-4]),
        "mean_last4": float(np.mean(last4)),
        "std_last4": float(np.std(last4, ddof=0)),
        "recent_change": float(history[-1] - history[-2]) if len(history) >= 2 else 0.0,
    }


def build_residual_dataset(
    prepared: pd.DataFrame,
    pipeline,
    max_horizon: int = 4,
    n_origins: int = 20,
    min_context: int = 32,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Create historical Chronos forecasts and residuals.

    origin_index means the number of observations known when a forecast is made.
    For example, origin_index=40 uses observations 0..39 as context and predicts 40 onward.
    """
    values = prepared["production_mt"].to_numpy(dtype=float)
    quarters = prepared["quarter"].to_numpy(dtype=int)
    n = len(values)

    if n < min_context + 5:
        raise ValueError(
            f"Dataset is too short for the selected setup. Need at least {min_context + 5} rows."
        )

    possible_origins = list(range(min_context, n))
    if n_origins is not None:
        possible_origins = possible_origins[-int(n_origins):]

    rows = []
    total = len(possible_origins)

    for count, origin in enumerate(possible_origins, start=1):
        pred_len = min(int(max_horizon), n - origin)
        if pred_len <= 0:
            continue

        history = values[:origin]
        raw_preds = chronos_point_forecast(pipeline, history, pred_len)

        for h in range(1, pred_len + 1):
            target_index = origin + h - 1
            target_quarter = int(quarters[target_index])
            raw = float(raw_preds[h - 1])
            actual = float(values[target_index])
            feat = make_feature_row(history, raw, target_quarter, h)
            rows.append(
                {
                    "origin_index": int(origin),
                    "target_index": int(target_index),
                    "origin_period": prepared.iloc[origin - 1]["period"],
                    "target_period": prepared.iloc[target_index]["period"],
                    "target_quarter": target_quarter,
                    "horizon": int(h),
                    "actual": actual,
                    "raw_forecast": raw,
                    "residual": actual - raw,
                    **feat,
                }
            )

        if progress_callback is not None:
            progress_callback(count, total)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No historical forecasts could be generated.")
    return out.sort_values(["origin_index", "horizon"]).reset_index(drop=True)


def make_gpr_model(random_state: int = 42) -> Pipeline:
    """
    Gaussian Process Regression residual model.

    Scaling is important because forecast values are measured in metric tons
    while quarter/horizon features are small numbers.
    """
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=np.ones(len(FEATURE_COLUMNS)), length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=1e-6,
        n_restarts_optimizer=1,
        random_state=random_state,
    )

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("gpr", gpr),
        ]
    )


def fit_gpr(residual_rows: pd.DataFrame) -> Pipeline:
    if len(residual_rows) < 8:
        raise ValueError("At least 8 residual examples are required to fit the GPR correction layer.")
    model = make_gpr_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(residual_rows[FEATURE_COLUMNS], residual_rows["residual"])
    return model


def predict_gpr_residual(model: Pipeline, feature_rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X_scaled = model.named_steps["scale"].transform(feature_rows[FEATURE_COLUMNS])
    gpr = model.named_steps["gpr"]
    mean, std = gpr.predict(X_scaled, return_std=True)
    return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def sequential_gpr_correction(
    residual_df: pd.DataFrame,
    min_train_rows: int = 12,
) -> pd.DataFrame:
    """
    Time-safe historical evaluation.

    At forecast origin t, GPR may only train on residuals whose target_index < t,
    because only those actual outcomes would already be known.
    """
    out = residual_df.copy()
    out["gpr_correction"] = np.nan
    out["gpr_std"] = np.nan
    out["corrected_forecast"] = np.nan
    out["gpr_train_rows"] = 0

    for origin in sorted(out["origin_index"].unique()):
        train = out[out["target_index"] < origin]
        test_mask = out["origin_index"] == origin
        test = out.loc[test_mask]

        out.loc[test_mask, "gpr_train_rows"] = len(train)

        if len(train) < int(min_train_rows):
            continue

        model = fit_gpr(train)
        mean, std = predict_gpr_residual(model, test)

        out.loc[test_mask, "gpr_correction"] = mean
        out.loc[test_mask, "gpr_std"] = std
        out.loc[test_mask, "corrected_forecast"] = test["raw_forecast"].to_numpy() + mean

    return out


def future_forecast_with_gpr(
    prepared: pd.DataFrame,
    pipeline,
    residual_df: pd.DataFrame,
    prediction_length: int = 4,
) -> pd.DataFrame:
    values = prepared["production_mt"].to_numpy(dtype=float)
    last_q = int(prepared.iloc[-1]["quarter"])

    raw_preds = chronos_point_forecast(pipeline, values, int(prediction_length))

    future_rows = []
    for h in range(1, int(prediction_length) + 1):
        target_q = ((last_q - 1 + h) % 4) + 1
        feat = make_feature_row(values, raw_preds[h - 1], target_q, h)
        future_rows.append(
            {
                "horizon": h,
                "target_quarter": target_q,
                "raw_forecast": float(raw_preds[h - 1]),
                **feat,
            }
        )

    future = pd.DataFrame(future_rows)

    # All historical residuals are known at the end of the observed dataset.
    train = residual_df[residual_df["target_index"] < len(prepared)].copy()
    if len(train) < 8:
        raise ValueError("Not enough historical residuals to train the final GPR correction model.")

    model = fit_gpr(train)
    mean, std = predict_gpr_residual(model, future)

    future["gpr_correction"] = mean
    future["gpr_std"] = std
    future["corrected_forecast"] = future["raw_forecast"] + mean

    # Build future year/quarter labels.
    year = int(prepared.iloc[-1]["year"])
    q = last_q
    labels = []
    for _ in range(int(prediction_length)):
        q += 1
        if q == 5:
            q = 1
            year += 1
        labels.append(f"{year} Q{q}")
    future["target_period"] = labels
    return future


def _smape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(pred)
    valid = denom > 0
    if not np.any(valid):
        return np.nan
    return float(np.mean(200.0 * np.abs(pred[valid] - actual[valid]) / denom[valid]))


def seasonal_mase_denominator(values: np.ndarray, season_length: int = 4) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= season_length:
        return np.nan
    denom = np.mean(np.abs(values[season_length:] - values[:-season_length]))
    return float(denom) if denom > 0 else np.nan


def calculate_metrics(
    evaluation_df: pd.DataFrame,
    full_series_values: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate raw Chronos and corrected Chronos metrics on rows for which
    a time-safe GPR correction was available.
    """
    valid = evaluation_df.dropna(subset=["corrected_forecast"]).copy()
    if valid.empty:
        return pd.DataFrame()

    mase_denom = seasonal_mase_denominator(full_series_values, season_length=4)

    result = []
    for name, col in [
        ("Raw Chronos", "raw_forecast"),
        ("Chronos + GPR", "corrected_forecast"),
    ]:
        actual = valid["actual"].to_numpy(dtype=float)
        pred = valid[col].to_numpy(dtype=float)
        err = pred - actual
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        smape = _smape(actual, pred)
        mase = float(mae / mase_denom) if np.isfinite(mase_denom) and mase_denom > 0 else np.nan
        result.append(
            {
                "Method": name,
                "MAE": mae,
                "RMSE": rmse,
                "sMAPE (%)": smape,
                "MASE": mase,
                "Evaluated forecasts": len(valid),
            }
        )
    return pd.DataFrame(result)


def horizon_metrics(
    evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    valid = evaluation_df.dropna(subset=["corrected_forecast"]).copy()
    rows = []
    for h, group in valid.groupby("horizon"):
        actual = group["actual"].to_numpy(float)
        raw = group["raw_forecast"].to_numpy(float)
        corr = group["corrected_forecast"].to_numpy(float)

        raw_mae = float(np.mean(np.abs(raw - actual)))
        corr_mae = float(np.mean(np.abs(corr - actual)))
        improvement = (
            100.0 * (raw_mae - corr_mae) / raw_mae
            if raw_mae > 0
            else np.nan
        )
        rows.append(
            {
                "Horizon": f"{int(h)} quarter(s)",
                "Raw Chronos MAE": raw_mae,
                "Chronos + GPR MAE": corr_mae,
                "MAE improvement (%)": improvement,
                "Forecasts": len(group),
            }
        )
    return pd.DataFrame(rows)
