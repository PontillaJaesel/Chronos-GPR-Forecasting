from __future__ import annotations

from datetime import datetime
import hashlib
import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import (
    prepare_input_data,
    load_chronos_pipeline,
    build_residual_dataset,
    sequential_gpr_correction,
    future_forecast_with_gpr,
    calculate_metrics,
    horizon_metrics,
)

APP_DIR = Path(__file__).resolve().parent
DEMO_PATH = APP_DIR / "sample_demo_palaysynthetic.csv"
MODEL_NAME = "amazon/chronos-t5-small"
MIN_QUARTERS = 37
SERIES_COLORS = {
    "Observed production": "#303c37",
    "Chronos": "#a47843",
    "Chronos + GPR": "#28614d",
}

st.set_page_config(page_title="Palay | Forecast comparison", page_icon="🌾", layout="wide")
st.markdown(f"<style>{(APP_DIR / 'styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_chronos():
    return load_chronos_pipeline(MODEL_NAME)


def section_label(number: str, label: str):
    st.markdown(f'<div class="section-label"><span>{number}</span>{label}</div>', unsafe_allow_html=True)


def production_chart(frame: pd.DataFrame, period: str, series: dict[str, str], height: int = 290):
    """Use the same readable units, colors, and legend across all comparisons."""
    plot = frame[[period, *series]].rename(columns={period: "Quarter", **series})
    order = plot["Quarter"].tolist()
    plot = plot.melt("Quarter", var_name="Series", value_name="Production")
    names = list(series.values())
    ticks = order[::max(4, 4 * int(np.ceil(len(order) / 32)))] if len(order) > 16 else order
    chart = (
        alt.Chart(plot)
        .encode(
            x=alt.X("Quarter:O", sort=order, title=None,
                    axis=alt.Axis(values=ticks, labelAngle=0, labelOverlap=True, labelPadding=12, tickSize=0)),
            y=alt.Y("Production:Q", title="Production (metric tons)",
                    scale=alt.Scale(zero=False), axis=alt.Axis(format="~s", tickCount=4, titlePadding=16)),
            color=alt.Color("Series:N", sort=names, title=None,
                            scale=alt.Scale(domain=names, range=[SERIES_COLORS[n] for n in names]),
                            legend=alt.Legend(orient="bottom", direction="horizontal", symbolType="stroke", padding=18)),
            strokeDash=alt.StrokeDash("Series:N", sort=names,
                                     scale=alt.Scale(domain=names, range=[
                                         [5, 4] if n == "Chronos" else [2, 2] if n == "Observed production" and len(names) > 1 else [1, 0]
                                         for n in names]),
                                     legend=None),
            tooltip=[alt.Tooltip("Quarter:N"), alt.Tooltip("Series:N", title="Method"),
                     alt.Tooltip("Production:Q", title="Metric tons", format=",.0f")],
        )
    )
    lines = chart.mark_line(strokeWidth=2.3, point=alt.OverlayMarkDef(size=42) if len(order) <= 16 else False)
    # Invisible hit targets expose each quarter's tooltip on dense history charts.
    chart = alt.layer(lines, chart.mark_point(opacity=0, size=120)) if len(order) > 16 else lines
    chart = (
        chart
        .properties(height=height)
        .configure(
            background="transparent", font="sans-serif",
            view=alt.ViewConfig(strokeWidth=0),
            axis=alt.AxisConfig(domain=False, gridColor="#e6e9e3", labelColor="#68716a", titleColor="#68716a",
                                labelFontSize=11, titleFontSize=11, titleFontWeight="normal"),
            axisX=alt.AxisConfig(grid=False),
            legend=alt.LegendConfig(labelColor="#48534c", labelFontSize=12),
        )
    )
    st.altair_chart(chart, width="stretch", theme=None)


def show_future(result):
    future = result["future"]
    st.subheader("The next quarters, side by side")
    st.caption("Chronos makes the original forecast. GPR estimates a correction from past errors. All values are in metric tons (MT).")

    first = future.iloc[0]
    st.markdown(f"**Next quarter · {first['target_period']}**")
    a, b, c = st.columns(3)
    a.metric("Original · Chronos", f"{first['raw_forecast']:,.0f} MT")
    b.metric("Estimated correction · GPR", f"{first['gpr_correction']:+,.0f} MT")
    c.metric("Adjusted · Chronos + GPR", f"{first['corrected_forecast']:,.0f} MT")

    production_chart(future, "target_period", {"raw_forecast": "Chronos", "corrected_forecast": "Chronos + GPR"}, height=260)
    st.dataframe(
        future[["target_period", "raw_forecast", "gpr_correction", "corrected_forecast"]],
        hide_index=True, width="stretch",
        column_config={
            "target_period": "Quarter",
            "raw_forecast": st.column_config.NumberColumn("Chronos (MT)", format="%,.0f"),
            "gpr_correction": st.column_config.NumberColumn("GPR correction (MT)", format="%,.0f"),
            "corrected_forecast": st.column_config.NumberColumn("Chronos + GPR (MT)", format="%,.0f"),
        },
    )
    st.caption("Original forecast + estimated correction = adjusted forecast. A positive correction raises the forecast; a negative one lowers it.")

    with st.expander("How uncertain is the correction?"):
        st.write("GPR also estimates the standard deviation (SD) of its correction. A larger SD means more uncertainty about the adjustment. This is not an uncertainty interval for the full production forecast.")
        st.dataframe(
            future[["target_period", "gpr_std"]], hide_index=True, width="stretch",
            column_config={"target_period": "Quarter", "gpr_std": st.column_config.NumberColumn("Correction SD (MT)", format="%,.0f")},
        )

    st.download_button("Download forecasts (.csv)", future.to_csv(index=False).encode("utf-8"),
                       file_name="palay_forecast_comparison.csv", mime="text/csv", key="download_forecast", on_click="ignore")


def show_accuracy(result):
    metrics = result["metrics"]
    evaluated = result["evaluated"]
    st.subheader("Did learning from past errors help?")
    st.caption("We test on historical quarters with known production, then compare both methods on the same forecasts. Lower error is better.")
    if metrics.empty:
        st.info("There are not enough past errors for a fair comparison yet. Increase historical test points or lower the minimum examples in Advanced settings, then run again.")
    else:
        raw_mae = float(metrics.loc[metrics["Method"] == "Raw Chronos", "MAE"].iloc[0])
        gpr_mae = float(metrics.loc[metrics["Method"] == "Chronos + GPR", "MAE"].iloc[0])
        change = 100 * (raw_mae - gpr_mae) / raw_mae if raw_mae > 0 else np.nan
        if np.isfinite(change) and change > 0:
            verdict = f"The adjustment reduced average error by {change:.1f}%."
        elif np.isfinite(change) and change < 0:
            verdict = f"The adjustment increased average error by {abs(change):.1f}%."
        elif raw_mae == gpr_mae:
            verdict = "Both methods had the same average error."
        else:
            verdict = "Chronos had zero average error; the adjustment increased it."
        st.markdown(f"**{verdict}**")
        a, b = st.columns(2)
        a.metric("Chronos · average error", f"{raw_mae:,.0f} MT")
        b.metric("Chronos + GPR · average error", f"{gpr_mae:,.0f} MT")
        count = int(metrics.iloc[0]["Evaluated forecasts"])
        st.caption(f"Mean absolute error (MAE) is how far forecasts are from observed production, on average. Based on {count} forecasts across 1–4 quarters ahead; early forecasts without enough prior errors are excluded.")
        if result["is_demo"]:
            st.caption("These results use synthetic data. They demonstrate the method, but cannot establish accuracy on real Philippine production.")

        one = evaluated.dropna(subset=["corrected_forecast"])
        one = one[one["horizon"] == 1]
        if not one.empty:
            st.markdown("**One quarter ahead · forecast vs. observed production**")
            production_chart(one, "target_period", {"actual": "Observed production", "raw_forecast": "Chronos", "corrected_forecast": "Chronos + GPR"})

        with st.expander("All accuracy metrics & forecast horizons"):
            st.write("Each metric compares the same historical forecasts. Lower values are better.")
            display = metrics.replace({"Method": {"Raw Chronos": "Chronos"}})
            st.dataframe(display, width="stretch", hide_index=True, column_config={
                "MAE": st.column_config.NumberColumn("MAE (MT)", format="%,.0f"),
                "RMSE": st.column_config.NumberColumn("RMSE (MT)", format="%,.0f"),
                "sMAPE (%)": st.column_config.NumberColumn(format="%.3f"),
                "MASE": st.column_config.NumberColumn(format="%.3f"),
            })
            st.markdown("**MAE** is the average absolute error. **RMSE** gives more weight to large errors. **sMAPE** expresses error as a percentage. **MASE** scales error against the series’ average year-over-year change.")
            by_horizon = horizon_metrics(evaluated).rename(columns={"Raw Chronos MAE": "Chronos MAE"})
            st.markdown("**Performance by forecast horizon**")
            st.caption("A horizon of 1 quarter means predicting the next quarter; 4 means predicting a year ahead. Positive improvement means GPR reduced MAE.")
            st.dataframe(by_horizon, width="stretch", hide_index=True, column_config={
                "Chronos MAE": st.column_config.NumberColumn(format="%,.0f"),
                "Chronos + GPR MAE": st.column_config.NumberColumn(format="%,.0f"),
                "MAE improvement (%)": st.column_config.NumberColumn(format="%+.2f%%"),
            })

    with st.expander("Inspect the historical forecast errors"):
        st.write("An error, or residual, is observed production minus the original forecast. GPR learns from these examples. Each historical correction uses only errors whose outcomes were already known at that forecast date.")
        residuals = result["residuals"]
        columns = ["origin_period", "target_period", "horizon", "actual", "raw_forecast", "residual"]
        st.dataframe(residuals[columns], width="stretch", hide_index=True, column_config={
            "origin_period": "Last known quarter", "target_period": "Predicted quarter", "horizon": "Quarters ahead",
            "actual": st.column_config.NumberColumn("Observed (MT)", format="%,.0f"),
            "raw_forecast": st.column_config.NumberColumn("Chronos (MT)", format="%,.0f"),
            "residual": st.column_config.NumberColumn("Error (MT)", format="%,.0f"),
        })
        st.download_button("Download historical evaluation (.csv)", evaluated.to_csv(index=False).encode("utf-8"),
                           file_name="palay_historical_evaluation.csv", mime="text/csv", key="download_evaluation", on_click="ignore")


st.markdown('''
<div class="masthead"><div class="wordmark">PALAY <span>/ FORECAST STUDY</span></div><div class="edition">Research proof of concept</div></div>
<div class="intro"><h1>Quarterly palay forecasts.</h1><p>Can learning from past errors improve the next forecast? <br>Compare Chronos with a GPR correction using Philippine palay production data.</p></div>
<div class="method-strip" aria-label="How the comparison works">
  <div><span class="method-number">01</span><strong>Make a forecast</strong><p>Chronos predicts production from quarterly history.</p></div>
  <div><span class="method-number">02</span><strong>Learn from past errors</strong><p>GPR looks for patterns in earlier forecast mistakes.</p></div>
  <div><span class="method-number">03</span><strong>Compare the adjustment</strong><p>Add the estimated correction and check the difference.</p></div>
</div>
''', unsafe_allow_html=True)

with st.expander("A quick guide to the experiment"):
    st.markdown("**Palay** is unhusked rice. This experiment forecasts quarterly production in metric tons. **Chronos** is the pretrained forecasting model; **Gaussian Process Regression (GPR)** is an added model that learns its past errors.")
    st.markdown("For example, if Chronos predicted **100 MT** and observed production was **110 MT**, the error was **+10 MT**. GPR learns from many such errors to estimate an adjustment for a new forecast.")
    st.markdown("**What we are testing:** whether the adjusted forecasts are closer to observed production. Improvement is not guaranteed. During historical testing, GPR only sees errors that would already have been known at the time; Chronos itself is kept unchanged.")

section_label("01", "Set up the comparison")
controls, preview = st.columns([1, 2.15], gap="large")
data = None
with controls:
    st.subheader("Start with your data")
    source = st.radio("Data source", ["Demo data", "Upload CSV"], horizontal=True, key="data_source")
    is_demo = source == "Demo data"
    input_bytes = DEMO_PATH.read_bytes() if is_demo else None
    source_name = "Synthetic demo"
    if is_demo:
        st.caption("Synthetic quarterly data, ready to explore. For research results, upload official PSA data.")
    else:
        upload = st.file_uploader("Quarterly production CSV", type=["csv"])
        if upload is not None:
            input_bytes = upload.getvalue()
            source_name = upload.name
        else:
            source_name = "No file selected"

    with st.expander("CSV format & sample file"):
        st.code("year,quarter,production_mt\n2024,1,4680000\n2024,2,4250000", language="csv")
        st.caption("Use one series with at least 37 consecutive quarters and positive production in metric tons. Quarters can be 1–4 or Q1–Q4. The two rows above illustrate the format only.")
        st.download_button("Download sample CSV", DEMO_PATH.read_bytes(), file_name=DEMO_PATH.name, mime="text/csv", on_click="ignore")

    if input_bytes is not None:
        try:
            data = prepare_input_data(pd.read_csv(io.BytesIO(input_bytes)))
            if data.empty:
                raise ValueError("The CSV has no data rows. Add quarterly observations below the header.")
            if not np.isfinite(data["production_mt"]).all():
                raise ValueError("Production must contain finite numeric values.")
        except Exception as exc:
            st.error(f"Please check your CSV. {exc}")
            data = None

    horizon = st.selectbox("How far ahead?", options=[1, 2, 3, 4], index=3, key="horizon",
                          format_func=lambda n: "4 quarters · 1 year" if n == 4 else f"{n} quarter{'s' if n > 1 else ''}")
    with st.expander("Advanced settings"):
        max_origins = max(5, min(28, len(data) - 32)) if data is not None else 28
        if "backtest_origins" not in st.session_state:
            st.session_state["backtest_origins"] = min(16, max_origins)
        elif st.session_state["backtest_origins"] > max_origins:
            st.session_state["backtest_origins"] = max_origins
        backtest_origins = st.slider("Historical test points", 4, max_origins,
                                    key="backtest_origins",
                                    help="How many past dates to forecast from. Each tests up to 4 quarters ahead. More dates provide more error examples and take longer to run.")
        min_gpr_rows = st.slider("Past errors needed before correction", 8, 24, 12, key="min_gpr_rows",
                                 help="During historical testing, wait for this many known error examples before applying GPR. The final future correction uses all available historical errors.")
        st.caption("Forecast model: Chronos-T5 Small. Historical testing always compares 1–4 quarters ahead.")

    ready = data is not None and len(data) >= MIN_QUARTERS
    if data is not None and not ready:
        st.warning(f"Add {MIN_QUARTERS - len(data)} more consecutive quarters to run this comparison. At least {MIN_QUARTERS} are needed.")
    run = st.button("Run comparison", type="primary", width="stretch", disabled=not ready)
    st.caption("May take a few minutes on CPU. The first run also downloads the pretrained model.")

with preview:
    st.subheader("Production history")
    if data is not None:
        st.caption(f"{source_name} · {data.iloc[0]['period']} – {data.iloc[-1]['period']}")
        a, b, c = st.columns(3)
        a.metric("Quarters available", len(data))
        b.metric("Latest quarter", data.iloc[-1]["period"])
        latest_production = float(data.iloc[-1]["production_mt"])
        latest_label = f"{latest_production / 1_000_000:,.2f}M MT" if latest_production >= 1_000_000 else f"{latest_production:,.0f} MT"
        c.metric("Latest production", latest_label,
                 help=f"{latest_production:,.0f} metric tons. M means million.")
        production_chart(data, "period", {"production_mt": "Observed production"}, height=250)
        st.caption("The recurring rises and falls show seasonal production patterns. Hover over the chart to see each quarter’s value.")
        with st.expander("View the quarterly data"):
            st.dataframe(data[["year", "quarter", "production_mt"]], hide_index=True, width="stretch",
                         column_config={"year": st.column_config.NumberColumn("Year", format="%d"), "quarter": "Quarter",
                                        "production_mt": st.column_config.NumberColumn("Production (MT)", format="%,.0f")})
    else:
        st.markdown('<div class="empty-state"><div class="empty-label">YOUR DATA STARTS HERE</div><h3>Add quarterly production data</h3><p>Upload a CSV to preview the history, or choose Demo data to explore the full comparison.</p></div>', unsafe_allow_html=True)

signature = (hashlib.sha256(input_bytes).hexdigest(), source_name, is_demo, horizon, backtest_origins, min_gpr_rows) if ready else None

section_label("02", "Read the results")
if run and ready:
    with st.status("Running the comparison…", expanded=True) as status:
        try:
            st.write("1. Loading the Chronos forecasting model…")
            pipeline, device = get_chronos()
            st.write("2. Forecasting historical quarters to collect past errors…")
            progress = st.progress(0)

            def update_progress(done, total):
                progress.progress(done / max(total, 1), text=f"Historical test {done} of {total}")

            residuals = build_residual_dataset(data, pipeline, max_horizon=4, n_origins=backtest_origins,
                                               min_context=32, progress_callback=update_progress)
            progress.empty()
            st.write("3. Testing GPR corrections using only errors already known at each date…")
            evaluated = sequential_gpr_correction(residuals, min_train_rows=min_gpr_rows)
            metrics = calculate_metrics(evaluated, data["production_mt"].to_numpy(float))
            st.write("4. Producing the next quarterly forecasts…")
            future = future_forecast_with_gpr(data, pipeline, residuals, prediction_length=horizon)
            st.session_state["comparison_result"] = {
                "signature": signature, "future": future, "evaluated": evaluated, "metrics": metrics,
                "residuals": residuals, "source_name": source_name, "is_demo": is_demo,
                "horizon": horizon, "backtest_origins": backtest_origins, "min_gpr_rows": min_gpr_rows,
                "completed_at": datetime.now().strftime("%H:%M"),
            }
            status.update(label="Comparison complete", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="The comparison could not finish", state="error", expanded=True)
            st.error("Check that the dependencies are installed and that the first run can reach Hugging Face to download the model, then try again.")
            with st.expander("Error details"):
                st.code(str(exc), language=None)

result = st.session_state.get("comparison_result")
if result is None:
    st.markdown('''<div class="empty-state results-empty"><div class="empty-label">READY WHEN YOU ARE</div><h3>See what the correction changes.</h3><p>Run the comparison to see the upcoming forecasts and whether GPR reduced errors on historical data.</p><div class="result-preview"><span><strong>Next quarters</strong>Original forecast + correction = adjusted forecast</span><span><strong>Historical accuracy</strong>Both methods compared with known production</span></div></div>''', unsafe_allow_html=True)
else:
    if result["signature"] != signature:
        st.warning("The data or settings have changed. These are the previous results; run the comparison again to update them.")
    st.caption(f"Completed at {result['completed_at']} · {result['source_name']} · {result['horizon']} quarter(s) ahead · {result['backtest_origins']} historical test points · {result['min_gpr_rows']} minimum past errors")
    forecast_tab, accuracy_tab = st.tabs(["Next quarters", "Historical accuracy"])
    with forecast_tab:
        show_future(result)
    with accuracy_tab:
        show_accuracy(result)

st.markdown('<footer class="page-footer"><span>PALAY / Chronos + GPR</span><p>Academic prototype. Demo data is synthetic. This is not an official PSA, DA, or NFA forecast and is not for operational or policy decisions.</p></footer>', unsafe_allow_html=True)

