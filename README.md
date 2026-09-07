
# Chronos + GPR Palay Forecasting — Functional POC

This is a **working Proof of Concept** for the proposed BS Computer Science thesis:

> **Gaussian Process Residual Modeling Enhancement of Chronos for Quarterly Philippine Palay Production Forecasting**

The POC is intentionally small. It demonstrates the thesis contribution rather than building a full agricultural information system.

## What the POC proves

The prototype implements this pipeline:

**Quarterly Palay Data → Chronos → Raw Forecast → Historical Residuals → Gaussian Process Regression → Corrected Forecast**

Chronos is kept **unchanged**. GPR is an added residual-learning layer that tries to learn the errors left by Chronos and estimate a correction.

The app also performs a **time-safe rolling evaluation** so that a historical correction only learns from residuals whose actual outcomes would already have been known at that point.

---

## Requirements

Recommended:

- Windows 10/11, macOS, or Linux
- Python 3.10–3.12
- 16 GB RAM recommended
- Internet access on the first run so Chronos-T5 Small can be downloaded
- GPU is optional; CPU works but historical Chronos backtesting will be slower

Chronos-T5 Small is a pretrained model. **You are not training Chronos from scratch.**

---

## Windows PowerShell setup

Open PowerShell inside this POC folder.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

Streamlit will display a local address such as:

```text
http://localhost:8501
```

Open it in your browser.

If PowerShell blocks the virtual environment activation script, you can run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---


## Chronos API compatibility note

The POC intentionally calls Chronos using positional arguments:

```python
forecast = pipeline.predict(context, prediction_length)
```

This matches the official Chronos-T5 examples and avoids a compatibility error in
some installed `chronos-forecasting` versions where `predict()` does not accept
`context=` as a keyword argument.


## Quick test

The project includes:

`sample_demo_palaysynthetic.csv`

This is **synthetic data**, not official PSA data. Use it only to prove that the software pipeline can run.

In the app:

1. Select **Demo data** and review the production history.
2. Choose how many quarters ahead to forecast. **Advanced settings** contains the historical test points (16 by default) and minimum past errors.
3. Press **Run comparison**.
4. On the first run, Chronos-T5 Small will be downloaded.
5. Use **Next quarters** to see the original Chronos forecast, estimated GPR correction, and adjusted forecast. Download the complete forecast as CSV.
6. Use **Historical accuracy** to see whether the correction reduced average error. Detailed metrics, results by forecast horizon, and the historical evaluation CSV are available inside the expanders.

Results stay available during the session when controls change or CSV files are downloaded. If you change the data or settings, a notice marks the previous results until you run the comparison again. Uploads need at least 37 consecutive quarterly observations.

Because Chronos must be run repeatedly for historical residual generation, CPU execution may take several minutes.

To check the core logic and interface without downloading Chronos:

```powershell
python test_core.py
python -m unittest test_app -v
```

The interface tests substitute a deterministic forecast for Chronos inference while using the real GPR correction and metric calculations.

---

## Using official PSA data

For the actual thesis experiment, download the quarterly palay production data from the **PSA OpenSTAT Palay and Corn Production Database** and prepare a CSV with exactly these columns:

```csv
year,quarter,production_mt
1987,1,....
1987,2,....
1987,3,....
1987,4,....
1988,1,....
```

Rules:

- `year`: four-digit year
- `quarter`: `1`, `2`, `3`, `4` or `Q1`, `Q2`, `Q3`, `Q4`
- `production_mt`: production in metric tons
- one row per quarter
- quarters must be consecutive
- do not mix national and provincial series in one file

The POC is designed first for the **national total palay production** series. Irrigated and rainfed palay can later be tested separately for robustness.

---

## What the GPR learns

For each historical Chronos forecast, the program computes:

```text
Residual = Actual PSA production - Chronos forecast
```

The GPR correction layer receives a small number of features:

- target quarter
- forecasting horizon
- Chronos raw forecast
- latest production value
- same-quarter production from the previous year
- average of the latest four quarters
- variability of the latest four quarters
- latest quarter-to-quarter change

The GPR target is the historical **Chronos residual**.

For a new forecast:

```text
Corrected forecast = Chronos forecast + GPR-predicted residual
```

This directly demonstrates the thesis improvement.

---

## Why the historical evaluation avoids future-data leakage

Suppose the prototype is pretending that it is making a forecast at historical time `t`.

The GPR model is allowed to learn only from residuals whose actual target values occurred **before time t**.

It is not allowed to learn from a forecast error whose actual value would still have been unknown at that moment.

This is implemented in `sequential_gpr_correction()` in `core.py`.

---

## Project files

- `app.py` — Streamlit POC interface
- `styles.css` — responsive interface styling
- `.streamlit/config.toml` — shared colors and typography
- `core.py` — Chronos, residual generation, GPR correction, and evaluation logic
- `sample_demo_palaysynthetic.csv` — synthetic data for a software demonstration
- `requirements.txt` — required Python packages
- `test_core.py` — lightweight tests for the non-Chronos GPR logic
- `test_app.py` — interface flow and result-persistence checks using a test forecast
- `POC_EXPLANATION.md` — short dean/panel explanation

---

## Important research limitation

The POC demonstrates **implementability**, not final thesis performance.

The synthetic demo cannot establish that the method improves real Philippine palay forecasts. The final thesis must run the same method on official PSA data using a predetermined evaluation protocol.

It is scientifically acceptable if Chronos + GPR does not always outperform raw Chronos. The thesis question is whether the residual enhancement improves the model, for which horizons, and under what conditions.

---

## Research-use disclaimer

This prototype is for academic experimentation. It is not an official forecast from PSA, DA, or NFA and should not be used for policy, procurement, trading, or operational decisions.
