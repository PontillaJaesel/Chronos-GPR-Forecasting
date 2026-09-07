# Chronos + Gaussian Process Residual Modeling POC

## Proof of Concept for Philippine Palay Production Forecasting

### Proposed Working Thesis Title

**Gaussian Process Residual Modeling Enhancement of Chronos for Quarterly Philippine Palay Production Forecasting**

---

# 1. Overview

This repository contains a functional Proof of Concept (POC) for a proposed forecasting method that combines the pretrained **Chronos-T5 Small time-series forecasting model** with **Gaussian Process Regression (GPR)-based residual modeling**.

The purpose of the POC is to demonstrate that the proposed computational method can actually be implemented and can produce the intended outputs.

The prototype is not intended to be a complete agricultural forecasting or decision-support system. Instead, it focuses specifically on demonstrating the proposed thesis contribution:

> **Use Gaussian Process Regression to learn the historical forecast errors left by Chronos and use the predicted residual error to adjust future Chronos forecasts.**

The complete implemented pipeline is:

```text
Quarterly Palay Production Data
              ↓
      Data Validation
              ↓
     Chronos-T5 Small
              ↓
     Raw Chronos Forecast
              ↓
Historical Forecast Simulation
              ↓
    Residual Calculation
              ↓
Gaussian Process Regression
              ↓
 Predicted Residual Correction
              ↓
   Chronos + GPR Forecast
              ↓
Historical Performance Evaluation
              ↓
Future Forecast Output
```

The POC uses an actual pretrained Chronos model and an actual Gaussian Process Regression model. The models are not simulated, hard-coded, or replaced by manually generated predictions.

---

# 2. Main Purpose of the POC

The POC was developed to answer the following implementation question:

> **Can Gaussian Process Regression be integrated with Chronos so that historical Chronos forecasting errors can be learned and used to correct future Chronos forecasts?**

The POC demonstrates that this process is technically feasible.

It verifies that the proposed method can:

1. accept quarterly time-series data;
2. validate the structure of the input;
3. run the real pretrained Chronos model;
4. generate historical Chronos forecasts;
5. calculate the errors or residuals produced by Chronos;
6. construct a residual-learning dataset;
7. train a real Gaussian Process Regression model;
8. prevent future information from leaking into historical model training;
9. predict the likely residual error of a new Chronos forecast;
10. apply the predicted residual as a correction;
11. compare Raw Chronos and Chronos + GPR;
12. calculate forecasting error metrics;
13. evaluate different forecast horizons; and
14. produce future corrected forecasts through a working Streamlit interface.

Therefore, the POC demonstrates the actual computational process proposed for the thesis rather than functioning only as a graphical mock-up.

---

# 3. Project Structure

The POC is composed of the following files:

```text
chronos_gpr_poc/
│
├── app.py
├── core.py
├── requirements.txt
├── sample_demo_palaysynthetic.csv
├── test_core.py
├── POC_EXPLANATION.md
└── README.md
```

### `app.py`

Contains the Streamlit graphical interface.

It is responsible for:

- allowing the user to choose a dataset;
- displaying the historical time series;
- configuring forecast settings;
- running the complete Chronos + GPR pipeline;
- displaying Raw Chronos results;
- displaying Chronos + GPR results;
- presenting forecasting metrics;
- displaying graphs; and
- presenting future corrected forecasts.

### `core.py`

Contains the main computational logic of the POC.

This includes:

- data validation;
- Chronos model loading;
- Chronos inference;
- feature construction;
- historical residual generation;
- GPR construction;
- GPR training;
- sequential time-aware residual correction;
- future forecasting;
- MAE calculation;
- RMSE calculation;
- sMAPE calculation;
- MASE calculation; and
- horizon-specific evaluation.

### `requirements.txt`

Contains the Python packages needed to execute the POC, including:

```text
streamlit
chronos-forecasting
torch
pandas
numpy
scikit-learn
matplotlib
```

### `sample_demo_palaysynthetic.csv`

Contains synthetic quarterly production values.

This dataset is included only so that the complete POC can be executed before official PSA data are prepared.

It must not be interpreted as official Philippine agricultural data.

### `test_core.py`

Contains lightweight tests for important parts of the computational implementation, including:

- data preparation;
- feature generation;
- GPR training;
- sequential residual correction; and
- forecasting metric generation.

---

# 4. User Interface and System Flow

The POC uses **Streamlit** to provide a simple local web interface.

When the program is started using:

```powershell
streamlit run app.py
```

the application guides the user through the following process.

---

## Step 1 — Load Quarterly Data

The user can either:

- use the included synthetic demonstration dataset; or
- upload a prepared CSV file.

The expected CSV format is:

```csv
year,quarter,production_mt
2020,1,5000000
2020,2,5200000
2020,3,4800000
2020,4,5500000
```

The three required variables are:

- `year`
- `quarter`
- `production_mt`

For the actual thesis, `production_mt` will represent quarterly Philippine palay production in metric tons.

---

# 5. Input Data Validation

Before any forecasting is performed, the function:

```python
prepare_input_data()
```

checks whether the dataset is suitable for the POC.

The program verifies that:

- all required columns exist;
- year values are valid;
- quarter values are between 1 and 4;
- production values are numeric;
- production values are positive;
- duplicate year-quarter combinations do not exist; and
- quarterly observations are consecutive.

For example:

```python
required = {"year", "quarter", "production_mt"}
```

and:

```python
if out.duplicated(["year", "quarter"]).any():
    raise ValueError(...)
```

The program also verifies that there are no missing quarters:

```python
gaps = np.diff(serial.to_numpy())

if len(gaps) and not np.all(gaps == 1):
    raise ValueError(...)
```

This validation helps prevent incorrectly structured datasets from silently producing misleading forecasts.

---

# 6. Historical Time-Series Visualization

After the dataset passes validation, the Streamlit application displays:

- the production history;
- number of available quarterly observations;
- first available quarter;
- latest available quarter; and
- latest production value.

This allows the user to inspect the input time series before executing the models.

---

# 7. Forecast Configuration

The application allows the user to configure:

### Future Forecast Horizon

The user can forecast:

- 1 quarter ahead;
- 2 quarters ahead;
- 3 quarters ahead; or
- 4 quarters ahead.

### Historical Forecast Origins

This controls how many historical forecast periods will be simulated to generate Chronos residual examples.

For example:

```text
Historical forecast origins = 16
```

means that the POC will simulate Chronos forecasts from 16 different historical points.

### Minimum Residual Examples Before GPR Training

The program also allows a minimum number of historical residuals to be required before GPR is allowed to perform a correction.

This prevents the residual model from being trained with extremely few examples.

---

# 8. Real Chronos Model

The POC uses the actual pretrained:

```text
amazon/chronos-t5-small
```

model.

Chronos is loaded using:

```python
pipeline = ChronosPipeline.from_pretrained(
    model_name,
    device_map=device_map,
    torch_dtype=torch_dtype,
)
```

The program detects whether a CUDA-compatible GPU is available.

If a GPU is available, it can use the GPU.

Otherwise, Chronos runs using the CPU.

For example, the application may display:

```text
Chronos loaded on: CPU
```

This confirms that the real pretrained model has been successfully loaded.

Chronos is not trained from scratch in the POC.

It remains pretrained and frozen.

---

# 9. Real Chronos Forecast Generation

The historical production values are converted into a tensor and passed to Chronos:

```python
context = torch.tensor(
    history,
    dtype=torch.float32
)
```

Chronos is then executed using:

```python
forecast = pipeline.predict(
    context,
    int(prediction_length),
)
```

Chronos produces multiple probabilistic forecast samples.

The POC converts these samples into a point forecast using the median:

```python
return np.median(
    samples,
    axis=0
).astype(float)
```

Therefore, the values displayed as:

```text
Raw Chronos Forecast
```

are actual outputs generated by the pretrained Chronos-T5 Small model.

They are not manually entered or randomly generated by the Streamlit application.

---

# 10. Historical Forecast Simulation

GPR cannot learn Chronos forecasting errors unless historical Chronos forecasts are first generated.

The POC therefore performs historical forecast simulation through:

```python
build_residual_dataset()
```

At multiple historical points, the program pretends that later observations have not yet occurred.

For example:

```text
Historical data available until 2021 Q4
                  ↓
Chronos forecasts 2022 Q1–Q4
                  ↓
Forecasts are compared with the known actual 2022 values
                  ↓
Chronos residual errors are calculated
```

The same procedure is repeated across multiple historical forecast origins.

This creates a collection of real historical Chronos forecasting errors.

---

# 11. Residual Calculation

The residual is calculated as:

```text
Residual = Actual Value - Chronos Forecast
```

The source code implements this using:

```python
"residual": actual - raw
```

For example:

```text
Actual Production:
5,200,000 MT

Chronos Forecast:
5,000,000 MT

Residual:
+200,000 MT
```

A positive residual means Chronos underestimated the actual value.

A negative residual means Chronos overestimated the actual value.

For example:

```text
Actual:
5,000,000

Chronos:
5,300,000

Residual:
-300,000
```

The residuals become the values that GPR attempts to learn.

---

# 12. Why Residual Modeling Is Used

Chronos remains responsible for generating the primary forecast.

The GPR model does not replace Chronos.

Instead, the proposed method asks:

> **Does Chronos leave forecasting errors that contain predictable patterns?**

If such patterns exist, GPR may be able to estimate the error Chronos is likely to make under similar conditions.

The proposed approach is therefore:

```text
Chronos learns the time series
          +
GPR learns the remaining Chronos errors
```

rather than:

```text
Chronos versus GPR
```

This distinction is important because the thesis contribution is the **residual-modeling enhancement applied after Chronos**.

---

# 13. GPR Feature Construction

For every historical Chronos forecast, the program creates features that describe the conditions under which that forecast was generated.

The current features are:

```python
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
```

---

## Target Quarter

The quarter is represented using:

```python
quarter_sin
quarter_cos
```

instead of directly treating Q1, Q2, Q3, and Q4 as unrelated numeric values.

This helps represent the cyclical nature of quarterly seasonality.

---

## Forecast Horizon

The feature:

```python
horizon
```

indicates whether the prediction is:

```text
1 quarter ahead
2 quarters ahead
3 quarters ahead
4 quarters ahead
```

Chronos may produce different forecasting errors depending on how far ahead it is predicting.

---

## Raw Chronos Forecast

```python
raw_forecast
```

contains the original Chronos prediction.

This allows GPR to determine whether forecast errors behave differently depending on the magnitude of the Chronos prediction.

---

## Latest Known Value

```python
last_value
```

contains the latest production observation available when the forecast is generated.

---

## Seasonal Lag

```python
seasonal_lag4
```

contains the value from four quarters earlier.

For quarterly data, a four-quarter lag represents approximately the corresponding seasonal position from the previous year.

---

## Recent Annual Mean

```python
mean_last4
```

contains the average of the latest four known quarters.

---

## Recent Variability

```python
std_last4
```

describes how much production has varied during the latest four observations.

---

## Recent Change

```python
recent_change
```

contains:

```text
Latest production - Previous quarter production
```

This gives GPR information regarding recent movement in the time series.

---

# 14. Feature Scaling

The features contain very different numerical magnitudes.

For example:

```text
quarter_sin = 1
horizon = 2
raw_forecast = 5,000,000
recent_change = 250,000
```

The POC therefore applies:

```python
StandardScaler()
```

before the features are given to GPR.

The implemented pipeline is:

```python
Pipeline(
    [
        ("scale", StandardScaler()),
        ("gpr", gpr),
    ]
)
```

This prevents differences in raw numerical scale from unnecessarily affecting the Gaussian Process kernel calculations.

---

# 15. Real Gaussian Process Regression Model

The POC uses:

```python
GaussianProcessRegressor
```

from `scikit-learn`.

It is constructed as:

```python
kernel = (
    ConstantKernel(
        1.0,
        (1e-2, 1e3)
    )
    *
    RBF(
        length_scale=np.ones(
            len(FEATURE_COLUMNS)
        ),
        length_scale_bounds=(
            1e-2,
            1e3
        )
    )
    +
    WhiteKernel(
        noise_level=1.0,
        noise_level_bounds=(
            1e-5,
            1e3
        )
    )
)
```

The model is then created using:

```python
gpr = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    alpha=1e-6,
    n_restarts_optimizer=1,
    random_state=42,
)
```

This means that the POC uses a real Gaussian Process model.

The correction values are not manually specified.

GPR must actually learn a relationship between the available forecasting features and the residual errors generated by Chronos.

---

# 16. Training the GPR Residual Model

The GPR model is trained using:

```python
model.fit(
    residual_rows[FEATURE_COLUMNS],
    residual_rows["residual"]
)
```

Therefore:

```text
Input to GPR:
Forecasting conditions

Target learned by GPR:
Chronos residual
```

Conceptually:

```text
Quarter
Forecast horizon
Raw Chronos forecast
Recent production history
Seasonal information
          ↓
         GPR
          ↓
Expected Chronos residual
```

---

# 17. Prevention of Future-Data Leakage

One of the most important validity controls in the POC is contained in:

```python
sequential_gpr_correction()
```

During historical evaluation, the GPR model is only allowed to learn from residuals whose actual target values occurred before the forecast origin:

```python
train = out[
    out["target_index"] < origin
]
```

For example, if the program is simulating a forecast during 2022:

```text
Can be used:
2019 residuals
2020 residuals
2021 residuals

Cannot be used:
2023 residuals
2024 residuals
2025 residuals
```

This is important because, in a real forecasting situation, future actual production values would not yet be known.

Without this restriction, GPR could accidentally receive information from the future, causing unrealistically strong historical performance.

The POC therefore performs a **time-aware sequential evaluation** instead of randomly mixing historical and future observations.

---

# 18. GPR Residual Prediction

Once GPR has learned from historical residuals, it predicts the expected residual for new Chronos forecasts.

The POC uses:

```python
mean, std = gpr.predict(
    X_scaled,
    return_std=True
)
```

This produces two useful outputs.

### Predicted Residual Mean

This is the correction GPR recommends applying to Chronos.

Example:

```text
GPR Predicted Correction:
-150,000 MT
```

### GPR Standard Deviation

This represents the uncertainty associated with that residual estimate.

Example:

```text
Predicted correction:
-150,000 MT

Uncertainty SD:
90,000 MT
```

A larger value indicates that GPR is less certain about the correction.

---

# 19. Final Chronos + GPR Forecast

The thesis enhancement is implemented using:

```python
future["corrected_forecast"] = (
    future["raw_forecast"]
    + mean
)
```

Therefore:

```text
Enhanced Forecast
=
Raw Chronos Forecast
+
GPR-Predicted Residual
```

Example:

```text
Raw Chronos Forecast:
5,500,000 MT

GPR Correction:
-100,000 MT

Chronos + GPR:
5,400,000 MT
```

If the correction is positive:

```text
Raw Chronos:
5,500,000 MT

GPR Correction:
+100,000 MT

Chronos + GPR:
5,600,000 MT
```

This is the central computational contribution demonstrated by the POC.

---

# 20. Historical Performance Evaluation

The prototype evaluates:

```text
Raw Chronos
```

against:

```text
Chronos + GPR
```

The POC does not automatically consider the enhanced method successful.

Instead, both forecasts are compared with known historical actual values.

This is important because the research question is:

> **Does the GPR residual-modeling enhancement actually reduce the forecasting errors left by Chronos?**

rather than assuming that GPR must improve the model.

---

# 21. Forecasting Metrics

The POC calculates four forecasting error metrics.

---

## Mean Absolute Error (MAE)

Implemented as:

```python
mae = float(
    np.mean(
        np.abs(err)
    )
)
```

MAE represents the average magnitude of the forecasting errors.

Lower MAE is better.

Example:

```text
MAE = 100,000 MT
```

means that the forecasts were approximately 100,000 metric tons away from the actual values on average.

---

## Root Mean Squared Error (RMSE)

Implemented as:

```python
rmse = float(
    np.sqrt(
        np.mean(err ** 2)
    )
)
```

RMSE gives larger forecasting errors greater influence.

Lower RMSE is better.

---

## Symmetric Mean Absolute Percentage Error (sMAPE)

The POC calculates percentage-based error using:

```python
200.0 *
np.abs(pred - actual) /
(
    np.abs(actual)
    +
    np.abs(pred)
)
```

Lower sMAPE indicates better forecasting performance.

---

## Mean Absolute Scaled Error (MASE)

The POC also calculates MASE using a quarterly seasonal reference with:

```python
season_length=4
```

MASE provides a scale-independent comparison with a seasonal-naive reference.

For the final thesis implementation, the MASE scaling procedure should be finalized according to the exact predefined rolling training and evaluation protocol so that only appropriate historical information contributes to the scale calculation.

---

# 22. Evaluation by Forecast Horizon

The POC evaluates performance separately for:

```text
1-quarter ahead
2-quarter ahead
3-quarter ahead
4-quarter ahead
```

This allows the proposed study to determine whether GPR behaves differently depending on the forecast horizon.

For example, the final research may discover that GPR:

```text
improves 1-quarter forecasting
but
does not improve 4-quarter forecasting
```

Such a result would still provide useful experimental evidence about the conditions under which the residual enhancement is effective.

---

# 23. Future Forecast Generation

After historical residual examples have been generated, the program trains a final GPR residual model using all residuals whose actual values are known at the end of the available dataset.

Chronos then generates forecasts for the selected future horizon.

The program produces:

```text
Target Quarter
Raw Chronos Forecast
GPR Estimated Correction
GPR Correction Uncertainty
Chronos + GPR Forecast
```

For example:

```text
2026 Q1

Raw Chronos:
5,851,188 MT

GPR Correction:
-11,601 MT

Final Forecast:
5,839,587 MT
```

These values are calculated dynamically by the models.

They are not stored as predetermined outputs.

---

# 24. Streamlit Result Presentation

The POC interface presents:

### Historical Input Graph

Shows quarterly production values.

### Dataset Statistics

Shows:

- number of observations;
- first quarter;
- latest quarter;
- latest production value.

### Historical Residual Count

Shows how many historical Chronos error examples were generated.

### Raw Chronos vs Chronos + GPR Metrics

Displays:

- MAE;
- RMSE;
- sMAPE;
- MASE.

### MAE Change

Shows whether GPR reduced or increased the average Chronos error.

A positive improvement means GPR reduced MAE.

A negative result means GPR increased MAE.

### Horizon-Level Results

Shows how GPR performs at each forecast horizon.

### Historical Forecast Graph

Shows:

```text
Actual
Raw Chronos
Chronos + GPR
```

### Future Forecast Table

Displays:

```text
Raw Chronos
GPR correction
GPR uncertainty
Corrected forecast
```

---

# 25. Why the POC Is Computationally Valid

The POC is considered computationally valid because the major outputs are produced by actual algorithms and actual model execution.

The following components are real:

| Component | Implementation |
|---|---|
| Chronos model | Real pretrained Chronos-T5 Small |
| Chronos inference | Actual model inference |
| Historical forecasts | Generated dynamically |
| Residuals | Calculated from forecast vs actual |
| GPR | Real `GaussianProcessRegressor` |
| GPR training | Performed during execution |
| Residual correction | Predicted by GPR |
| Final forecast | Raw Chronos + predicted residual |
| MAE | Calculated from actual errors |
| RMSE | Calculated from actual errors |
| sMAPE | Calculated from actual errors |
| Horizon evaluation | Calculated separately |
| Interface | Displays actual computational outputs |

The Streamlit application is therefore not simply displaying predefined results.

It acts as the presentation layer for the underlying forecasting and machine-learning pipeline.

---

# 26. Why the Evaluation Procedure Is Valid for a POC

Time-series forecasting cannot be evaluated correctly using information that would not have existed when the prediction was made.

The POC therefore simulates historical forecasting chronologically.

For every forecast origin:

```text
Past observations
      ↓
Chronos forecast
      ↓
Previously known residuals only
      ↓
GPR correction
      ↓
Compare with actual future value
```

The key restriction:

```python
train = out[
    out["target_index"] < origin
]
```

helps prevent GPR from learning from future actual observations.

This makes the evaluation more representative of how the method would operate in a real forecasting situation.

---

# 27. Meaning of "Accuracy" in the POC

Accuracy must be interpreted carefully.

There are three different forms of accuracy relevant to this POC.

---

## A. Implementation Accuracy

The POC accurately implements the proposed method because:

```text
Chronos actually generates forecasts
        ↓
Residuals are actually calculated
        ↓
GPR actually learns those residuals
        ↓
GPR actually produces a correction
        ↓
The correction is actually applied to Chronos
```

This part has been successfully demonstrated.

---

## B. Evaluation Accuracy

The POC calculates forecasting errors from actual model outputs instead of inventing an "accuracy percentage."

The evaluation uses:

```text
MAE
RMSE
sMAPE
MASE
```

and compares the two methods using the same historical forecast cases.

Therefore, if GPR makes the forecasts worse, the application reports the deterioration instead of forcing the proposed method to appear successful.

This behavior supports the validity of the POC.

---

## C. Real-World Forecasting Accuracy

The POC does **not yet establish real Philippine palay forecasting accuracy**.

The included demonstration dataset is synthetic.

Therefore:

```text
POC results
≠
final Philippine palay forecasting results
```

The synthetic dataset verifies the software implementation only.

The final thesis experiment must use official Philippine Statistics Authority data before conclusions about actual forecasting performance can be made.

---

# 28. Why a Negative GPR Result Does Not Invalidate the POC

During the synthetic demonstration, GPR may sometimes worsen Chronos instead of improving it.

For example:

```text
Raw Chronos MAE:
104,588 MT

Chronos + GPR MAE:
141,410 MT
```

This does not indicate that the POC is broken.

It demonstrates that the evaluation is actually measuring the outputs of the models.

If the application were artificially designed to make GPR look better, it would always report an improvement.

Instead, the POC allows the proposed enhancement to:

```text
improve Chronos
have little effect
or
worsen Chronos
```

depending on the learned residual structure.

The actual thesis experiment will determine which of these occurs on official Philippine palay-production data.

---

# 29. Synthetic Demonstration Dataset

The bundled file:

```text
sample_demo_palaysynthetic.csv
```

contains artificially generated quarterly values.

Its only purpose is to answer:

> **Does the complete software pipeline work?**

It is not intended to answer:

> **Does GPR improve Chronos for actual Philippine palay production?**

The synthetic data allow testing of:

- file validation;
- Chronos inference;
- historical forecasting;
- residual generation;
- GPR training;
- GPR correction;
- evaluation metrics; and
- future forecast generation.

The actual research dataset will replace this file during the full thesis experiment.

---

# 30. What the POC Successfully Proves

The functional POC demonstrates that:

1. Chronos-T5 Small can be integrated into the proposed forecasting pipeline.
2. Quarterly production data can be supplied as Chronos context.
3. Historical Chronos predictions can be generated.
4. Residual forecasting errors can be extracted.
5. Residual features can be created.
6. GPR can be trained to model those residual errors.
7. GPR can estimate both a correction and uncertainty.
8. Residual learning can be performed chronologically.
9. The predicted residual can modify a future Chronos forecast.
10. Raw Chronos and Chronos + GPR can be directly evaluated.
11. Performance can be analyzed separately by forecasting horizon.
12. The entire proposed process can run through a functional interface.

Therefore, the proposed method is technically implementable.

---

# 31. What the POC Does Not Claim

The POC does not claim that:

- GPR will always improve Chronos;
- the synthetic dataset represents actual Philippine production;
- the current synthetic performance is the expected thesis result;
- the prototype is an official PSA forecast;
- the prototype is an official DA or NFA forecasting tool;
- the current evaluation is the final thesis experiment; or
- the current interface represents a complete agricultural information system.

The actual effectiveness of the enhancement remains an experimental research question.

---

# 32. Current POC Limitations

The POC is designed to demonstrate feasibility rather than serve as the final research implementation.

Several improvements should be made before final thesis experimentation.

### Official PSA Dataset

The synthetic dataset must be replaced by official quarterly Philippine palay-production data.

### Larger Rolling Evaluation

The number of historical forecast origins should be finalized and expanded according to the thesis evaluation protocol.

### Reproducibility

The final experiment should explicitly record:

```text
Python version
Chronos version
PyTorch version
scikit-learn version
NumPy version
Pandas version
random seeds
model configuration
dataset version
```

Package versions should eventually be pinned in `requirements.txt`.

### Chronos Randomness

Because Chronos produces probabilistic samples, final research experiments should explicitly control random seeds where possible and document the sampling configuration.

### Final MASE Procedure

The MASE denominator should be calculated according to the finalized time-aware training/evaluation design rather than being treated as a final research implementation in the current POC.

### Feature and Kernel Validation

The current GPR features and kernel demonstrate feasibility.

The final thesis must justify or experimentally validate the final feature set and GPR configuration.

---

# 33. Difference Between the POC and the Final Thesis Experiment

## POC

The goal is:

> **Demonstrate that the proposed method can be implemented.**

The POC answers:

```text
Can Chronos run?
Yes.

Can its historical errors be obtained?
Yes.

Can GPR learn those errors?
Yes.

Can GPR generate a correction?
Yes.

Can the correction modify Chronos?
Yes.

Can both versions be evaluated?
Yes.
```

---

## Final Thesis Experiment

The goal will be:

> **Determine whether the proposed GPR residual-modeling enhancement improves Chronos for quarterly Philippine palay-production forecasting.**

The final study will investigate:

```text
How much does it improve or worsen forecasts?

Does the result depend on forecast horizon?

Are improvements consistent?

Are differences statistically meaningful?

Does residual correction generalize across historical forecast periods?
```

These questions cannot be answered using the synthetic POC dataset alone.

---

# 34. POC Validity Summary

The prototype satisfies the definition of a functional Proof of Concept because it demonstrates the complete implementation of the proposed computational contribution.

It is valid as a POC because:

```text
✓ It uses the real Chronos model.
✓ It uses real Gaussian Process Regression.
✓ Chronos predictions are dynamically generated.
✓ Residuals are calculated from actual model errors.
✓ GPR is trained during execution.
✓ The GPR correction is generated by the trained model.
✓ The corrected forecast is dynamically calculated.
✓ Historical evaluation is time-aware.
✓ Performance metrics come from model predictions.
✓ Negative results are not hidden.
✓ The complete pipeline produces measurable outputs.
```

The POC therefore verifies **technical feasibility and implementation correctness**.

It does not yet establish the real-world forecasting superiority of the proposed method.

---

# 35. Conclusion

This Proof of Concept successfully demonstrates the implementation of a **Gaussian Process Regression residual-modeling enhancement for Chronos-T5 Small**.

The complete working process is:

```text
Quarterly production history
          ↓
Real Chronos-T5 Small
          ↓
Raw forecast
          ↓
Historical Chronos residuals
          ↓
Real Gaussian Process Regression
          ↓
Predicted residual correction
          ↓
Chronos + GPR forecast
          ↓
Time-aware performance evaluation
```

The POC therefore establishes that the proposed thesis contribution is technically feasible and capable of producing the intended computational outputs.

The current synthetic dataset is used only for software verification. The full thesis will use official Philippine Statistics Authority quarterly palay-production data to determine whether GPR residual modeling provides a statistically and practically meaningful improvement over the original Chronos forecasts.

The POC should therefore be interpreted as:

> **Evidence that the method can be built and executed, not yet evidence that the method is more accurate on real Philippine palay-production data.**