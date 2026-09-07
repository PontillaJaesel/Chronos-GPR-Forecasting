
# POC Explanation for the Dean / Panel

## Proposed thesis
**Gaussian Process Residual Modeling Enhancement of Chronos for Quarterly Philippine Palay Production Forecasting**

## What is being improved?
Chronos is the existing forecasting method.

The thesis adds **Gaussian Process Regression (GPR) residual modeling** after Chronos.

The proposed process is:

**Palay history → Chronos → Raw forecast → GPR error correction → Final forecast**

## What does GPR do?
During historical testing, the program records:

**Residual = Actual production − Chronos prediction**

GPR is trained to learn patterns in those remaining errors.

When Chronos makes a new forecast, GPR estimates the error that Chronos is likely to leave behind. That estimated error is added to the original forecast.

## What does the POC demonstrate?
The working prototype:

1. loads quarterly palay data;
2. runs Chronos on historical periods;
3. creates a dataset of Chronos forecasting errors;
4. trains GPR on past residual errors only;
5. compares Raw Chronos against Chronos + GPR;
6. produces a new forecast showing:
   - original Chronos prediction;
   - GPR estimated correction;
   - corrected prediction.

## Why is this a Computer Science contribution?
The thesis is not just asking which existing model is better.

It integrates an established residual-learning technique with a pretrained time-series foundation model and experimentally determines whether this enhancement reduces Chronos forecasting error on a small Philippine agricultural time series.

## What is NOT being claimed?
- The group did not invent Chronos.
- The group did not invent Gaussian Process Regression.
- The POC is not an official agricultural forecasting system.
- Improvement is not guaranteed; the experiment determines whether the enhancement works and for which forecast horizons.

## Minimum successful POC
A successful proof of concept is one where the software can:

**input quarterly data → run Chronos → learn historical Chronos residuals with GPR → output a corrected forecast**

even before the final full thesis evaluation is completed.
