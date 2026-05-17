# Hands-on with Orange: Time Series Analysis

**Herdiantri Sufriyana**
Graduate Institute of Artificial Intelligence and Big Data in Healthcare
National Taiwan University of Nursing and Health Sciences

---

## Table of Contents

1. [Subtopics](#subtopics)
2. [Prerequisites](#prerequisites)
3. [Step 12: Time Series Fundamentals](#step-12-time-series-fundamentals) — 12a. Load and visualize, 12b. Decomposition, 12c. Autocorrelation
4. [Step 13: Forecasting & Causality](#step-13-forecasting--causality) — 13a. ARIMA forecasting, 13b. VAR multivariate forecasting, 13c. Granger causality

---

## Subtopics

- Loading and visualizing time series data
- Decomposing trend, seasonality, and residuals
- Autocorrelation and periodicity
- Forecasting with ARIMA
- Multivariate time series with VAR
- Granger causality testing

[Back to Table of Contents](#table-of-contents)

---

## Prerequisites

Install the required add-on before this session:
1. Open Orange → **Options** → **Add-ons**
2. Check **Orange3-Timeseries**
3. Click **OK** and restart Orange

Prepare data:
- A **.csv** file with a **date/time column** and one or more numeric columns (e.g., monthly patient counts, daily lab values, weekly admissions)
- Alternatively, use Orange's built-in **Yahoo Finance** widget to load stock data

[Back to Table of Contents](#table-of-contents)

---

## Step 12: Time Series Fundamentals

**12a. Load and visualize**

**Widgets:** File (or Yahoo Finance), As Timeseries, Line Chart

1. Load your time series data:
   - **Option A**: Drag **File** → load your .csv, then connect **File** → **As Timeseries** via **Data**
   - **Option B**: Drag **Yahoo Finance** → enter a stock ticker (e.g., AAPL) and date range, then connect **Yahoo Finance** → **As Timeseries** via **Time series**
2. In **As Timeseries**, set the **time variable** to your date/time column
3. Connect **As Timeseries** → **Line Chart** via **Time series**
   - Select the variable(s) to plot

**Check these:**
- Is there a visible trend (upward/downward over time)?
- Is there a repeating seasonal pattern?
- Are there any sudden jumps or outliers?

**12b. Decomposition**

**Widgets:** Seasonal Adjustment, Line Chart

4. Connect **As Timeseries** → **Seasonal Adjustment** via **Time series**
   - Set **Model type**: **Additive** (try **Multiplicative** if the seasonal amplitude grows over time)
   - Set **Season period** based on your data (e.g., 12 for monthly, 7 for daily with weekly pattern)
5. Connect **Seasonal Adjustment** → **Line Chart** (label it "Decomposition") via **Time series**
   - View the decomposed components: **Trend**, **Seasonal**, **Residual**

**Check these:**
- Does the trend component show a clear direction?
- Is the seasonal pattern consistent across time?
- Are the residuals random (no remaining pattern)?
- Does additive or multiplicative decomposition fit better?

**12c. Autocorrelation**

**Widgets:** Correlogram, Periodogram

6. Connect **As Timeseries** → **Correlogram** via **Time series**
   - The **ACF** (autocorrelation function) shows correlation at each lag
   - The **PACF** (partial autocorrelation) shows direct correlation after removing intermediate lags
   - Significant spikes above the confidence band indicate meaningful lags
7. Connect **As Timeseries** → **Periodogram** via **Time series**
   - Peaks indicate dominant frequencies (periodicities) in the data

**Check these:**
- At which lags does the ACF show significant autocorrelation?
- Does the ACF decay slowly (suggests differencing needed) or cut off sharply?
- Does the PACF cut off after a few lags (suggests AR order)?
- What is the dominant period in the Periodogram?

[Back to Table of Contents](#table-of-contents)

---

## Step 13: Forecasting & Causality

**13a. ARIMA forecasting**

**Widgets:** ARIMA Model, Line Chart, Model Evaluation

ARIMA has three parameters:
- **p** (AR order) — number of autoregressive lags (guided by PACF cutoff)
- **d** (differencing) — number of times to difference (0 if stationary, 1 if trending)
- **q** (MA order) — number of moving average lags (guided by ACF cutoff)

8. Connect **As Timeseries** → **ARIMA Model** via **Time series**
   - Set **p**, **d**, **q** based on the ACF/PACF from Step 12c
   - Set **Forecast steps** (e.g., 12 for 12 periods ahead)
9. Connect **ARIMA Model** → **Line Chart** (label it "ARIMA forecast") via **Time series**
   - The forecast is overlaid on the original series with confidence intervals
10. Connect **ARIMA Model** → **Model Evaluation** via **Time series**
    - Note the error metrics: **RMSE**, **MAE**, **MAPE**

**Check these:**
- Does the forecast follow the trend and seasonality?
- Are the confidence intervals reasonable or very wide?
- Try different p, d, q values — does the RMSE improve?

**13b. VAR multivariate forecasting**

**Widgets:** VAR Model, Line Chart, Model Evaluation

VAR extends ARIMA to multiple variables — each variable is predicted using its own lags **and** the lags of other variables.

11. Ensure your data has **two or more numeric columns** in **As Timeseries**
12. Connect **As Timeseries** → **VAR Model** via **Time series**
    - Set **Forecast steps** (same as ARIMA for comparison)
    - Set **Maxlags** (e.g., 5–10)
13. Connect **VAR Model** → **Line Chart** (label it "VAR forecast") via **Time series**
14. Connect **VAR Model** → **Model Evaluation** via **Time series**
    - Compare RMSE with ARIMA — does adding other variables improve the forecast?

**Check these:**
- Does the VAR forecast outperform ARIMA?
- Which variable benefits most from including the other variables?

**13c. Granger causality**

**Widgets:** Granger Causality, Data Table

Granger causality tests whether past values of variable X help predict variable Y beyond what Y's own past values provide. It does not prove true causation but indicates predictive temporal relationships.

15. Connect **As Timeseries** → **Granger Causality** via **Time series**
    - Set **Max lag** (e.g., 5)
    - Set **Significance level**: **0.05**
16. Connect **Granger Causality** → **Data Table** via **Data**
    - Each row shows a variable pair and its p-value
    - p-value ≤ 0.05 → X Granger-causes Y (X's past helps predict Y)

**Check these:**
- Which variable pairs show significant Granger causality?
- Is the relationship unidirectional (X → Y only) or bidirectional (X ↔ Y)?
- How does this compare to the correlation-based causal discovery from Meeting 06? Granger causality uses temporal ordering, while correlation-based methods do not.

[Back to Table of Contents](#table-of-contents)
