This project focuses on forecasting temperature data using ARIMA and SARIMA models. We analyze temperature data, perform necessary transformations, and compare the performance of ARIMA and SARIMA models. The dataset contains monthly temperature data from January 2014 to December 2023.

## Objective

The main goal is to fit and find the relatively best statistical time series model for forecasting temperature and to compare the performance of ARIMA and SARIMA models based on predictive accuracy using metrics like MAE and RMSE.

## Features

- **Data Analysis**: Preliminary exploration of the data, including plotting, identifying trends, seasonality, and stationarity.
- **ARIMA and SARIMA Modelling**: Using grid search to find the optimal parameters for ARIMA and SARIMA models.
- **Forecasting**: Applying the models to forecast future temperature values.
- **Model Comparison**: Evaluating and comparing ARIMA and SARIMA based on predictive performance (MAE, RMSE) and residual diagnostics.

## Installation

To run this project, ensure you have Python installed along with the following packages:

## Methodology

### 1. Data

- **Data Source**: Monthly temperature data for the period of 2014 to 2023 from ERA5. The data is at Python-Projects/Project-1/data/era5_data_IND.nc

### 2. Model Selection and Optimization

The project utilizes the `arima_sarima_best_fit()` function to find the best model parameters using a grid search over the following parameters:

- **ARIMA**: Optimized over a range of (p,d,q) values.
- **SARIMA**: Optimized over both seasonal and non-seasonal parameters (P,D,Q,S) and (p,d,q).

The function parameters include:

- `ts`: The time series data
- `pdq`: List of (p,d,q) tuples for ARIMA
- `pdqs`: List of seasonal (P,D,Q,S) tuples for SARIMA
- `model_type`: Choose between 'ARIMA' or 'SARIMA'

### 3. Model Fitting

For each of the ARIMA and SARIMA models, the project:

- Fits the model using optimal parameters found via grid search.
- Summarizes the model using key statistical metrics like AIC, BIC, Heteroskedasticity, and Skewness.
- Ensures residuals are checked for white noise characteristics using diagnostics like Ljung-Box and Jarque-Bera tests.

### 4. Performance Metrics

To evaluate the models, we use the following metrics:

- **Mean Absolute Error (MAE)**: Measures average magnitude of errors in a set of predictions.
- **Root Mean Squared Error (RMSE)**: Gives more weight to larger errors compared to MAE.
  
### 5. Results and Discussion

- **ARIMA vs SARIMA**: ARIMA was a simpler model but SARIMA significantly outperformed ARIMA in terms of MAE and RMSE due to its ability to model seasonality more effectively.
  
  **Comparison results**:
  - ARIMA: MAE = 2.49, RMSE = 3.05
  - SARIMA: MAE = 0.70, RMSE = 0.86

- **Interpretation of SARIMAX Results**: Although SARIMA had statistically non-significant parameters, it still provided better predictive accuracy, underscoring the trade-off between interpretability and accuracy.

### 6. Stationarity and Trend Analysis

To determine whether the data was stationary, we applied the Augmented Dickey-Fuller (ADF) Test. If the p-value was greater than 0.05, it indicated non-stationarity, prompting us to perform first- and second-order differencing.

## Conclusion

The SARIMA model proved to be the more effective model for temperature forecasting due to its ability to handle both seasonal and non-seasonal components. While ARIMA provided simpler results, its predictive accuracy was lower.

---

### Note

The AIC values might differ between model outputs due to differences in the optimization process, convergence issues, or random initialization during parameter estimation. For reliable results, always ensure that the model's diagnostic metrics and convergence criteria are met.

---
