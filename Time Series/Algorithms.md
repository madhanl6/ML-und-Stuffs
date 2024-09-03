# Univariate Time Series Algorithms

### **Autoregressive (AR) Model**
- **Purpose**: Models the current value of the series as a linear combination of its previous values.
- **Usage**: Effective for time series with autocorrelation, where past values influence future values.

### **Moving Average (MA) Model**
- **Purpose**: Models the current value as a linear function of past forecast errors.
- **Usage**: Useful for capturing the impact of past forecast errors on current values.

### **Autoregressive Moving Average (ARMA) Model**
- **Purpose**: Combines AR and MA models to account for both past values and past forecast errors.
- **Usage**: Suitable for stationary time series where both autoregressive and moving average components are needed.

### **Autoregressive Integrated Moving Average (ARIMA) Model**
- **Purpose**: Extends ARMA to include differencing to make the series stationary.
- **Usage**: Ideal for non-stationary time series that require differencing to achieve stationarity.

### **Seasonal ARIMA (SARIMA) Model**
- **Purpose**: Extends ARIMA to handle seasonal effects.
- **Usage**: Suitable for time series with seasonal patterns, adding seasonal components to the ARIMA model.

### **State Space Models**
- **Purpose**: Uses state-space representation to model time series, including the Kalman filter for estimation.
- **Usage**: Provides a flexible framework for modeling time series with various underlying states and transitions.

### **Autoregressive Integrated Moving Average with Exogenous Variables (ARIMAX) Model**
- **Purpose**: Extends ARIMA to include exogenous variables (external regressors).
- **Usage**: Useful for incorporating external factors that affect the time series.

### **Seasonal ARIMAX (SARIMAX) Model**
- **Purpose**: Extends SARIMAX to include exogenous variables.
- **Usage**: Combines seasonal ARIMA with external regressors for forecasting with seasonal patterns and additional influencing factors.

### **Exponential Smoothing State Space Models (ETS)**
- **Purpose**: A class of state space models that includes exponential smoothing methods with components for error, trend, and seasonality.
- **Types**:
  - **ETS(A,A,A)**: Additive error, additive trend, additive seasonality
  - **ETS(M,A,A)**: Multiplicative error, additive trend, additive seasonality
  - **ETS(A,M,A)**: Additive error, multiplicative trend, additive seasonality
  - **ETS(A,A,M)**: Additive error, additive trend, multiplicative seasonality

### **Exponential Smoothing Methods (ESM)**
- **Purpose**: A family of forecasting techniques that apply weighted averages to historical data, with more weight given to more recent observations.
- **Usage**: Adapt quickly to changes in data and provide forecasts that are updated as new observations become available.

### **Simple Exponential Smoothing (SES)**
- **Purpose**: Best suited for series without trends or seasonality. It smooths the data with a single smoothing parameter.
- **Usage**: Ideal for series where only recent observations are relevant for forecasting.

### **Holt’s Linear Trend Method**
- **Purpose**: Extends SES by adding a component to handle linear trends in the data.
- **Usage**: Useful for time series with a linear trend component.

### **Holt-Winters Seasonal Method**
- **Purpose**: Further extends Holt's method by incorporating a seasonal component, making it suitable for data with both trend and seasonality.
- **Usage**: Best for time series with both trend and seasonal patterns.

### **Long Short-Term Memory Networks (LSTM)**
- **Purpose**: A type of Recurrent Neural Network (RNN) designed for sequence prediction problems.
- **Usage**: Effective for capturing long-term dependencies in time series data.

### **Gated Recurrent Units (GRU)**
- **Purpose**: Another RNN variant, similar to LSTM but with a different gating mechanism.
- **Usage**: Suitable for sequence prediction tasks, often providing faster training and similar performance to LSTM.

### **Prophet**
- **Purpose**: Developed by Facebook, this model handles seasonality and holidays and is robust to missing data.
- **Usage**: Ideal for time series with strong seasonal effects and additional holidays, providing flexible and interpretable forecasts.


# Multivariate Time Series Algorithms

- **Vector Autoregression (VAR)**: Models multiple time series variables simultaneously with lagged values of each variable.
- **Vector Autoregressive Moving Average (VARMA)**: Extends VAR to include moving average components.
- **Vector Autoregressive Integrated Moving Average (VARIMA)**: Extends VARMA with differencing to handle non-stationarity.
- **Dynamic Factor Models (DFM)**: Extracts common factors driving multiple time series.
- **Canonical Correlation Analysis (CCA)**: Analyzes the relationship between two sets of time series data.
- **Multivariate State Space Models**: Uses state-space representation to model multiple time series.
- **Multivariate Long Short-Term Memory Networks (LSTM)**: Extends LSTM to handle multiple input and output time series.
- **Multivariate Gated Recurrent Units (GRU)**: Extends GRU to manage multiple time series.
- **Temporal Convolutional Networks (TCN)**: Uses convolutional layers for sequence modeling in multiple time series.
- **Transfer Function Models**: Models the relationship between input and output time series.
- **Hidden Markov Models (HMM)**: Models sequences of observed data with underlying hidden states.
- **Dynamic Time Warping (DTW)**: Measures similarity between two time series which may vary in time or speed.

These algorithms can be applied depending on the specific requirements of the time series data you are working with, such as the need for seasonal decomposition, handling of multiple variables, or capturing long-term dependencies.

## Choosing the Right Library

- **For traditional statistical models**: 
  - `statsmodels`
  - `pmdarima`
  - `Prophet`
  - `Darts`

- **For deep learning models**: 
  - `TensorFlow`
  - `PyTorch`

- **For feature extraction**: 
  - `TSFEL`
  - `tsfresh`

- **For machine learning and boosting**: 
  - `xgboost`
  - `catBoost`
  - `sklearn`

- **For time series classification and clustering**: 
  - `pyts`
  - `tslearn`
  - `sktime`

By using these libraries, you can effectively tackle various aspects of time series analysis and forecasting.

