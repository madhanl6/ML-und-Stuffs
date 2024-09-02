# Univariate Time Series Algorithms

- **Autoregressive (AR) Model**: Models the current value of the series as a linear combination of its previous values.
- **Moving Average (MA) Model**: Models the current value as a linear function of past forecast errors.
- **Autoregressive Moving Average (ARMA) Model**: Combines AR and MA models to account for both past values and past forecast errors.
- **Autoregressive Integrated Moving Average (ARIMA) Model**: Extends ARMA to include differencing to make the series stationary.
- **Seasonal ARIMA (SARIMA) Model**: Extends ARIMA to handle seasonal effects.
- **Exponential Smoothing Methods (ESM)**: A family of forecasting techniques that apply weighted averages to historical data, with more weight given to more recent observations. These methods adapt quickly to changes in the data and provide forecasts that are updated as new observations are available. There are several variations:
  - **Simple Exponential Smoothing**: Suitable for data without trend or seasonality, calculating forecasts based on a weighted average of past observations.
  - **Holt’s Linear Trend Model**: Extends Simple Exponential Smoothing to account for linear trends in the data with components for level and trend.
  - **Holt-Winters Seasonal Model**: Further extends Holt’s model by incorporating seasonal effects, available in:
    - **Additive**: For constant seasonal fluctuations.
    - **Multiplicative**: For seasonal variations that change proportionally with the data level.
- **State Space Models**: Uses state-space representation to model time series, including the Kalman filter for estimation.
- **Prophet**: Developed by Facebook, this model handles seasonality and holidays and is robust to missing data.
- **Long Short-Term Memory Networks (LSTM)**: A type of Recurrent Neural Network (RNN) designed for sequence prediction problems.
- **Gated Recurrent Units (GRU)**: Another RNN variant, similar to LSTM but with a different gating mechanism.
- **Seasonal Decomposition of Time Series (STL)**: Decomposes time series into seasonal, trend, and residual components.
- **Fourier Analysis**: Uses Fourier transforms to decompose time series into frequency components.
- **Wavelet Analysis**: Decomposes time series into wavelets for analyzing localized variations.

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

