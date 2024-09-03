# Univariate Time Series Algorithms

- **Autoregressive (AR) Model**: 
  - **Purpose**: Models the current value of the series as a linear combination of its previous values.
  - **Usage**: Effective for time series with autocorrelation, where past values influence future values.

- **Moving Average (MA) Model**: 
  - **Purpose**: Models the current value as a linear function of past forecast errors.
  - **Usage**: Useful for capturing the impact of past forecast errors on current values.

- **Autoregressive Moving Average (ARMA) Model**: 
  - **Purpose**: Combines AR and MA models to account for both past values and past forecast errors.
  - **Usage**: Suitable for stationary time series where both autoregressive and moving average components are needed.

- **Autoregressive Integrated Moving Average (ARIMA) Model**: 
  - **Purpose**: Extends ARMA to include differencing to make the series stationary.
  - **Usage**: Ideal for non-stationary time series that require differencing to achieve stationarity.

- **Seasonal ARIMA (SARIMA) Model**: 
  - **Purpose**: Extends ARIMA to handle seasonal effects.
  - **Usage**: Suitable for time series with seasonal patterns, adding seasonal components to the ARIMA model.

- **State Space Models**: 
  - **Purpose**: Uses state-space representation to model time series, including the Kalman filter for estimation.
  - **Usage**: Provides a flexible framework for modeling time series with various underlying states and transitions.
  - **Variants**:
    - **Autoregressive Integrated Moving Average with Exogenous Variables (ARIMAX) Model**: Extends ARIMA to include exogenous variables (external regressors).
    - **Seasonal ARIMAX (SARIMAX) Model**: Extends SARIMAX to include exogenous variables.
    - **Exponential Smoothing State Space Models (ETS)**: A class of state-space models that includes exponential smoothing methods with components for error, trend, and seasonality.
      - **ETS(A,A,A)**: Additive error, additive trend, additive seasonality
      - **ETS(M,A,A)**: Multiplicative error, additive trend, additive seasonality
      - **ETS(A,M,A)**: Additive error, multiplicative trend, additive seasonality
      - **ETS(A,A,M)**: Additive error, additive trend, multiplicative seasonality

- **Exponential Smoothing Methods (ESM)**: 
  - **Purpose**: A family of forecasting techniques that apply weighted averages to historical data, with more weight given to more recent observations.
  - **Usage**: Adapt quickly to changes in data and provide forecasts that are updated as new observations become available.
  - **Variants**:
    - **Simple Exponential Smoothing (SES)**: Best suited for series without trends or seasonality. It smooths the data with a single smoothing parameter.
    - **Holt’s Linear Trend Method**: Extends SES by adding a component to handle linear trends in the data.
    - **Holt-Winters Seasonal Method**: Further extends Holt's method by incorporating a seasonal component, making it suitable for data with both trend and seasonality.

- **Long Short-Term Memory Networks (LSTM)**: 
  - **Purpose**: A type of Recurrent Neural Network (RNN) designed for sequence prediction problems.
  - **Usage**: Effective for capturing long-term dependencies in time series data.

- **Gated Recurrent Units (GRU)**: 
  - **Purpose**: Another RNN variant, similar to LSTM but with a different gating mechanism.
  - **Usage**: Suitable for sequence prediction tasks, often providing faster training and similar performance to LSTM.

- **Prophet**: 
  - **Purpose**: Developed by Facebook, this model handles seasonality and holidays and is robust to missing data.
  - **Usage**: Ideal for time series with strong seasonal effects and additional holidays, providing flexible and interpretable forecasts.



# Multivariate Time Series Algorithms

- **Vector Autoregression (VAR)**:
  - **Purpose**: Models multiple time series variables simultaneously using lagged values of each variable, capturing interdependencies.
  - **Usage**: Suitable for analyzing and forecasting systems where variables influence each other over time.

- **Vector Autoregressive Moving Average (VARMA)**:
  - **Purpose**: Extends VAR by including moving average components to capture more complex relationships between variables.
  - **Usage**: Ideal for time series data where both autoregressive and moving average effects are present.

- **Vector Autoregressive Integrated Moving Average (VARIMA)**:
  - **Purpose**: Builds on VARMA by incorporating differencing to handle non-stationarity in the data.
  - **Usage**: Effective for time series with trends or seasonality that require differencing to achieve stationarity.

- **Multivariate State Space Models**:
  - **Purpose**: Uses state-space representations to model multiple time series, capturing complex dynamics and relationships.
  - **Usage**: Suitable for time series with intricate dependencies and unobserved components influencing the data.

- **Multivariate Long Short-Term Memory Networks (LSTM)**:
  - **Purpose**: Extends LSTM networks to handle multiple input and output time series, capturing long-term dependencies.
  - **Usage**: Ideal for forecasting and modeling time series with complex temporal patterns and interactions.

- **Multivariate Gated Recurrent Units (GRU)**:
  - **Purpose**: Adapts GRU networks for multiple time series, providing a simpler alternative to LSTMs with similar capabilities.
  - **Usage**: Suitable for handling multiple time series with temporal dependencies while maintaining computational efficiency.

- **Temporal Convolutional Networks (TCN)**:
  - **Purpose**: Uses convolutional layers to model sequences, applying to multiple time series with temporal patterns.
  - **Usage**: Effective for time series forecasting where convolutional approaches offer advantages over recurrent methods.

- **Dynamic Factor Models (DFM)**:
  - **Purpose**: Extracts common factors driving multiple time series, simplifying analysis by reducing dimensionality.
  - **Usage**: Ideal for uncovering underlying factors influencing multiple time series and reducing complexity in high-dimensional data.

- **Canonical Correlation Analysis (CCA)**:
  - **Purpose**: Analyzes the relationships between two sets of time series data, identifying associations and dependencies.
  - **Usage**: Useful for exploring correlations and interactions between different time series datasets.

- **Transfer Function Models**:
  - **Purpose**: Models the relationship between input and output time series, capturing how changes in one series affect another.
  - **Usage**: Suitable for systems where inputs and outputs are related, such as in control systems or economic modeling.

- **Hidden Markov Models (HMM)**:
  - **Purpose**: Models sequences of observed data with underlying hidden states, providing insights into latent structures.
  - **Usage**: Ideal for time series with unobserved states influencing observed data, such as in speech or finance.

- **Dynamic Time Warping (DTW)**:
  - **Purpose**: Measures similarity between two time series that may vary in time or speed, allowing for flexible alignment.
  - **Usage**: Effective for comparing time series with temporal distortions or varying speeds, such as in pattern recognition or anomaly detection.

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

