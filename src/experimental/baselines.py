# src/pipeline/baselines.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def persistence_forecast(series: pd.Series) -> pd.Series:
    """
    Predict next value = current value (naïve).
    Returns a series of forecasts aligned to series.index[1:].
    """
    return series.shift(1).dropna()

def moving_average_forecast(series: pd.Series, window: int = 5) -> pd.Series:
    """
    Predict next value = rolling mean of last `window` values.
    Forecasts aligned to series.index[window:].
    """
    return series.rolling(window).mean().shift(1).dropna()

def arima_forecast(series: pd.Series, order=(1,0,0)) -> pd.Series:
    """
    Fit ARIMA(order) on the series, then one-step-ahead forecasts.
    This refits at each step (walk-forward). For speed, we’ll fit once on train.
    Returns forecast series for the test portion.
    """
    try:
        # Use last 10 observations for testing to avoid NaN issues
        test_size = 10
        train = series[:-test_size]
        test = series[-test_size:]
        
        # Fit ARIMA model on training data with plain numeric index
        train_values = train.values
        model = ARIMA(train_values, order=order).fit()
        
        # Get simple forecast without complex indexing
        forecast_values = model.forecast(steps=test_size)
        
        # Convert to pandas Series with proper index
        forecast = pd.Series(forecast_values, index=test.index)
        return forecast
        
    except Exception as e:
        print(f"ARIMA failed: {e}, returning persistence as fallback")
        # Return simple persistence forecast as fallback
        test_size = 10
        return pd.Series([series.iloc[-test_size-1]] * test_size, index=series.iloc[-test_size:].index)