'''
Tasks:
	1.	From your Math write-up, implement in features.py:
	•	RSI (14-period)
	•	Bollinger Bands (20-period ±2σ)
	•	MACD (12-/26-EMA & 9-EMA signal)
	•	VWAP (per minute)
'''

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


class TechnicalIndicators:
    """
    Technical indicators for financial time series analysis.
    
    This class implements various technical analysis indicators including:
    - RSI (Relative Strength Index)
    - Bollinger Bands
    - MACD (Moving Average Convergence Divergence)
    - VWAP (Volume Weighted Average Price)
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with price/volume data.
        
        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = data.copy()
        
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI measures the speed and change of price movements.
        Values above 70 typically indicate overbought conditions,
        values below 30 indicate oversold conditions.
        
        Args:
            period: Number of periods for RSI calculation (default: 14)
            
        Returns:
            pd.Series: RSI values
        """
        close = self.data['close']
        delta = close.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate exponential moving averages of gains and losses
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()
        
        # Calculate Relative Strength (RS), handling division by zero
        rs = avg_gains / avg_losses.replace(0, np.nan)
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Bollinger Bands consist of:
        - Middle Band: Simple Moving Average
        - Upper Band: Middle Band + (std_dev * standard deviation)
        - Lower Band: Middle Band - (std_dev * standard deviation)
        
        Args:
            period: Number of periods for moving average (default: 20)
            std_dev: Number of standard deviations (default: 2.0)
            
        Returns:
            Dict containing 'upper', 'middle', 'lower' bands
        """
        close = self.data['close']
        
        # Calculate middle band (Simple Moving Average)
        middle = close.rolling(window=period).mean()
        
        # Calculate rolling standard deviation
        std = close.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD consists of:
        - MACD Line: 12-period EMA - 26-period EMA
        - Signal Line: 9-period EMA of MACD Line
        - Histogram: MACD Line - Signal Line
        
        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
            
        Returns:
            Dict containing 'macd', 'signal', 'histogram'
        """
        close_series = self.data['close']
        
        # Calculate fast and slow EMAs
        fast_ema = exponential_moving_average(close_series, fast_period)
        slow_ema = exponential_moving_average(close_series, slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD)
        signal_line = exponential_moving_average(macd_line, signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_vwap(self) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        VWAP is calculated as the cumulative sum of (price * volume) 
        divided by cumulative volume for each minute/period.
        
        Typical price = (high + low + close) / 3
        
        Returns:
            pd.Series: VWAP values
        """
        # Calculate typical price
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        
        # Calculate price * volume
        pv = typical_price * self.data['volume']
        
        # Calculate cumulative sums
        cumulative_pv = pv.cumsum()
        cumulative_volume = self.data['volume'].cumsum()
        
        # Calculate VWAP, avoiding division by zero
        vwap = cumulative_pv / cumulative_volume.replace(0, np.nan)
        
        return vwap.ffill()
    
    def calculate_all_features(self) -> pd.DataFrame:
        """
        Calculate all technical indicators and return as DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with all calculated features
        """
        features_df = self.data.copy()
        
        # Calculate RSI
        features_df['rsi'] = self.calculate_rsi()
        
        # Calculate Bollinger Bands
        bollinger = self.calculate_bollinger_bands()
        features_df['bb_upper'] = bollinger['upper']
        features_df['bb_middle'] = bollinger['middle']
        features_df['bb_lower'] = bollinger['lower']
        
        # Calculate MACD
        macd = self.calculate_macd()
        features_df['macd'] = macd['macd']
        features_df['macd_signal'] = macd['signal']
        features_df['macd_histogram'] = macd['histogram']
        
        # Calculate VWAP
        features_df['vwap'] = self.calculate_vwap()
        
        return features_df


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create all technical indicators.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with all features
    """
    indicators = TechnicalIndicators(data)
    return indicators.calculate_all_features()


# Helper functions for common calculations
def exponential_moving_average(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        series: Price series
        period: Number of periods
        
    Returns:
        pd.Series: EMA values
    """
    return series.ewm(span=period, adjust=False).mean()


def simple_moving_average(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        series: Price series
        period: Number of periods
        
    Returns:
        pd.Series: SMA values
    """
    return series.rolling(window=period).mean()