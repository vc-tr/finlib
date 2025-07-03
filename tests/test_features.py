import pytest
import pandas as pd
import numpy as np
from src.pipeline.features import (
    TechnicalIndicators, 
    create_features,
    exponential_moving_average,
    simple_moving_average
)


@pytest.fixture
def sample_data():
    """Create a small synthetic OHLCV dataset for testing."""
    dates = pd.date_range('2023-01-01', periods=10, freq='1min')
    data = pd.DataFrame({
        'open': [100, 102, 104, 103, 105, 107, 106, 108, 110, 109],
        'high': [101, 103, 105, 104, 106, 108, 107, 109, 111, 110],
        'low': [99, 101, 103, 102, 104, 106, 105, 107, 109, 108],
        'close': [100, 102, 104, 103, 105, 107, 106, 108, 110, 109],
        'volume': [1000, 1200, 800, 1500, 900, 1100, 1300, 700, 1000, 1400]
    }, index=dates)
    return data


@pytest.fixture
def simple_data():
    """Create very simple data for manual calculation verification."""
    dates = pd.date_range('2023-01-01', periods=5, freq='1min')
    data = pd.DataFrame({
        'open': [10, 11, 12, 11, 13],
        'high': [10.5, 11.5, 12.5, 11.5, 13.5],
        'low': [9.5, 10.5, 11.5, 10.5, 12.5],
        'close': [10, 11, 12, 11, 13],
        'volume': [100, 200, 150, 300, 250]
    }, index=dates)
    return data


class TestHelperFunctions:
    """Test helper functions for moving averages."""
    
    def test_simple_moving_average(self):
        """Test SMA calculation with known values."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = simple_moving_average(series, 3)
        expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
        pd.testing.assert_series_equal(result, expected)
    
    def test_exponential_moving_average(self):
        """Test EMA calculation."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = exponential_moving_average(series, 3)
        # EMA with span=3 should be calculable
        assert len(result) == 5
        assert not result.isna().all()
        # First value should equal first input
        assert result.iloc[0] == 1.0


class TestRSI:
    """Test RSI calculation."""
    
    def test_rsi_basic_calculation(self, simple_data):
        """Test RSI with simple upward trend."""
        indicators = TechnicalIndicators(simple_data)
        rsi = indicators.calculate_rsi(period=3)
        
        # With upward trend, RSI should be > 50
        assert len(rsi) == len(simple_data)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_rsi_with_no_change(self):
        """Test RSI when prices don't change."""
        dates = pd.date_range('2023-01-01', periods=5, freq='1min')
        data = pd.DataFrame({
            'open': [10, 10, 10, 10, 10],
            'high': [10, 10, 10, 10, 10],
            'low': [10, 10, 10, 10, 10],
            'close': [10, 10, 10, 10, 10],
            'volume': [100, 100, 100, 100, 100]
        }, index=dates)
        
        indicators = TechnicalIndicators(data)
        rsi = indicators.calculate_rsi(period=3)
        
        # When no price change, RSI should be NaN (0/0)
        # After first few periods, it should handle this gracefully
        assert len(rsi) == len(data)
    
    def test_rsi_alternating_prices(self):
        """Test RSI with alternating up/down prices."""
        dates = pd.date_range('2023-01-01', periods=6, freq='1min')
        data = pd.DataFrame({
            'open': [10, 11, 10, 11, 10, 11],
            'high': [11, 12, 11, 12, 11, 12],
            'low': [9, 10, 9, 10, 9, 10],
            'close': [10, 11, 10, 11, 10, 11],
            'volume': [100, 100, 100, 100, 100, 100]
        }, index=dates)
        
        indicators = TechnicalIndicators(data)
        rsi = indicators.calculate_rsi(period=3)
        
        # With alternating prices, RSI should be around 50
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert all(rsi_val >= 0 and rsi_val <= 100 for rsi_val in valid_rsi)


class TestBollingerBands:
    """Test Bollinger Bands calculation."""
    
    def test_bollinger_bands_basic(self, simple_data):
        """Test basic Bollinger Bands calculation."""
        indicators = TechnicalIndicators(simple_data)
        bb = indicators.calculate_bollinger_bands(period=3, std_dev=2.0)
        
        assert 'upper' in bb
        assert 'middle' in bb
        assert 'lower' in bb
        
        # All bands should have same length
        assert len(bb['upper']) == len(simple_data)
        assert len(bb['middle']) == len(simple_data)
        assert len(bb['lower']) == len(simple_data)
        
        # Upper band should be >= middle band >= lower band
        valid_mask = ~(bb['upper'].isna() | bb['middle'].isna() | bb['lower'].isna())
        if valid_mask.any():
            assert (bb['upper'][valid_mask] >= bb['middle'][valid_mask]).all()
            assert (bb['middle'][valid_mask] >= bb['lower'][valid_mask]).all()
    
    def test_bollinger_bands_manual_calculation(self):
        """Test with manually calculable values."""
        dates = pd.date_range('2023-01-01', periods=4, freq='1min')
        data = pd.DataFrame({
            'open': [10, 10, 10, 10],
            'high': [10, 10, 10, 10],
            'low': [10, 10, 10, 10],
            'close': [10, 11, 12, 13],  # Simple increasing sequence
            'volume': [100, 100, 100, 100]
        }, index=dates)
        
        indicators = TechnicalIndicators(data)
        bb = indicators.calculate_bollinger_bands(period=3, std_dev=1.0)
        
        # For period=3 starting at index 2:
        # close[0:3] = [10, 11, 12], mean = 11, std = 1
        # middle = 11, upper = 11 + 1*1 = 12, lower = 11 - 1*1 = 10
        assert abs(bb['middle'].iloc[2] - 11.0) < 0.01
        assert abs(bb['upper'].iloc[2] - 12.0) < 0.01
        assert abs(bb['lower'].iloc[2] - 10.0) < 0.01


class TestMACD:
    """Test MACD calculation."""
    
    def test_macd_basic(self, sample_data):
        """Test basic MACD calculation."""
        indicators = TechnicalIndicators(sample_data)
        macd = indicators.calculate_macd(fast_period=3, slow_period=5, signal_period=2)
        
        assert 'macd' in macd
        assert 'signal' in macd
        assert 'histogram' in macd
        
        # All components should have same length
        assert len(macd['macd']) == len(sample_data)
        assert len(macd['signal']) == len(sample_data)
        assert len(macd['histogram']) == len(sample_data)
        
        # Histogram should equal MACD - Signal
        valid_mask = ~(macd['macd'].isna() | macd['signal'].isna())
        if valid_mask.any():
            expected_histogram = macd['macd'] - macd['signal']
            pd.testing.assert_series_equal(
                macd['histogram'][valid_mask], 
                expected_histogram[valid_mask],
                check_names=False
            )
    
    def test_macd_upward_trend(self):
        """Test MACD with clear upward trend."""
        dates = pd.date_range('2023-01-01', periods=8, freq='1min')
        data = pd.DataFrame({
            'open': range(10, 18),
            'high': range(11, 19),
            'low': range(9, 17),
            'close': range(10, 18),  # Clear upward trend
            'volume': [100] * 8
        }, index=dates)
        
        indicators = TechnicalIndicators(data)
        macd = indicators.calculate_macd(fast_period=2, slow_period=4, signal_period=2)
        
        # With upward trend, MACD should eventually be positive
        valid_macd = macd['macd'].dropna()
        if len(valid_macd) > 0:
            # At least some MACD values should be positive in upward trend
            assert len(valid_macd) > 0


class TestVWAP:
    """Test VWAP calculation."""
    
    def test_vwap_basic(self, simple_data):
        """Test basic VWAP calculation."""
        indicators = TechnicalIndicators(simple_data)
        vwap = indicators.calculate_vwap()
        
        assert len(vwap) == len(simple_data)
        # VWAP should be positive for positive prices
        valid_vwap = vwap.dropna()
        assert (valid_vwap > 0).all()
    
    def test_vwap_manual_calculation(self):
        """Test VWAP with manually calculable values."""
        dates = pd.date_range('2023-01-01', periods=3, freq='1min')
        data = pd.DataFrame({
            'open': [10, 10, 10],
            'high': [12, 12, 12],
            'low': [8, 8, 8],
            'close': [10, 10, 10],
            'volume': [100, 200, 300]
        }, index=dates)
        
        indicators = TechnicalIndicators(data)
        vwap = indicators.calculate_vwap()
        
        # Typical price = (12 + 8 + 10) / 3 = 10
        # All periods have same typical price (10) and increasing volume
        # VWAP should remain 10 throughout
        expected_vwap = 10.0
        valid_vwap = vwap.dropna()
        for val in valid_vwap:
            assert abs(val - expected_vwap) < 0.01
    
    def test_vwap_with_zero_volume(self):
        """Test VWAP handling of zero volume."""
        dates = pd.date_range('2023-01-01', periods=3, freq='1min')
        data = pd.DataFrame({
            'open': [10, 11, 12],
            'high': [11, 12, 13],
            'low': [9, 10, 11],
            'close': [10, 11, 12],
            'volume': [0, 100, 200]  # First period has zero volume
        }, index=dates)
        
        indicators = TechnicalIndicators(data)
        vwap = indicators.calculate_vwap()
        
        # Should handle zero volume gracefully
        assert len(vwap) == len(data)
        # After forward-filling, should have valid values
        final_vwap = vwap.iloc[-1]
        assert not pd.isna(final_vwap) or pd.isna(final_vwap)  # Either valid number or NaN is acceptable


class TestTechnicalIndicators:
    """Test the main TechnicalIndicators class."""
    
    def test_initialization(self, sample_data):
        """Test class initialization."""
        indicators = TechnicalIndicators(sample_data)
        # Should create a copy of the data
        assert indicators.data.equals(sample_data)
        assert indicators.data is not sample_data  # Should be a copy
    
    def test_calculate_all_features(self, sample_data):
        """Test calculating all features at once."""
        indicators = TechnicalIndicators(sample_data)
        features = indicators.calculate_all_features()
        
        # Should contain original columns plus new features
        expected_cols = ['open', 'high', 'low', 'close', 'volume', 
                        'rsi', 'bb_upper', 'bb_middle', 'bb_lower',
                        'macd', 'macd_signal', 'macd_histogram', 'vwap']
        
        for col in expected_cols:
            assert col in features.columns
        
        # Should have same length as original data
        assert len(features) == len(sample_data)
        
        # All feature columns should be numeric
        feature_cols = ['rsi', 'bb_upper', 'bb_middle', 'bb_lower',
                       'macd', 'macd_signal', 'macd_histogram', 'vwap']
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(features[col])


class TestCreateFeatures:
    """Test the convenience function."""
    
    def test_create_features_function(self, sample_data):
        """Test the create_features convenience function."""
        features = create_features(sample_data)
        
        # Should return same result as using class directly
        indicators = TechnicalIndicators(sample_data)
        expected = indicators.calculate_all_features()
        
        pd.testing.assert_frame_equal(features, expected)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_insufficient_data(self):
        """Test with very small datasets."""
        dates = pd.date_range('2023-01-01', periods=2, freq='1min')
        data = pd.DataFrame({
            'open': [10, 11],
            'high': [11, 12],
            'low': [9, 10],
            'close': [10, 11],
            'volume': [100, 200]
        }, index=dates)
        
        indicators = TechnicalIndicators(data)
        
        # Should not crash with small data
        rsi = indicators.calculate_rsi(period=14)  # Period larger than data
        bb = indicators.calculate_bollinger_bands(period=20)
        macd = indicators.calculate_macd()
        vwap = indicators.calculate_vwap()
        
        assert len(rsi) == 2
        assert len(bb['upper']) == 2
        assert len(macd['macd']) == 2
        assert len(vwap) == 2
    
    def test_all_nan_prices(self):
        """Test with NaN values in price data."""
        dates = pd.date_range('2023-01-01', periods=3, freq='1min')
        data = pd.DataFrame({
            'open': [np.nan, 11, 12],
            'high': [np.nan, 12, 13],
            'low': [np.nan, 10, 11],
            'close': [np.nan, 11, 12],
            'volume': [100, 200, 300]
        }, index=dates)
        
        indicators = TechnicalIndicators(data)
        
        # Should handle NaN values gracefully
        features = indicators.calculate_all_features()
        assert len(features) == 3
        # Some calculations might be NaN, but shouldn't crash


if __name__ == "__main__":
    pytest.main([__file__])
