# Create a new file: scripts/qc_analysis.py
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.data_fetcher_yahoo import YahooDataFetcher

def fetch_spy_data():
    """Step 1: Fetch 5 days of SPY 1-min data"""
    fetcher = YahooDataFetcher(max_retries=3, retry_delay=1.0)
    
    # Fetch 5 days of 1-minute data
    df = fetcher.fetch_ohlcv("SPY", "1m", period="5d")
    
    print(f"Fetched {len(df)} rows of SPY 1-min data")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Save raw data for analysis
    df.to_csv("data/raw_spy_1min.csv")
    print("Saved raw data to data/raw_spy_1min.csv")
    
    return df

if __name__ == "__main__":
    raw_data = fetch_spy_data()
    
def analyze_timestamp_quality(df: pd.DataFrame):
    """Step 2: Analyze raw timestamps for quality issues"""
    
    print("\n" + "="*60)
    print("TIMESTAMP QUALITY ANALYSIS")
    print("="*60)
    
    # Basic info
    print(f"\n1. BASIC TIMESTAMP INFO:")
    print(f"   Total rows: {len(df)}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Timezone: {df.index.tz}")
    
    # Check for duplicates
    print(f"\n2. DUPLICATE ANALYSIS:")
    duplicates = df.index.duplicated()
    print(f"   Duplicate timestamps: {duplicates.sum()}")
    if duplicates.any():
        print(f"   Duplicate times: {df.index[duplicates].tolist()}")
    
    # Analyze by trading day
    print(f"\n3. DAILY BREAKDOWN:")
    daily_analysis = []
    
    # Group by trading date (remove time component)
    df['date'] = df.index.date
    
    for date, day_data in df.groupby('date'):
        day_info = analyze_single_day(day_data, date)
        daily_analysis.append(day_info)
        
        print(f"   {date}: {len(day_data)} bars, "
              f"{day_info['missing_minutes']} missing, "
              f"{day_info['gaps']} gaps, "
              f"trading hours: {day_info['first_bar']} - {day_info['last_bar']}")
    
    # Overall statistics
    print(f"\n4. MISSING DATA SUMMARY:")
    total_missing = sum(day['missing_minutes'] for day in daily_analysis)
    total_gaps = sum(day['gaps'] for day in daily_analysis)
    
    print(f"   Total missing minutes across all days: {total_missing}")
    print(f"   Total gaps detected: {total_gaps}")
    print(f"   Average missing minutes per day: {total_missing/len(daily_analysis):.1f}")
    
    # DST analysis
    print(f"\n5. DST ANALYSIS:")
    analyze_dst_issues(df)
    
    return daily_analysis

def analyze_single_day(day_data: pd.DataFrame, date) -> dict:
    """Analyze a single trading day for missing intervals"""
    
    day_data = day_data.drop('date', axis=1)  # Remove the date column we added
    
    # Expected trading hours (9:30 AM - 4:00 PM ET = 390 minutes)
    expected_minutes = 390
    actual_minutes = len(day_data)
    missing_minutes = expected_minutes - actual_minutes
    
    # Find gaps (more than 1 minute between consecutive timestamps)
    time_diffs = day_data.index.to_series().diff()
    gaps = (time_diffs > pd.Timedelta('1 min')).sum()
    
    # Get first and last bar times
    first_bar = day_data.index.min().time() if len(day_data) > 0 else None
    last_bar = day_data.index.max().time() if len(day_data) > 0 else None
    
    return {
        'date': date,
        'actual_bars': actual_minutes,
        'expected_bars': expected_minutes,
        'missing_minutes': missing_minutes,
        'gaps': gaps,
        'first_bar': first_bar,
        'last_bar': last_bar,
        'coverage_pct': (actual_minutes / expected_minutes) * 100 if expected_minutes > 0 else 0
    }

def analyze_dst_issues(df: pd.DataFrame):
    """Check for Daylight Saving Time transition issues"""
    
    # Look for unusual time jumps that might indicate DST
    time_diffs = df.index.to_series().diff()
    
    # Find unusually large gaps (> 5 minutes might indicate DST)
    large_gaps = time_diffs[time_diffs > pd.Timedelta('5 min')]
    
    if len(large_gaps) > 0:
        print(f"   Large time gaps detected: {len(large_gaps)}")
        for timestamp, gap in large_gaps.items():
            print(f"   - {timestamp}: {gap} gap")
    else:
        print("   No significant DST-related gaps detected")
    
    # Check for typical DST dates (March/November Sundays)
    dst_dates = df.index[df.index.month.isin([3, 11]) & (df.index.weekday == 6)]
    if len(dst_dates) > 0:
        print(f"   Data includes potential DST transition dates: {dst_dates.date}")

def detailed_gap_analysis(df: pd.DataFrame):
    """Provide detailed analysis of where gaps occur"""
    
    print(f"\n6. DETAILED GAP ANALYSIS:")
    
    # Find all gaps > 1 minute
    time_diffs = df.index.to_series().diff()
    gaps = time_diffs[time_diffs > pd.Timedelta('1 min')]
    
    if len(gaps) == 0:
        print("   No gaps detected!")
        return
    
    print(f"   Found {len(gaps)} gaps:")
    
    for timestamp, gap in gaps.items():
        # Calculate how many minutes are missing
        missing_mins = int(gap.total_seconds() / 60) - 1
        prev_timestamp = timestamp - gap
        
        print(f"   - Gap from {prev_timestamp.strftime('%Y-%m-%d %H:%M')} "
              f"to {timestamp.strftime('%Y-%m-%d %H:%M')}: "
              f"{gap} ({missing_mins} missing minutes)")

if __name__ == "__main__":
    # Step 1: Fetch data
    raw_data = fetch_spy_data()
    
    # Step 2: Analyze timestamps  
    daily_analysis = analyze_timestamp_quality(raw_data)
    detailed_gap_analysis(raw_data)
    
    # Save analysis results
    analysis_df = pd.DataFrame(daily_analysis)
    analysis_df.to_csv("data/daily_analysis.csv", index=False)
    print(f"\nSaved daily analysis to data/daily_analysis.csv")