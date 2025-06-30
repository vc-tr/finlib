#!/usr/bin/env python3
"""
Complete QC Analysis for SPY 1-minute data
Analyzes missing intervals, duplicates, and DST gaps for the 5-day dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz

def load_and_prepare_data():
    """Load the SPY data and prepare for analysis"""
    print("Loading SPY 1-minute data...")
    df = pd.read_csv("data/raw_spy_1min.csv", index_col=0, parse_dates=True)
    
    print(f"✓ Loaded {len(df)} rows of data")
    print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
    
    # Handle timezone access more robustly
    try:
        tz_info = getattr(df.index, 'tz', 'No timezone info')
        print(f"✓ Timezone: {tz_info}")
    except AttributeError:
        print("✓ Timezone: No timezone info available")
    
    return df

def analyze_trading_sessions(df):
    """Analyze each trading day for completeness"""
    print("\n" + "="*60)
    print("TRADING SESSIONS ANALYSIS")
    print("="*60)
    
    # Convert to Eastern Time for analysis
    df_et = df.copy()
    if df_et.index.tz is not None:
        df_et.index = df_et.index.tz_convert('US/Eastern')
    
    daily_stats = []
    unique_dates = sorted(df_et.index.date)
    
    for trading_date in unique_dates:
        day_data = df_et[df_et.index.date == trading_date]
        
        if len(day_data) == 0:
            continue
            
        # Trading session details
        first_bar = day_data.index.min().time()
        last_bar = day_data.index.max().time()
        actual_bars = len(day_data)
        
        # Standard trading session: 9:30 AM - 4:00 PM ET = 390 minutes
        expected_bars = 390
        missing_bars = expected_bars - actual_bars
        coverage_pct = (actual_bars / expected_bars) * 100
        
        # Check if it's a full or partial trading day
        full_day = last_bar >= time(15, 59)  # Market closes at 4:00 PM
        session_type = "Full" if full_day else "Partial"
        
        daily_stat = {
            'date': trading_date,
            'session_type': session_type,
            'first_bar': first_bar,
            'last_bar': last_bar,
            'actual_bars': actual_bars,
            'expected_bars': expected_bars,
            'missing_bars': missing_bars,
            'coverage_pct': coverage_pct
        }
        daily_stats.append(daily_stat)
        
        print(f"  {trading_date} ({session_type}): {actual_bars:3d}/{expected_bars} bars "
              f"({missing_bars:3d} missing), {first_bar} - {last_bar}, "
              f"coverage: {coverage_pct:.1f}%")
    
    return daily_stats

def find_all_gaps(df):
    """Find all missing 1-minute intervals"""
    print("\n" + "="*60)
    print("MISSING INTERVALS ANALYSIS")
    print("="*60)
    
    # Sort by timestamp to ensure proper order
    df_sorted = df.sort_index()
    
    # Calculate time differences between consecutive bars
    time_diffs = df_sorted.index.to_series().diff()
    
    # Find gaps > 1 minute
    gaps = time_diffs[time_diffs > pd.Timedelta('1 min')]
    
    if len(gaps) == 0:
        print("   ✓ No gaps detected - data is continuous!")
        return []
    
    print(f"   Found {len(gaps)} gaps:")
    
    gap_details = []
    total_missing_minutes = 0
    
    for timestamp, gap_duration in gaps.items():
        missing_mins = int(gap_duration.total_seconds() / 60) - 1
        total_missing_minutes += missing_mins
        
        prev_timestamp = timestamp - gap_duration
        
        gap_info = {
            'gap_start': prev_timestamp,
            'gap_end': timestamp,
            'gap_duration': gap_duration,
            'missing_minutes': missing_mins,
            'trading_day': timestamp.date()
        }
        gap_details.append(gap_info)
        
        print(f"   - {prev_timestamp.strftime('%m/%d %H:%M')} to "
              f"{timestamp.strftime('%m/%d %H:%M')}: {missing_mins} minute(s) missing")
    
    print(f"\n   Total missing minutes across all gaps: {total_missing_minutes}")
    return gap_details

def check_for_duplicates(df):
    """Check for duplicate timestamps"""
    print("\n" + "="*60)
    print("DUPLICATE TIMESTAMPS ANALYSIS")
    print("="*60)
    
    duplicates = df.index.duplicated()
    duplicate_count = duplicates.sum()
    
    print(f"   Duplicate timestamps: {duplicate_count}")
    
    if duplicate_count > 0:
        dup_times = df.index[duplicates]
        print("   Duplicate times found:")
        for dup_time in dup_times:
            print(f"   - {dup_time}")
        return dup_times.tolist()
    else:
        print("   ✓ No duplicate timestamps found")
        return []

def analyze_dst_implications(df):
    """Analyze potential DST transition issues"""
    print("\n" + "="*60)
    print("DST TRANSITION ANALYSIS")
    print("="*60)
    
    # Check for unusually large gaps that might indicate DST issues
    time_diffs = df.index.to_series().diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta('10 min')]
    
    if len(large_gaps) > 0:
        print(f"   Large time gaps detected ({len(large_gaps)}):")
        for timestamp, gap in large_gaps.items():
            print(f"   - {timestamp}: {gap} gap")
    else:
        print("   ✓ No significant DST-related gaps detected")
    
    # Check if data spans potential DST transition periods
    date_range = pd.date_range(df.index.min(), df.index.max())
    
    # Get months more safely
    months_in_data = set(date_range.to_pydatetime()[0].month if len(date_range) > 0 else [])
    for dt in date_range:
        months_in_data.add(dt.month)
    
    dst_months_found = months_in_data.intersection({3, 11})
    
    if dst_months_found:
        print(f"   Data spans months with potential DST transitions: {dst_months_found}")
        if 3 in dst_months_found:
            print("   - March: Spring forward (clocks advance 1 hour)")
        if 11 in dst_months_found:
            print("   - November: Fall back (clocks retreat 1 hour)")
    else:
        print("   Data does not span typical DST transition months")

def generate_summary_statistics(df, daily_stats, gap_details):
    """Generate overall summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall statistics
    total_trading_days = len(daily_stats)
    total_expected_bars = sum(day['expected_bars'] for day in daily_stats)
    total_actual_bars = len(df)
    total_missing_bars = total_expected_bars - total_actual_bars
    overall_coverage = (total_actual_bars / total_expected_bars) * 100
    
    print(f"   Trading days analyzed: {total_trading_days}")
    print(f"   Total expected bars (5 days × 390 min): {total_expected_bars}")
    print(f"   Total actual bars: {total_actual_bars}")
    print(f"   Total missing bars: {total_missing_bars}")
    print(f"   Overall data coverage: {overall_coverage:.1f}%")
    
    # Gap statistics
    if gap_details:
        gaps_per_day = len(gap_details) / total_trading_days
        missing_per_gap = np.mean([gap['missing_minutes'] for gap in gap_details])
        largest_gap = max([gap['missing_minutes'] for gap in gap_details])
        
        print(f"\n   Gap Statistics:")
        print(f"   - Total gaps: {len(gap_details)}")
        print(f"   - Average gaps per day: {gaps_per_day:.1f}")
        print(f"   - Average minutes missing per gap: {missing_per_gap:.1f}")
        print(f"   - Largest single gap: {largest_gap} minutes")
        
        # Gap distribution by day
        gap_by_day = {}
        for gap in gap_details:
            day = gap['trading_day']
            if day not in gap_by_day:
                gap_by_day[day] = 0
            gap_by_day[day] += gap['missing_minutes']
        
        print(f"\n   Missing minutes by trading day:")
        for day, missing_mins in gap_by_day.items():
            print(f"   - {day}: {missing_mins} minutes missing")

def propose_data_handling_strategy():
    """Propose concrete strategies for handling the data quality issues"""
    print("\n" + "="*60)
    print("RECOMMENDED DATA HANDLING STRATEGY")
    print("="*60)
    
    strategies = [
        "1. MISSING DATA BACKFILL RULES:",
        "   a) Small gaps (1-2 minutes): Forward-fill OHLC, set volume=0",
        "   b) Medium gaps (3-10 minutes): Mark as NaN, investigate cause",
        "   c) Large gaps (>10 minutes): Mark as NaN, likely trading halts",
        "",
        "2. IMPLEMENTATION IN PIPELINE:",
        "   a) Create complete minute-by-minute index for trading hours",
        "   b) Reindex data to identify all missing intervals",
        "   c) Apply conditional backfill based on gap size",
        "   d) Add metadata column to flag original vs imputed data",
        "",
        "3. QUALITY CONTROL CHECKS:",
        "   a) Validate trading hours (9:30 AM - 4:00 PM ET)",
        "   b) Check for weekend/holiday data (should not exist)",
        "   c) Verify volume data consistency (no negative values)",
        "   d) Flag any OHLC inconsistencies (high < low, etc.)",
        "",
        "4. MONITORING METRICS:",
        "   a) Data completeness percentage per day",
        "   b) Gap frequency and size distribution",
        "   c) Volume profile consistency",
        "   d) Price continuity checks"
    ]
    
    for strategy in strategies:
        print(f"   {strategy}")

def save_results(daily_stats, gap_details):
    """Save analysis results to CSV files"""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save daily statistics
    daily_df = pd.DataFrame(daily_stats)
    daily_df.to_csv("data/daily_completeness_analysis.csv", index=False)
    print("   ✓ Daily statistics saved to data/daily_completeness_analysis.csv")
    
    # Save gap details if any exist
    if gap_details:
        gaps_df = pd.DataFrame(gap_details)
        gaps_df.to_csv("data/missing_intervals_analysis.csv", index=False)
        print("   ✓ Gap details saved to data/missing_intervals_analysis.csv")
    else:
        print("   ✓ No gaps to save")

def main():
    """Main analysis function"""
    print("SPY 1-MINUTE DATA QUALITY ANALYSIS")
    print("="*60)
    print("Analyzing 5 trading days of SPY 1-minute OHLCV data")
    print("Source: YahooDataFetcher")
    
    # Load and analyze data
    df = load_and_prepare_data()
    daily_stats = analyze_trading_sessions(df)
    gap_details = find_all_gaps(df)
    duplicates = check_for_duplicates(df)
    analyze_dst_implications(df)
    generate_summary_statistics(df, daily_stats, gap_details)
    propose_data_handling_strategy()
    save_results(daily_stats, gap_details)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Results ready for QC report documentation.")

if __name__ == "__main__":
    main() 