#!/usr/bin/env python3
"""
Fill Missing SPY 1-Minute Data
Fills intraday gaps while ignoring overnight periods between trading sessions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_trading_hours_index(start_date, end_date):
    """Create complete trading hours index (13:30 - 20:00 UTC)"""
    print("Creating complete trading hours index...")
    
    # Convert to pandas timestamps if they aren't already
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Create date range for business days only
    business_days = pd.bdate_range(start=start_dt.date(), end=end_dt.date())
    
    trading_times = []
    for date in business_days:
        # Create 13:30 to 20:00 UTC (9:30 AM - 4:00 PM ET = 390 minutes)
        start_time = pd.Timestamp.combine(date, time(13, 30))
        end_time = pd.Timestamp.combine(date, time(20, 0))
        
        # Add UTC timezone
        start_time = start_time.tz_localize('UTC')
        end_time = end_time.tz_localize('UTC')
        
        # Generate minute-by-minute timestamps
        day_times = pd.date_range(start=start_time, end=end_time, freq='1min')
        trading_times.extend(day_times)
    
    # Create final index
    trading_index = pd.DatetimeIndex(trading_times)
    
    print(f"✓ Created {len(trading_index)} trading minute timestamps")
    return trading_index

def identify_trading_gaps(df, trading_index):
    """Identify gaps during trading hours only"""
    print("\nIdentifying gaps during trading hours...")
    
    # Find missing timestamps in trading hours
    missing_times = trading_index.difference(df.index)
    
    if len(missing_times) == 0:
        print("✓ No gaps found during trading hours")
        return []
    
    # Group consecutive missing times into gaps
    gaps = []
    current_gap_start = None
    current_gap_times = []
    
    for i, missing_time in enumerate(missing_times):
        if current_gap_start is None:
            # Start new gap
            current_gap_start = missing_time
            current_gap_times = [missing_time]
        else:
            # Check if this continues the current gap (within 2 minutes)
            if missing_time - current_gap_times[-1] <= timedelta(minutes=2):
                current_gap_times.append(missing_time)
            else:
                # End current gap and start new one
                gaps.append({
                    'start': current_gap_start,
                    'end': current_gap_times[-1],
                    'missing_times': current_gap_times.copy(),
                    'duration_minutes': len(current_gap_times)
                })
                current_gap_start = missing_time
                current_gap_times = [missing_time]
    
    # Don't forget the last gap
    if current_gap_times:
        gaps.append({
            'start': current_gap_start,
            'end': current_gap_times[-1],
            'missing_times': current_gap_times.copy(),
            'duration_minutes': len(current_gap_times)
        })
    
    print(f"✓ Found {len(gaps)} gaps during trading hours")
    for gap in gaps:
        print(f"   - {gap['start'].strftime('%m/%d %H:%M')} to {gap['end'].strftime('%m/%d %H:%M')}: {gap['duration_minutes']} minute(s)")
    
    return gaps

def fill_small_gaps(df, gaps, max_gap_size=5):
    """Fill gaps smaller than max_gap_size minutes"""
    print(f"\nFilling gaps ≤ {max_gap_size} minutes...")
    
    filled_data = []
    df_filled = df.copy()
    
    for gap in gaps:
        if gap['duration_minutes'] <= max_gap_size:
            # Get the last valid data point before the gap
            last_timestamp = gap['start'] - timedelta(minutes=1)
            
            try:
                # Find the closest previous data point
                previous_data = df_filled[df_filled.index <= last_timestamp].iloc[-1]
                
                # Create filled rows
                for missing_time in gap['missing_times']:
                    filled_row = {
                        'open': previous_data['open'],
                        'high': previous_data['high'], 
                        'low': previous_data['low'],
                        'close': previous_data['close'],
                        'volume': 0,  # Set volume to 0 for filled data
                        'filled': True  # Flag to indicate this data was filled
                    }
                    
                    # Add to dataframe
                    filled_row_df = pd.DataFrame([filled_row], index=[missing_time])
                    df_filled = pd.concat([df_filled, filled_row_df])
                    filled_data.append(missing_time)
                
                print(f"   ✓ Filled {gap['duration_minutes']} minute gap at {gap['start'].strftime('%m/%d %H:%M')}")
                
            except IndexError:
                print(f"   ⚠ Could not fill gap at {gap['start'].strftime('%m/%d %H:%M')} - no previous data")
        else:
            print(f"   ⚠ Skipped large gap ({gap['duration_minutes']} min) at {gap['start'].strftime('%m/%d %H:%M')}")
    
    # Sort by timestamp
    df_filled = df_filled.sort_index()
    
    print(f"✓ Filled {len(filled_data)} data points")
    return df_filled

def add_data_quality_flags(df):
    """Add flags to indicate data quality"""
    if 'filled' not in df.columns:
        df['filled'] = False
    
    # Add gap analysis flags
    df['large_gap_after'] = False
    
    # Check for gaps > 10 minutes after each point
    time_diffs = df.index.to_series().diff().shift(-1)
    large_gap_mask = time_diffs > pd.Timedelta('10 min')
    df.loc[large_gap_mask, 'large_gap_after'] = True
    
    return df

def validate_filled_data(df_original, df_filled):
    """Validate the data filling results"""
    print("\n" + "="*60)
    print("DATA FILLING VALIDATION")
    print("="*60)
    
    original_count = len(df_original)
    filled_count = len(df_filled)
    added_points = filled_count - original_count
    
    print(f"   Original data points: {original_count}")
    print(f"   Filled data points: {filled_count}")
    print(f"   Points added: {added_points}")
    
    if 'filled' in df_filled.columns:
        filled_points = df_filled['filled'].sum()
        print(f"   Flagged as filled: {filled_points}")
        
        # Check for any remaining gaps during trading hours
        trading_hours_data = df_filled.between_time('13:30', '20:00')
        time_diffs = trading_hours_data.index.to_series().diff()
        remaining_gaps = time_diffs[time_diffs > pd.Timedelta('1 min')]
        
        if len(remaining_gaps) > 0:
            print(f"   ⚠ Remaining gaps during trading hours: {len(remaining_gaps)}")
            for timestamp, gap in remaining_gaps.items():
                if gap <= pd.Timedelta('10 min'):  # Only show smaller gaps
                    print(f"     - {timestamp.strftime('%m/%d %H:%M')}: {gap}")
        else:
            print("   ✓ No remaining gaps during trading hours")
    
    return {
        'original_count': original_count,
        'filled_count': filled_count,
        'points_added': added_points
    }

def main():
    """Main data filling function"""
    print("SPY 1-MINUTE DATA FILLING")
    print("="*60)
    print("Filling intraday gaps while preserving overnight periods")
    
    # Load original data
    print("\nLoading original SPY data...")
    df_original = pd.read_csv("data/raw_spy_1min.csv", index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(df_original)} rows")
    print(f"✓ Date range: {df_original.index.min()} to {df_original.index.max()}")
    
    # Create complete trading hours index
    trading_index = create_trading_hours_index(
        df_original.index.min(),
        df_original.index.max()
    )
    
    # Identify gaps during trading hours only
    gaps = identify_trading_gaps(df_original, trading_index)
    
    if not gaps:
        print("\n✓ No gaps to fill during trading hours")
        df_filled = df_original.copy()
    else:
        # Fill small gaps
        df_filled = fill_small_gaps(df_original, gaps, max_gap_size=5)
    
    # Add quality flags
    df_filled = add_data_quality_flags(df_filled)
    
    # Validate results
    validation_stats = validate_filled_data(df_original, df_filled)
    
    # Save filled data
    output_file = "data/spy_1min_filled.csv"
    df_filled.to_csv(output_file)
    print(f"\n✓ Saved filled data to {output_file}")
    
    # Generate summary
    print("\n" + "="*60)
    print("FILLING SUMMARY")
    print("="*60)
    print(f"   Strategy: Fill gaps ≤ 5 minutes during trading hours only")
    print(f"   Original data: {validation_stats['original_count']} points")
    print(f"   Filled data: {validation_stats['filled_count']} points")
    print(f"   Data points added: {validation_stats['points_added']}")
    
    if 'filled' in df_filled.columns:
        filled_percentage = (df_filled['filled'].sum() / len(df_filled)) * 100
        print(f"   Percentage filled: {filled_percentage:.2f}%")
    
    print(f"\n   Overnight gaps: Preserved (ignored as requested)")
    print(f"   Small intraday gaps: Forward-filled with volume=0")
    print(f"   Large intraday gaps: Preserved for manual review")
    
    print(f"\n✓ Data filling complete!")
    return df_filled

if __name__ == "__main__":
    filled_df = main() 