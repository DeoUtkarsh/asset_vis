"""
CSV Validation Script
====================
Validates the processed_fleet_data.csv file to ensure data quality and correctness.

This script checks:
1. Data Quality (row count, columns, missing values)
2. Metric Validation (MTBF, failure rates, dates)
3. Business Logic (make/model combinations, bad actor rankings)
4. Sample Data Display
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
CSV_FILE = "processed_fleet_data.csv"

def validate_csv():
    """
    Main validation function
    """
    print("=" * 70)
    print("CSV VALIDATION REPORT")
    print("=" * 70)
    print()
    
    # Check if file exists
    if not Path(CSV_FILE).exists():
        print(f"❌ ERROR: File '{CSV_FILE}' not found!")
        print("   Make sure you've run data_pipeline.py first.")
        return False
    
    # Read CSV
    print(f"📂 Reading: {CSV_FILE}")
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✅ File loaded successfully")
        print()
    except Exception as e:
        print(f"❌ ERROR: Could not read CSV file: {e}")
        return False
    
    # ========================================================================
    # 1. DATA QUALITY CHECKS
    # ========================================================================
    print("=" * 70)
    print("1. DATA QUALITY CHECKS")
    print("=" * 70)
    
    print(f"📊 Total Records: {len(df)}")
    print(f"📊 Total Columns: {len(df.columns)}")
    print()
    
    # Check for required columns
    required_columns = [
        'make_model', 'component_id', 'equipment_type', 'make', 'model',
        'failure_count', 'mtbf_hours', 'failure_rate_per_1000h', 'vessels_affected'
    ]
    
    print("🔍 Checking Required Columns:")
    missing_cols = []
    for col in required_columns:
        if col in df.columns:
            print(f"   ✅ {col}")
        else:
            print(f"   ❌ {col} - MISSING!")
            missing_cols.append(col)
    
    if missing_cols:
        print(f"\n⚠️  WARNING: {len(missing_cols)} required columns are missing!")
    else:
        print(f"\n✅ All required columns present!")
    print()
    
    # Check missing values
    print("🔍 Missing Values Analysis:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    critical_cols = ['make_model', 'component_id', 'failure_count', 'make', 'model']
    for col in critical_cols:
        if col in df.columns:
            missing = missing_data[col]
            percent = missing_percent[col]
            status = "❌" if missing > 0 else "✅"
            print(f"   {status} {col}: {missing} missing ({percent:.1f}%)")
    
    print()
    
    # Data types check
    print("🔍 Data Types:")
    print(f"   Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   Text columns: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"   Date columns: {len(df.select_dtypes(include=['datetime64']).columns)}")
    print()
    
    # ========================================================================
    # 2. METRIC VALIDATION
    # ========================================================================
    print("=" * 70)
    print("2. METRIC VALIDATION")
    print("=" * 70)
    
    # MTBF validation
    if 'mtbf_hours' in df.columns:
        mtbf_data = df['mtbf_hours'].dropna()
        if len(mtbf_data) > 0:
            print(f"📈 MTBF Statistics:")
            print(f"   Valid values: {len(mtbf_data)} / {len(df)} ({len(mtbf_data)/len(df)*100:.1f}%)")
            print(f"   Min: {mtbf_data.min():.1f} hours")
            print(f"   Max: {mtbf_data.max():.1f} hours")
            print(f"   Mean: {mtbf_data.mean():.1f} hours")
            print(f"   Median: {mtbf_data.median():.1f} hours")
            
            # Check for reasonable values (between 1 hour and 1 year)
            reasonable = mtbf_data[(mtbf_data >= 1) & (mtbf_data <= 8760)]
            if len(reasonable) == len(mtbf_data):
                print(f"   ✅ All MTBF values are reasonable (1-8760 hours)")
            else:
                outliers = len(mtbf_data) - len(reasonable)
                print(f"   ⚠️  {outliers} MTBF values outside reasonable range")
        else:
            print(f"   ⚠️  No MTBF values calculated")
        print()
    
    # Failure Rate validation
    if 'failure_rate_per_1000h' in df.columns:
        fr_data = df['failure_rate_per_1000h'].dropna()
        if len(fr_data) > 0:
            print(f"📈 Failure Rate Statistics:")
            print(f"   Valid values: {len(fr_data)} / {len(df)} ({len(fr_data)/len(df)*100:.1f}%)")
            print(f"   Min: {fr_data.min():.3f} per 1000h")
            print(f"   Max: {fr_data.max():.3f} per 1000h")
            print(f"   Mean: {fr_data.mean():.3f} per 1000h")
            
            # Check for positive values
            negative = fr_data[fr_data < 0]
            if len(negative) == 0:
                print(f"   ✅ All failure rates are positive")
            else:
                print(f"   ❌ {len(negative)} negative failure rates found!")
        else:
            print(f"   ⚠️  No failure rate values calculated")
        print()
    
    # Failure Count validation
    if 'failure_count' in df.columns:
        fc_data = df['failure_count']
        print(f"📈 Failure Count Statistics:")
        print(f"   Total failures: {fc_data.sum()}")
        print(f"   Min: {fc_data.min()}")
        print(f"   Max: {fc_data.max()}")
        print(f"   Mean: {fc_data.mean():.1f}")
        print(f"   Components with >= 3 failures: {len(fc_data[fc_data >= 3])}")
        print(f"   Components with 0 failures: {len(fc_data[fc_data == 0])}")
        print()
    
    # Date validation
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        print(f"📅 Date Columns Found: {date_cols}")
        for col in date_cols:
            if col in df.columns:
                date_data = pd.to_datetime(df[col], errors='coerce')
                valid_dates = date_data.dropna()
                if len(valid_dates) > 0:
                    print(f"   {col}:")
                    print(f"      Valid dates: {len(valid_dates)} / {len(df)}")
                    print(f"      Range: {valid_dates.min()} to {valid_dates.max()}")
        print()
    
    # ========================================================================
    # 3. BUSINESS LOGIC VALIDATION
    # ========================================================================
    print("=" * 70)
    print("3. BUSINESS LOGIC VALIDATION")
    print("=" * 70)
    
    # Make/Model combinations
    if 'make_model' in df.columns:
        unique_make_models = df['make_model'].nunique()
        print(f"📋 Unique Make/Model Combinations: {unique_make_models}")
        
        # Show top 10 by failure count
        if 'failure_count' in df.columns:
            top_make_models = df.groupby('make_model')['failure_count'].sum().sort_values(ascending=False).head(10)
            print(f"\n   Top 10 Make/Models by Total Failures:")
            for idx, (make_model, count) in enumerate(top_make_models.items(), 1):
                print(f"      {idx}. {make_model}: {count} failures")
        print()
    
    # Component distribution
    if 'component_id' in df.columns:
        component_counts = df['component_id'].value_counts()
        print(f"📋 Component Distribution:")
        print(f"   Unique components: {df['component_id'].nunique()}")
        print(f"   Top components:")
        for component, count in component_counts.head(10).items():
            print(f"      {component}: {count} records")
        print()
    
    # Equipment type distribution
    if 'equipment_type' in df.columns:
        eq_type_counts = df['equipment_type'].value_counts()
        print(f"📋 Equipment Type Distribution:")
        for eq_type, count in eq_type_counts.items():
            print(f"   {eq_type}: {count} records")
        print()
    
    # Bad Actor validation
    if 'is_bad_actor' in df.columns:
        bad_actors = df[df['is_bad_actor'] == True]
        print(f"📋 Bad Actors:")
        print(f"   Total bad actors identified: {len(bad_actors)}")
        
        if 'bad_actor_rank' in df.columns:
            ranks = bad_actors['bad_actor_rank'].dropna()
            if len(ranks) > 0:
                print(f"   Rank range: {ranks.min()} to {ranks.max()}")
                print(f"   Components ranked in top 10: {len(ranks[ranks <= 10])}")
        print()
    
    # Make/Model Summary validation
    if 'total_failures' in df.columns and 'failure_count' in df.columns:
        # Check if summary matches component totals
        print(f"📋 Make/Model Summary Validation:")
        summary_check = df.groupby('make_model')['failure_count'].sum()
        if 'total_failures' in df.columns:
            summary_totals = df.groupby('make_model')['total_failures'].first()
            matches = (summary_check == summary_totals).all()
            if matches:
                print(f"   ✅ Summary totals match component totals")
            else:
                mismatches = (summary_check != summary_totals).sum()
                print(f"   ⚠️  {mismatches} make/model summaries don't match component totals")
        print()
    
    # ========================================================================
    # 4. SAMPLE DATA DISPLAY
    # ========================================================================
    print("=" * 70)
    print("4. SAMPLE DATA")
    print("=" * 70)
    
    print(f"\n📄 First 5 Rows:")
    print(df.head().to_string())
    print()
    
    print(f"\n📄 Columns in CSV:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    print()
    
    # Show records with highest failure counts
    if 'failure_count' in df.columns:
        print(f"📄 Top 5 Records by Failure Count:")
        top_failures = df.nlargest(5, 'failure_count')[
            ['make_model', 'component_id', 'failure_count', 'mtbf_hours', 'failure_rate_per_1000h']
        ]
        print(top_failures.to_string())
        print()
    
    # Show bad actors sample
    if 'is_bad_actor' in df.columns:
        bad_actor_sample = df[df['is_bad_actor'] == True].head(5)
        if len(bad_actor_sample) > 0:
            print(f"📄 Sample Bad Actors:")
            cols_to_show = ['make_model', 'component_id', 'bad_actor_rank', 'failure_count', 'failure_rate_per_1000h']
            cols_to_show = [c for c in cols_to_show if c in bad_actor_sample.columns]
            print(bad_actor_sample[cols_to_show].to_string())
            print()
    
    # ========================================================================
    # 5. SUMMARY & RECOMMENDATIONS
    # ========================================================================
    print("=" * 70)
    print("5. VALIDATION SUMMARY")
    print("=" * 70)
    
    issues_found = []
    warnings_found = []
    
    # Check for critical issues
    if missing_cols:
        issues_found.append(f"Missing {len(missing_cols)} required columns")
    
    if 'failure_count' in df.columns:
        zero_failures = len(df[df['failure_count'] == 0])
        if zero_failures > 0:
            warnings_found.append(f"{zero_failures} records with 0 failures (may need filtering)")
    
    if 'mtbf_hours' in df.columns:
        mtbf_missing = df['mtbf_hours'].isna().sum()
        if mtbf_missing > len(df) * 0.5:
            warnings_found.append(f"{mtbf_missing} records missing MTBF ({mtbf_missing/len(df)*100:.1f}%)")
    
    if len(df) == 0:
        issues_found.append("CSV file is empty!")
    
    # Print summary
    if len(issues_found) == 0 and len(warnings_found) == 0:
        print("\n✅ VALIDATION PASSED!")
        print("   CSV file looks good and ready for dashboard.")
    else:
        if issues_found:
            print("\n❌ CRITICAL ISSUES FOUND:")
            for issue in issues_found:
                print(f"   - {issue}")
        
        if warnings_found:
            print("\n⚠️  WARNINGS:")
            for warning in warnings_found:
                print(f"   - {warning}")
    
    print()
    print("=" * 70)
    print("Validation Complete!")
    print("=" * 70)
    
    return len(issues_found) == 0


if __name__ == "__main__":
    success = validate_csv()
    exit(0 if success else 1)

