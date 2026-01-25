"""
Final Solution Validation Script
=================================
Validates that all features from the problem statement are implemented
"""

import pandas as pd
from pathlib import Path

def validate_solution():
    """
    Validate the complete solution against problem statement requirements
    """
    print("=" * 70)
    print("FINAL SOLUTION VALIDATION")
    print("=" * 70)
    print()
    
    csv_file = "processed_fleet_data.csv"
    
    if not Path(csv_file).exists():
        print(f"❌ ERROR: {csv_file} not found!")
        return False
    
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded: {csv_file} ({len(df)} records)")
    print()
    
    # Track scores
    total_checks = 0
    passed_checks = 0
    
    def check(condition, name):
        nonlocal total_checks, passed_checks
        total_checks += 1
        if condition:
            passed_checks += 1
            print(f"  ✅ {name}")
            return True
        else:
            print(f"  ❌ {name}")
            return False
    
    # ========================================
    # 1. ASSET FAILURE ANALYTICS
    # ========================================
    print("=" * 70)
    print("1. ASSET FAILURE ANALYTICS")
    print("=" * 70)
    
    check('make_model' in df.columns, "Has Make/Model column")
    check('failure_count' in df.columns, "Has Failure Count")
    check('mtbf_hours' in df.columns, "Has MTBF (hours)")
    check('failure_rate_per_1000h' in df.columns, "Has Failure Rate per 1000h")
    print()
    
    # ========================================
    # 2. ROOT CAUSE ANALYTICS
    # ========================================
    print("=" * 70)
    print("2. ROOT CAUSE ANALYTICS")
    print("=" * 70)
    
    check('root_cause_percentage_confirmed' in df.columns, "Has % Failures with Confirmed Root Cause")
    check('root_cause_human_error' in df.columns, "Has Human Error count")
    check('root_cause_operational_error' in df.columns, "Has Operational Error count")
    check('root_cause_machinery_failure' in df.columns, "Has Machinery Failure count")
    check('root_cause_other' in df.columns, "Has Other root cause count")
    check('root_cause_unknown' in df.columns, "Has Unknown root cause count")
    print()
    
    # ========================================
    # 3. ACTION TAKEN ANALYTICS
    # ========================================
    print("=" * 70)
    print("3. ACTION TAKEN ANALYTICS")
    print("=" * 70)
    
    check('action_inspect_count' in df.columns, "Has Inspect action count")
    check('action_repair_count' in df.columns, "Has Repair action count")
    check('action_replace_count' in df.columns, "Has Replace action count")
    check('action_temporary_count' in df.columns, "Has Temporary action count")
    check('action_recurrence_rate' in df.columns, "Has Recurrence Rate")
    check('action_recurrence_by_type' in df.columns, "Has Recurrence Rate by Action Type")
    check('highest_recurrence_action' in df.columns, "Has Highest Recurrence Action")
    check('action_permanently_eliminates' in df.columns, "Has Actions Permanently Eliminates flag")
    check('mtbf_before_action' in df.columns, "Has MTBF Before Action")
    check('mtbf_after_action' in df.columns, "Has MTBF After Action")
    check('stakeholder_communication_summary' in df.columns, "Has Stakeholder Communication Summary")
    print()
    
    # ========================================
    # 4. BAD ACTOR ANALYTICS
    # ========================================
    print("=" * 70)
    print("4. BAD ACTOR ANALYTICS")
    print("=" * 70)
    
    check('component_id' in df.columns, "Has Component ID")
    check('mtbf_hours' in df.columns, "Has MTBF")
    check('mtbr_hours' in df.columns, "Has MTBR")
    check('mttr_hours' in df.columns, "Has MTTR")
    check('vessels_affected' in df.columns, "Has Vessels Affected")
    check('bad_actor_rank' in df.columns, "Has Bad Actor Rank")
    check('is_bad_actor' in df.columns, "Has Is Bad Actor flag")
    check('mtbf_vs_fleet' in df.columns, "Has MTBF vs Fleet")
    check('trend' in df.columns, "Has Trend (↑ ↓ →)")
    print()
    
    # ========================================
    # 5. FAILURE MODES
    # ========================================
    print("=" * 70)
    print("5. FAILURE MODES")
    print("=" * 70)
    
    check('top_failure_mode' in df.columns, "Has Top Failure Mode")
    check('top_failure_mode_count' in df.columns, "Has Failure Mode Count")
    check('top_failure_mode_rate' in df.columns, "Has Failure Mode Rate")
    check('failure_mode_distribution' in df.columns, "Has Failure Mode Distribution")
    print()
    
    # ========================================
    # 6. SPARE PARTS CONSUMPTION
    # ========================================
    print("=" * 70)
    print("6. SPARE PARTS CONSUMPTION")
    print("=" * 70)
    
    check('estimated_cost_per_failure' in df.columns, "Has Cost per Failure")
    check('total_cost_impact' in df.columns, "Has Total Cost Impact")
    check('parts_per_failure' in df.columns, "Has Parts per Failure")
    check('stock_risk' in df.columns, "Has Stock Risk (Low/Medium/High)")
    check('top_parts' in df.columns, "Has Top Parts list")
    check('top_parts_consumption' in df.columns, "Has Top Parts Consumption")
    
    # Check if parts data is populated
    has_parts_data = df['top_parts'].notna().sum() > 0
    check(has_parts_data, "Parts data is populated")
    
    has_cost_data = df['estimated_cost_per_failure'].notna().sum() > 0
    check(has_cost_data, "Cost data is populated")
    print()
    
    # ========================================
    # 7. SUMMARY COLUMNS
    # ========================================
    print("=" * 70)
    print("7. MAKE/MODEL SUMMARY COLUMNS")
    print("=" * 70)
    
    check('total_failures' in df.columns, "Has Total Failures (summary)")
    check('total_vessels' in df.columns, "Has Total Vessels (summary)")
    check('overall_mtbf_hours' in df.columns, "Has Overall MTBF (summary)")
    check('overall_failure_rate_per_1000h' in df.columns, "Has Overall Failure Rate (summary)")
    check('component_count' in df.columns, "Has Component Count")
    print()
    
    # ========================================
    # FINAL SCORE
    # ========================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    score_pct = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    print(f"\nScore: {passed_checks}/{total_checks} ({score_pct:.1f}%)")
    print()
    
    if score_pct >= 90:
        print("✅ EXCELLENT! Solution meets almost all requirements.")
    elif score_pct >= 75:
        print("✅ GOOD! Solution meets most requirements.")
    elif score_pct >= 50:
        print("⚠️  PARTIAL. Solution needs more work.")
    else:
        print("❌ INCOMPLETE. Major features missing.")
    
    # Show sample data
    print("\n" + "=" * 70)
    print("SAMPLE DATA")
    print("=" * 70)
    
    # Show parts data if available
    print("\nComponents with Parts Data:")
    parts_cols = ['make_model', 'component_id', 'top_parts', 'estimated_cost_per_failure', 'stock_risk']
    parts_cols = [c for c in parts_cols if c in df.columns]
    parts_data = df[df['top_parts'].notna()][parts_cols].head(5) if 'top_parts' in df.columns else pd.DataFrame()
    if len(parts_data) > 0:
        print(parts_data.to_string(index=False))
    else:
        print("  No parts data found")
    
    print()
    return score_pct >= 75


if __name__ == "__main__":
    success = validate_solution()
    exit(0 if success else 1)

