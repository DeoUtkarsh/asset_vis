"""
Asset Failure Analytics - Streamlit Dashboard
=============================================
This dashboard implements the 3-Level Hierarchy from IMPLEMENTATION_PLAN.txt

Level 1: Make/Model Selection (Landing Page)
Level 2: Top 10 Bad Actors for Selected Make/Model
Level 3: Detailed Analytics (Inline Expand)

Sections Implemented:
- Section: FINAL OUTPUT STRUCTURE - 3-LEVEL HIERARCHY
- Section 6: Dashboard Development (Phase 6)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Asset Failure Analytics - Bad Actor Leaderboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .bad-actor-row {
        padding: 0.5rem;
        border-left: 4px solid #ff6b6b;
        margin: 0.25rem 0;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """
    Load processed fleet data from CSV
    """
    csv_file = "processed_fleet_data.csv"
    if not Path(csv_file).exists():
        st.error(f"❌ Error: {csv_file} not found! Please run data_pipeline.py first.")
        st.stop()
    
    df = pd.read_csv(csv_file)
    
    # Convert date columns
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_make_model_summary(df):
    """
    Section 5.9: Get Make/Model Summary Metrics for Level 1
    Returns summary dataframe with one row per Make/Model
    """
    summary_cols = [
        'make_model', 'total_failures', 'total_vessels', 'overall_mtbf_hours',
        'overall_failure_rate_per_1000h', 'component_count', 
        'first_failure_date_summary', 'last_failure_date_summary',
        'equipment_type_summary', 'make_summary', 'model_summary'
    ]
    
    # Get unique make_model records (use summary columns)
    available_cols = [col for col in summary_cols if col in df.columns]
    summary = df[available_cols].drop_duplicates(subset=['make_model']).copy()
    
    # If summary columns don't exist, calculate them
    if 'total_failures' not in summary.columns:
        summary = df.groupby('make_model').agg({
            'failure_count': 'sum',
            'vessels_affected': 'max',
            'mtbf_hours': 'mean',
            'failure_rate_per_1000h': 'mean',
            'component_id': 'nunique',
            'first_failure_date': 'min',
            'last_failure_date': 'max',
            'equipment_type': 'first',
            'make': 'first',
            'model': 'first'
        }).reset_index()
        summary.columns = [
            'make_model', 'total_failures', 'total_vessels', 'overall_mtbf_hours',
            'overall_failure_rate_per_1000h', 'component_count',
            'first_failure_date_summary', 'last_failure_date_summary',
            'equipment_type_summary', 'make_summary', 'model_summary'
        ]
    
    return summary.sort_values('total_failures', ascending=False)


def get_bad_actors_by_components(df, make_model):
    """
    Section 5.8: Get Top 10 Bad Actor Components for selected Make/Model
    """
    filtered = df[(df['make_model'] == make_model) & (df['is_bad_actor'] == True)].copy()
    
    if len(filtered) == 0:
        return pd.DataFrame()
    
    # Sort by failure rate (descending)
    filtered = filtered.sort_values('failure_rate_per_1000h', ascending=False, na_position='last')
    
    # Take top 10
    top_10 = filtered.head(10).copy()
    
    # Prepare display columns (trend and cost impact removed from Level 2)
    display_cols = {
        'component_id': 'Component',
        'failure_rate_per_1000h': 'Failure Rate / 1000h',
        'mtbf_hours': 'MTBF (hours)',
        'mtbf_vs_fleet': 'MTBF vs Fleet (%)',
        'vessels_affected': 'Vessels Affected',
        'failure_count': 'Total Failures',
        'bad_actor_rank': 'Rank'
    }
    
    result = top_10[[col for col in display_cols.keys() if col in top_10.columns]].copy()
    result = result.rename(columns=display_cols)
    
    # Format MTBF vs Fleet
    if 'MTBF vs Fleet (%)' in result.columns:
        result['MTBF vs Fleet (%)'] = result['MTBF vs Fleet (%)'].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
    
    return result


def get_bad_actors_by_failure_modes(df, make_model):
    """
    Get Top 10 Failure Modes for selected Make/Model
    Now uses actual failure mode data from CSV, ranked by failure rate
    """
    filtered = df[df['make_model'] == make_model].copy()
    
    if len(filtered) == 0:
        return pd.DataFrame()
    
    # Extract failure modes from failure_mode_distribution or use top_failure_mode
    failure_modes_list = []
    
    for idx, row in filtered.iterrows():
        # Get top failure mode for this component
        top_fm = row.get('top_failure_mode', None)
        top_fm_count = row.get('top_failure_mode_count', 0)
        top_fm_rate = row.get('top_failure_mode_rate', 0)
        
        if top_fm and top_fm != 'Unknown' and top_fm != 'Other':
            failure_modes_list.append({
                'Failure Mode': top_fm,
                'Failure Count': int(top_fm_count),
                'Failure Rate / 1000h': top_fm_rate if top_fm_rate > 0 else row.get('failure_rate_per_1000h', 0),
                'MTBF (hours)': row.get('mtbf_hours', 0),
                'Affected Components': row.get('component_id', 'Unknown'),
                'Vessels Affected': row.get('vessels_affected', 0)
            })
    
    if len(failure_modes_list) == 0:
        # Fallback: use component-based data
        for idx, row in filtered.iterrows():
            failure_modes_list.append({
                'Failure Mode': f"{row.get('component_id', 'Unknown')} - High Failure Rate",
                'Failure Count': row.get('failure_count', 0),
                'Failure Rate / 1000h': row.get('failure_rate_per_1000h', 0),
                'MTBF (hours)': row.get('mtbf_hours', 0),
                'Affected Components': row.get('component_id', 'Unknown'),
                'Vessels Affected': row.get('vessels_affected', 0)
            })
    
    fm_df = pd.DataFrame(failure_modes_list)
    if len(fm_df) > 0:
        # Group by failure mode and aggregate
        fm_agg = fm_df.groupby('Failure Mode').agg({
            'Failure Count': 'sum',
            'Failure Rate / 1000h': 'mean',  # Use mean for aggregated failure rate
            'MTBF (hours)': 'mean',
            'Vessels Affected': 'sum'
        }).reset_index()
        # Rank by Failure Rate (as per requirement), not count
        fm_agg = fm_agg.sort_values('Failure Rate / 1000h', ascending=False).head(10)
        return fm_agg
    
    return pd.DataFrame()


def get_bad_actors_by_parts(df, make_model):
    """
    Get Top 10 Spare Parts for selected Make/Model
    Uses top_parts and top_parts_consumption columns from CSV
    """
    filtered = df[df['make_model'] == make_model].copy()
    
    if len(filtered) == 0:
        return pd.DataFrame()
    
    parts_data = []
    
    for idx, row in filtered.iterrows():
        component_id = row.get('component_id', 'Unknown')
        top_parts_str = row.get('top_parts', None)
        top_consumption_str = row.get('top_parts_consumption', None)
        part_cost = row.get('estimated_cost_per_failure', None)
        parts_per_failure = row.get('parts_per_failure', None)
        stock_risk = row.get('stock_risk', None)
        
        # Parse top_parts (stored as string representation of list)
        if top_parts_str and pd.notna(top_parts_str) and top_parts_str != 'None':
            try:
                import ast
                top_parts = ast.literal_eval(top_parts_str)
                top_consumption = ast.literal_eval(top_consumption_str) if top_consumption_str and pd.notna(top_consumption_str) else [0] * len(top_parts)
                
                for i, part_name in enumerate(top_parts):
                    consumption = top_consumption[i] if i < len(top_consumption) else 0
                    parts_data.append({
                        'Part Name': part_name,
                        'Component': component_id,
                        'Consumption': int(consumption),
                        'Est. Cost/Failure': part_cost if pd.notna(part_cost) else 0,
                        'Parts/Failure': parts_per_failure if pd.notna(parts_per_failure) else 0,
                        'Stock Risk': stock_risk if stock_risk else 'N/A'
                    })
            except:
                pass
    
    if len(parts_data) == 0:
        return pd.DataFrame()
    
    # Create dataframe and aggregate by part name
    parts_df = pd.DataFrame(parts_data)
    
    # Group by part name and sum consumption
    parts_agg = parts_df.groupby('Part Name').agg({
        'Consumption': 'sum',
        'Est. Cost/Failure': 'mean',
        'Parts/Failure': 'mean',
        'Stock Risk': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
    }).reset_index()
    
    # Sort by consumption and take top 10
    parts_agg = parts_agg.sort_values('Consumption', ascending=False).head(10)
    
    # Format numbers
    parts_agg['Est. Cost/Failure'] = parts_agg['Est. Cost/Failure'].apply(
        lambda x: f"${x:.0f}" if pd.notna(x) and x > 0 else "N/A"
    )
    parts_agg['Parts/Failure'] = parts_agg['Parts/Failure'].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) and x > 0 else "N/A"
    )
    
    return parts_agg


def format_number(value, decimals=2):
    """Format number for display"""
    if pd.isna(value):
        return "N/A"
    if value >= 1000000:
        return f"{value/1000000:.2f}M"
    elif value >= 1000:
        return f"{value/1000:.2f}K"
    else:
        return f"{value:.{decimals}f}"


# ============================================================================
# LEVEL 1: MAKE/MODEL SELECTION (Landing Page)
# ============================================================================

def show_level_1_make_model_selection(df):
    """
    Level 1: Make/Model Selection
    Shows list of all Make/Model combinations with summary metrics
    """
    st.markdown('<div class="main-header">🚢 Asset Failure Analytics</div>', unsafe_allow_html=True)
    st.markdown("### Bad Actor Leaderboard - Make/Model Overview")
    st.markdown("---")
    
    # Get summary data
    summary = get_make_model_summary(df)
    
    if len(summary) == 0:
        st.warning("No data available. Please check your CSV file.")
        return None
    
    # Search and filter
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search Make/Model", placeholder="Type to search...")
    
    with col2:
        equipment_filter = st.selectbox(
            "Filter by Equipment Type",
            options=['All'] + list(summary['equipment_type_summary'].dropna().unique())
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=['Total Failures', 'Total Vessels', 'Failure Rate', 'MTBF']
        )
    
    # Apply filters
    filtered_summary = summary.copy()
    
    if search_term:
        filtered_summary = filtered_summary[
            filtered_summary['make_model'].str.contains(search_term, case=False, na=False)
        ]
    
    if equipment_filter != 'All':
        filtered_summary = filtered_summary[
            filtered_summary['equipment_type_summary'] == equipment_filter
        ]
    
    # Sort
    sort_map = {
        'Total Failures': 'total_failures',
        'Total Vessels': 'total_vessels',
        'Failure Rate': 'overall_failure_rate_per_1000h',
        'MTBF': 'overall_mtbf_hours'
    }
    if sort_by in sort_map:
        filtered_summary = filtered_summary.sort_values(
            sort_map[sort_by],
            ascending=False,
            na_position='last'
        )
    
    # Display summary metrics
    st.markdown(f"**Found {len(filtered_summary)} Make/Model combinations**")
    st.markdown("---")
    
    # Create display dataframe (trend and cost impact removed from Level 1)
    display_cols = ['make_model', 'equipment_type_summary', 'total_failures',
        'total_vessels', 'overall_mtbf_hours', 'overall_failure_rate_per_1000h',
        'component_count']
    display_cols = [c for c in display_cols if c in filtered_summary.columns]
    display_df = filtered_summary[display_cols].copy()
    
    # Rename columns
    column_rename = {
        'make_model': 'Make/Model',
        'equipment_type_summary': 'Equipment Type',
        'total_failures': 'Total Failures',
        'total_vessels': 'Total Vessels',
        'overall_mtbf_hours': 'MTBF (hours)',
        'overall_failure_rate_per_1000h': 'Failure Rate / 1000h',
        'component_count': 'Components'
    }
    display_df = display_df.rename(columns=column_rename)
    
    # Format numbers
    display_df['Total Failures'] = display_df['Total Failures'].apply(lambda x: format_number(x, 0))
    display_df['MTBF (hours)'] = display_df['MTBF (hours)'].apply(lambda x: format_number(x, 1) if pd.notna(x) else "N/A")
    display_df['Failure Rate / 1000h'] = display_df['Failure Rate / 1000h'].apply(lambda x: format_number(x, 2) if pd.notna(x) else "N/A")
    
    # Display table (selection via dropdown below)
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Select make/model via dropdown
    st.markdown("---")
    st.markdown("### Select Make/Model to view Bad Actors:")
    
    make_model_list = [''] + list(filtered_summary['make_model'].unique())
    selected_from_dropdown = st.selectbox(
        "Select Make/Model to view Bad Actors:",
        options=make_model_list,
        index=0,
        key="make_model_selector"
    )
    
    selected_make_model = selected_from_dropdown if selected_from_dropdown else None
    return selected_make_model


# ============================================================================
# LEVEL 2: TOP 10 BAD ACTORS
# ============================================================================

def show_level_2_bad_actors(df, make_model):
    """
    Level 2: Top 10 Bad Actors for Selected Make/Model
    Shows three tabs: Components, Failure Modes, Spare Parts
    """
    st.markdown("---")
    st.markdown(f"## 📊 Bad Actor Analysis: **{make_model}**")
    
    # Get make/model info
    make_model_data = df[df['make_model'] == make_model].iloc[0] if len(df[df['make_model'] == make_model]) > 0 else None
    
    if make_model_data is not None:
        # Display summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Failures", format_number(make_model_data.get('total_failures', 0), 0))
        with col2:
            st.metric("Total Vessels", int(make_model_data.get('total_vessels', 0)))
        with col3:
            mtbf = make_model_data.get('overall_mtbf_hours', None)
            st.metric("Overall MTBF", format_number(mtbf, 1) if pd.notna(mtbf) else "N/A")
        with col4:
            fr = make_model_data.get('overall_failure_rate_per_1000h', None)
            st.metric("Failure Rate", format_number(fr, 2) + " / 1000h" if pd.notna(fr) else "N/A")
        with col5:
            st.metric("Components", int(make_model_data.get('component_count', 0)))
    
    st.markdown("---")
    
    # Three tabs: Components, Failure Modes, Spare Parts
    tab1, tab2, tab3 = st.tabs(["🔧 Components", "⚠️ Failure Modes", "🔩 Spare Parts"])
    
    # TAB 1: Components
    with tab1:
        st.markdown("### Top 10 Bad Actor Components")
        components_df = get_bad_actors_by_components(df, make_model)
        
        if len(components_df) > 0:
            # Display table (selection via dropdown below)
            st.dataframe(
                components_df,
                use_container_width=True,
                height=400
            )
            
            # Store selected component for Level 3
            selected_component = None
            if st.session_state.get('selected_component'):
                selected_component = st.session_state.selected_component
            
            # Get selection from dataframe
            # For now, use a selectbox
            st.markdown("---")
            st.markdown("**Select Component for Detailed Analysis:**")
            component_list = [''] + list(components_df['Component'].unique())
            selected_component = st.selectbox(
                "Component:",
                options=component_list,
                key="component_selector"
            )
            
            if selected_component:
                st.session_state.selected_component = selected_component
                st.session_state.selected_make_model = make_model
                return selected_component
        else:
            st.info("No bad actor components found for this Make/Model.")
    
    # TAB 2: Failure Modes
    with tab2:
        st.markdown("### Top 10 Failure Modes")
        failure_modes_df = get_bad_actors_by_failure_modes(df, make_model)
        
        if len(failure_modes_df) > 0:
            st.dataframe(failure_modes_df, use_container_width=True, height=400)
        else:
            st.info("No failure mode data available for this Make/Model.")
    
    # TAB 3: Spare Parts (Top 10 by consumption for whole Make/Model)
    with tab3:
        st.markdown("### Top 10 Spare Parts by Consumption")
        parts_df = get_bad_actors_by_parts(df, make_model)
        
        if len(parts_df) > 0:
            st.dataframe(parts_df, use_container_width=True, height=400)
            
            # Summary metrics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            mm_data = df[df['make_model'] == make_model]
            total_cost = mm_data['total_cost_impact'].sum()
            avg_cost_per_failure = mm_data['estimated_cost_per_failure'].mean()
            
            with col1:
                st.metric("Total Parts Cost", format_number(total_cost, 0) if total_cost > 0 else "N/A")
            with col2:
                st.metric("Avg Cost per Failure", format_number(avg_cost_per_failure, 2) if pd.notna(avg_cost_per_failure) else "N/A")
            with col3:
                high_risk = len(mm_data[mm_data['stock_risk'] == 'High'])
                st.metric("Components with High Stock Risk", high_risk)
        else:
            st.info("No spare parts data available. Run data_pipeline.py to populate parts data.")
    
    return None


# ============================================================================
# LEVEL 3: DETAILED ANALYTICS (Inline Expand)
# ============================================================================

def show_level_3_detailed_analytics(df, make_model, component_id):
    """
    Level 3: Detailed Analytics for Selected Bad Actor
    Shows inline expanded details with all analytics
    """
    st.markdown("---")
    st.markdown("## 🔍 Detailed Component Analytics")
    
    # Get component data
    component_data = df[
        (df['make_model'] == make_model) & 
        (df['component_id'] == component_id)
    ].iloc[0] if len(df[(df['make_model'] == make_model) & (df['component_id'] == component_id)]) > 0 else None
    
    if component_data is None:
        st.warning("Component data not found.")
        return
    
    # A. Component Analytics
    with st.expander("📈 Component Analytics", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            mtbf = component_data.get('mtbf_hours', None)
            mtbf_display = format_number(mtbf, 1) if pd.notna(mtbf) else "N/A"
            st.metric("MTBF", mtbf_display + " hours")
            # Visual signal
            if pd.notna(mtbf) and mtbf < 100:
                st.markdown("🔴 **MTBF below peer average**")
        
        with col2:
            mtbr = component_data.get('mtbr_hours', None)
            mtbr_display = format_number(mtbr, 1) if pd.notna(mtbr) else "N/A"
            st.metric("MTBR", mtbr_display + " hours")
            # Visual signal
            if pd.notna(mtbr) and mtbr < 50:
                st.markdown("⚠️ **Low MTBR = repeat repairs**")
        
        with col3:
            mttr = component_data.get('mttr_hours', None)
            mttr_display = format_number(mttr, 1) if pd.notna(mttr) else "N/A"
            st.metric("MTTR", mttr_display + " hours")
        
        with col4:
            fr = component_data.get('failure_rate_per_1000h', None)
            fr_display = format_number(fr, 2) if pd.notna(fr) else "N/A"
            st.metric("Failure Rate", fr_display + " / 1000h")
        
        with col5:
            vessels = component_data.get('vessels_affected', 0)
            st.metric("Vessels Affected", int(vessels))
        
        # Chart: Failure trend (if we had time series data)
        st.markdown("---")
        st.markdown("**Failure Timeline:**")
        date_range = f"{component_data.get('first_failure_date', 'N/A')} to {component_data.get('last_failure_date', 'N/A')}"
        st.info(f"📅 {date_range}")
    
    # B. Failure Mode & Root Cause
    with st.expander("⚠️ Failure Mode & Root Cause Analysis", expanded=False):
        st.markdown("### Root Cause Snapshot")
        
        # Root cause breakdown
        rc_human = component_data.get('root_cause_human_error', 0)
        rc_operational = component_data.get('root_cause_operational_error', 0)
        rc_machinery = component_data.get('root_cause_machinery_failure', 0)
        rc_other = component_data.get('root_cause_other', 0)
        rc_unknown = component_data.get('root_cause_unknown', 0)
        total_rc = rc_human + rc_operational + rc_machinery + rc_other + rc_unknown
        
        rc_percentage = component_data.get('root_cause_percentage_confirmed', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("% with Confirmed Root Cause", f"{rc_percentage:.1f}%")
            
            if rc_percentage < 50:
                st.warning(f"⚠️ **{100-rc_percentage:.1f}% failures have unknown root cause — reliability gap**")
        
        with col2:
            # Root cause pie chart
            if total_rc > 0:
                fig_rc = px.pie(
                    values=[rc_human, rc_operational, rc_machinery, rc_other, rc_unknown],
                    names=['Human Error', 'Operational Error', 'Machinery Failure', 'Other', 'Unknown'],
                    title="Root Cause Distribution"
                )
                st.plotly_chart(fig_rc, use_container_width=True)
        
        # Root cause breakdown table
        st.markdown("### Root Cause Breakdown")
        rc_df = pd.DataFrame({
            'Root Cause Category': ['Human Error', 'Operational Error', 'Machinery Failure', 'Other', 'Unknown'],
            'Count': [rc_human, rc_operational, rc_machinery, rc_other, rc_unknown],
            'Percentage': [
                (rc_human/total_rc*100) if total_rc > 0 else 0,
                (rc_operational/total_rc*100) if total_rc > 0 else 0,
                (rc_machinery/total_rc*100) if total_rc > 0 else 0,
                (rc_other/total_rc*100) if total_rc > 0 else 0,
                (rc_unknown/total_rc*100) if total_rc > 0 else 0
            ]
        })
        st.dataframe(rc_df, use_container_width=True)
    
    # C. Action Taken
    with st.expander("🔧 Action Taken", expanded=False):
        st.markdown("### Action Taken Analytics")
        
        # Action type breakdown
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            inspect_count = component_data.get('action_inspect_count', 0)
            st.metric("Inspect", int(inspect_count))
        with col2:
            repair_count = component_data.get('action_repair_count', 0)
            st.metric("Repair", int(repair_count))
        with col3:
            replace_count = component_data.get('action_replace_count', 0)
            st.metric("Replace", int(replace_count))
        with col4:
            temp_count = component_data.get('action_temporary_count', 0)
            st.metric("Temporary", int(temp_count))
        
        # Recurrence rate
        recurrence_rate = component_data.get('action_recurrence_rate', None)
        highest_recurrence_action = component_data.get('highest_recurrence_action', None)
        highest_recurrence_rate = component_data.get('highest_recurrence_rate', None)
        
        if pd.notna(recurrence_rate):
            st.markdown(f"**Overall Action Recurrence Rate:** {recurrence_rate:.1f}%")
            if recurrence_rate > 50:
                st.warning("⚠️ **High recurrence rate - actions not effective**")
        
        # Actions with highest recurrence rate
        if highest_recurrence_action and pd.notna(highest_recurrence_rate):
            st.markdown(f"**Action with Highest Recurrence:** {highest_recurrence_action} ({highest_recurrence_rate:.1f}%)")
            if highest_recurrence_rate > 70:
                st.error(f"❌ **{highest_recurrence_action} has very high recurrence - consider alternative approach**")
        
        # Actions that permanently eliminate failures
        permanently_eliminates = component_data.get('action_permanently_eliminates', False)
        if permanently_eliminates:
            st.success("✅ **Actions have permanently eliminated failures (MTBF improved >2x)**")
        
        # MTBF before vs after
        mtbf_before = component_data.get('mtbf_before_action', None)
        mtbf_after = component_data.get('mtbf_after_action', None)
        
        if pd.notna(mtbf_before) and pd.notna(mtbf_after):
            st.markdown("### MTBF Before vs After Actions")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MTBF Before", f"{mtbf_before:.1f} hours")
            with col2:
                st.metric("MTBF After", f"{mtbf_after:.1f} hours")
            with col3:
                improvement = ((mtbf_after - mtbf_before) / mtbf_before * 100) if mtbf_before > 0 else 0
                st.metric("Change", f"{improvement:+.1f}%")
            
            # Auto flags
            if temp_count > 3:
                st.error("❌ **Repeated temporary mitigations detected**")
            if repair_count > 0 and improvement < 10:
                st.warning("⚠️ **Repair with no significant MTBF improvement**")
            
            # Check if actions permanently eliminate failures
            permanently_eliminates = component_data.get('action_permanently_eliminates', False)
            if permanently_eliminates:
                st.success("✅ **Actions have permanently eliminated failures (MTBF improved >2x)**")
        
        # Stakeholder Communication Summary
        stakeholder_comm = component_data.get('stakeholder_communication_summary', None)
        if stakeholder_comm:
            st.markdown("---")
            st.markdown("### Stakeholder Communication Summary")
            st.info(f"**Communications with:** {stakeholder_comm}")
    
    # D. Spare Parts Impact (separate expander)
    with st.expander("📦 Spare Parts Impact", expanded=False):
        cost_per_failure = component_data.get('estimated_cost_per_failure', None)
        total_cost = component_data.get('total_cost_impact', None)
        stock_risk = component_data.get('stock_risk', None)
        parts_per_failure = component_data.get('parts_per_failure', None)
        parts_detail_json = component_data.get('parts_detail_json', None)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cost per Failure", f"${format_number(cost_per_failure, 2)}" if pd.notna(cost_per_failure) and cost_per_failure > 0 else "N/A")
        with col2:
            st.metric("Total Cost Impact", f"${format_number(total_cost, 0)}" if pd.notna(total_cost) and total_cost > 0 else "N/A")
        with col3:
            st.metric("Parts per Failure", format_number(parts_per_failure, 2) if pd.notna(parts_per_failure) and parts_per_failure > 0 else "N/A")
        with col4:
            if stock_risk:
                risk_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
                st.metric("Stock Risk", f"{risk_color.get(stock_risk, '⚪')} {stock_risk}")
            else:
                st.metric("Stock Risk", "N/A")
        
        # Parts breakdown table (Part Name, Count, Unit Cost, Total Cost)
        if parts_detail_json and pd.notna(parts_detail_json) and str(parts_detail_json) != '[]':
            try:
                import json
                parts_list = json.loads(parts_detail_json)
                if parts_list:
                    parts_df = pd.DataFrame(parts_list)
                    parts_df = parts_df.rename(columns={
                        'part_name': 'Part Name',
                        'count': 'Count',
                        'unit_cost': 'Unit Cost ($)',
                        'total_cost': 'Total Cost ($)'
                    })
                    st.markdown("**Parts Breakdown:**")
                    st.dataframe(parts_df, use_container_width=True, hide_index=True)
            except Exception:
                pass
        else:
            st.info("No spare parts data available for this component.")
    
    # D. Decision & Action Panel (Sticky)
    st.markdown("---")
    st.markdown("### 🎯 Decision & Action Panel")
    
    with st.container():
        st.markdown("**Convert insight → execution immediately**")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("🔄 Change PM Strategy", use_container_width=True):
                st.session_state.action_type = "Change PM Strategy"
        
        with col2:
            if st.button("🔧 Engineering Fix", use_container_width=True):
                st.session_state.action_type = "Engineering Fix"
        
        with col3:
            if st.button("📞 Supplier Escalation", use_container_width=True):
                st.session_state.action_type = "Supplier Escalation"
        
        with col4:
            if st.button("📦 Increase Spares", use_container_width=True):
                st.session_state.action_type = "Increase Spares"
        
        with col5:
            if st.button("👁️ Monitor Only", use_container_width=True):
                st.session_state.action_type = "Monitor Only"
        
        # Action tracking form
        if st.session_state.get('action_type'):
            st.markdown("---")
            st.markdown(f"**Selected Action: {st.session_state.action_type}**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                owner = st.text_input("Owner", key="action_owner")
                due_date = st.date_input("Due Date", key="action_due_date")
            
            with col2:
                expected_outcome = st.selectbox(
                    "Expected Outcome",
                    options=["MTBF Improvement", "Failure Rate Reduction", "Cost Reduction", "Other"],
                    key="action_outcome"
                )
            
            with col3:
                status = st.selectbox(
                    "Status",
                    options=["Open", "Improving", "Closed"],
                    key="action_status"
                )
            
            if st.button("💾 Save Action", type="primary"):
                st.success(f"Action '{st.session_state.action_type}' saved for {make_model} - {component_id}")
                st.info(f"Owner: {owner} | Due: {due_date} | Status: {status}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """
    Main Streamlit app
    Implements 3-Level Hierarchy navigation
    """
    # Load data
    df = load_data()
    
    # Initialize session state
    if 'selected_make_model' not in st.session_state:
        st.session_state.selected_make_model = None
    if 'selected_component' not in st.session_state:
        st.session_state.selected_component = None
    
    # Navigation logic
    # Check if we have a selected component (Level 3)
    if st.session_state.get('selected_component') and st.session_state.get('selected_make_model'):
        make_model = st.session_state.selected_make_model
        component = st.session_state.selected_component
        
        # Show breadcrumb
        st.markdown(f"**Navigation:** [Make/Model List](#) > {make_model} > {component}")
        
        # Back button
        if st.button("← Back to Bad Actors"):
            st.session_state.selected_component = None
            st.rerun()
        
        # Show Level 3
        show_level_3_detailed_analytics(df, make_model, component)
    
    # Check if we have a selected make/model (Level 2)
    elif st.session_state.get('selected_make_model'):
        make_model = st.session_state.selected_make_model
        
        # Show breadcrumb
        st.markdown(f"**Navigation:** [Make/Model List](#) > {make_model}")
        
        # Back button
        if st.button("← Back to Make/Model List"):
            st.session_state.selected_make_model = None
            st.session_state.selected_component = None
            st.rerun()
        
        # Show Level 2
        selected_component = show_level_2_bad_actors(df, make_model)
        
        # If component selected, show Level 3
        if selected_component:
            st.session_state.selected_component = selected_component
            st.rerun()
    
    # Default: Show Level 1
    else:
        selected_make_model = show_level_1_make_model_selection(df)
        
        if selected_make_model:
            st.session_state.selected_make_model = selected_make_model
            st.rerun()


if __name__ == "__main__":
    main()

