"""
Asset Failure Analytics - Data Pipeline
========================================
This script implements the data processing pipeline as defined in IMPLEMENTATION_PLAN.txt

Sections Implemented:
- Section 3: Data Cleaning Plan
- Section 4: Data Merging Strategy  
- Section 5: Calculation Methodology
- Section 5.8: Top 10 Bad Actors (per Make/Model)
- Section 5.9: Make/Model Summary Metrics

Input: Excel files in ./Asset_Module/ folder
Output: processed_fleet_data.csv (unified dataset with calculated metrics)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_FOLDER = Path("Asset_Module")
OUTPUT_FILE = "processed_fleet_data.csv"

# Column name standardization mapping (Section 3.1)
COLUMN_MAPPING = {
    # Vessel identifiers
    "Vessel Name": "vessel_name",
    "Vessel_Name": "vessel_name",
    "Vessel": "vessel_name",
    "IMO_Number": "imo_number",
    "IMO No": "imo_number",
    "IMO No.": "imo_number",
    # Dates
    "Date of occurence": "failure_date",  # Context-dependent, will handle in processing
    "Date and Time of occurrence": "failure_date",
    "Date /Time in UTC": "failure_date",
    "Date/ Time of notification sent": "notification_date",
    # Components
    "Equipment": "equipment",
    "Sub Component": "sub_component",
    "Description of alert": "alert_description",
    "Details of Incident": "incident_description",
    "Description Of Event": "incident_description",
    # Actions
    "Action Taken": "action_taken",
    "Remarks from TSI/Vessel": "remarks",
    "Root Cause/Action/Remarks": "remarks",
    # Other common fields
    "Fleet": "fleet",
    "Category of alert": "alert_category",
    "Incident Category": "incident_category",
    "Severity Of Incident": "severity",
    "Possibility Of Recurrence": "recurrence_possibility",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def standardize_column_names(df, mapping=COLUMN_MAPPING):
    """
    Section 3.1: Column Name Standardization
    Standardize column names across all dataframes
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # Apply mapping
    rename_dict = {}
    for old_name, new_name in mapping.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name
    
    df = df.rename(columns=rename_dict)
    return df


def parse_date_column(df, date_col, date_formats=None):
    """
    Section 3.2: Date Format Standardization
    Parse dates with multiple format handling
    """
    if date_col not in df.columns:
        return df
    
    if date_formats is None:
        date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%m/%d/%Y_%H:%M',
            '%m/%d/%Y',
            '%d/%m/%Y',
        ]
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format='mixed')
    return df


def standardize_vessel_name(name):
    """
    Section 9.1: Vessel Name Standardization
    Basic standardization - can be enhanced with fuzzy matching
    """
    if pd.isna(name):
        return None
    return str(name).strip().upper()


def extract_component_from_description(description):
    """
    Section 9.2: Component Extraction from Descriptions
    Extract component identifiers from failure descriptions
    """
    if pd.isna(description):
        return None
    
    desc = str(description).upper()
    
    # Look for common patterns
    patterns = {
        'AE1': 'AE1', 'AE2': 'AE2', 'AE3': 'AE3', 'AE4': 'AE4',
        'ME1': 'ME1', 'ME2': 'ME2', 'ME UNIT': 'ME',
        'GE1': 'GE1', 'GE2': 'GE2', 'GE #1': 'GE1', 'GE #2': 'GE2',
        'TC': 'Turbocharger', 'TURBOCHARGER': 'Turbocharger',
        'FUEL PUMP': 'Fuel Pump', 'JCFW': 'JCFW Pump',
        'LO PURIFIER': 'LO Purifier', 'AIR COMPRESSOR': 'Air Compressor',
    }
    
    for pattern, component in patterns.items():
        if pattern in desc:
            return component
    
    return None


def parse_action_type(action_text):
    """
    Section 9.3: Action Type Parsing
    Extract action type from "Action Taken" column
    """
    if pd.isna(action_text):
        return None
    
    action = str(action_text).upper()
    
    if any(word in action for word in ['REPAIR', 'FIX', 'RECTIFY', 'RESTORE']):
        return 'Repair'
    elif any(word in action for word in ['REPLACE', 'RENEW', 'CHANGE', 'SUBSTITUTE']):
        return 'Replace'
    elif any(word in action for word in ['INSPECT', 'CHECK', 'EXAMINE', 'VERIFY']):
        return 'Inspect'
    elif any(word in action for word in ['TEMPORARY', 'MITIGATION', 'INTERIM', 'STOPGAP']):
        return 'Temporary'
    else:
        return 'Other'


def categorize_root_cause(remarks_text):
    """
    Section 9.4: Root Cause Extraction
    Categorize root cause from remarks
    """
    if pd.isna(remarks_text):
        return 'Unknown'
    
    remarks = str(remarks_text).upper()
    
    if any(word in remarks for word in ['HUMAN', 'OPERATOR', 'CREW', 'MISTAKE', 'ERROR', 'MISOPERATION']):
        return 'Human Error'
    elif any(word in remarks for word in ['OPERATIONAL', 'PROCEDURE', 'PROCESS', 'PROCEDURAL']):
        return 'Operational Error'
    elif any(word in remarks for word in ['FAILURE', 'BREAKDOWN', 'MALFUNCTION', 'DEFECT', 'SENSOR', 'COMPONENT']):
        return 'Machinery Failure'
    else:
        return 'Other'


def extract_stakeholder_communication(remarks_text):
    """
    Extract stakeholder communication summary from remarks
    Looks for mentions of makers, suppliers, stakeholders
    """
    if pd.isna(remarks_text):
        return None
    
    remarks = str(remarks_text)
    remarks_upper = remarks.upper()
    
    # Keywords indicating stakeholder communication
    stakeholder_keywords = {
        'maker': ['MAKER', 'MANUFACTURER', 'BUILDER', 'YANMAR', 'DAIHATSU', 'HYUNDAI', 'STX', 'WARTSILA', 'MITSUI'],
        'supplier': ['SUPPLIER', 'VENDOR', 'SUPPLY'],
        'stakeholder': ['STAKEHOLDER', 'MANAGEMENT', 'OWNER', 'OFFICE'],
        'communication': ['EMAIL', 'CALL', 'MEETING', 'DISCUSS', 'CONTACT', 'INFORM', 'NOTIFY']
    }
    
    found_mentions = []
    for category, keywords in stakeholder_keywords.items():
        if any(keyword in remarks_upper for keyword in keywords):
            found_mentions.append(category)
    
    if found_mentions:
        return ', '.join(found_mentions)
    
    return None


def extract_failure_mode(description):
    """
    Extract failure mode from incident description
    Common failure modes: High Temperature, Low Pressure, Leak, Vibration, Overload, etc.
    """
    if pd.isna(description):
        return 'Unknown'
    
    desc = str(description).upper()
    
    # Common failure mode patterns
    failure_modes = {
        'HIGH TEMPERATURE': ['HIGH TEMP', 'TEMPERATURE HIGH', 'OVERHEAT', 'HOT', 'EXCESSIVE TEMP'],
        'LOW TEMPERATURE': ['LOW TEMP', 'TEMPERATURE LOW', 'COLD', 'UNDER TEMP'],
        'HIGH PRESSURE': ['HIGH PRESSURE', 'PRESSURE HIGH', 'OVER PRESSURE'],
        'LOW PRESSURE': ['LOW PRESSURE', 'PRESSURE LOW', 'UNDER PRESSURE'],
        'LEAK': ['LEAK', 'LEAKAGE', 'SEEPAGE', 'DRIP'],
        'VIBRATION': ['VIBRATION', 'VIBRATE', 'SHAKE', 'OSCILLATION'],
        'OVERLOAD': ['OVERLOAD', 'OVER LOAD', 'EXCESSIVE LOAD'],
        'UNDERLOAD': ['UNDERLOAD', 'UNDER LOAD', 'INSUFFICIENT LOAD'],
        'BLOCKAGE': ['BLOCK', 'BLOCKAGE', 'CLOG', 'OBSTRUCTION'],
        'CORROSION': ['CORRODE', 'CORROSION', 'RUST', 'OXIDATION'],
        'WEAR': ['WEAR', 'WEARING', 'ABRASION', 'DETERIORATION'],
        'BREAKDOWN': ['BREAKDOWN', 'BREAK DOWN', 'FAILURE', 'MALFUNCTION'],
        'SENSOR FAULT': ['SENSOR', 'TRANSMITTER', 'PROBE FAULT', 'SENSOR ERROR'],
        'ALARM': ['ALARM', 'ALERT', 'WARNING'],
        'NO START': ['NO START', 'WON\'T START', 'FAIL TO START', 'START FAILURE'],
        'STOPPED': ['STOPPED', 'STOP', 'SHUT DOWN', 'TRIP'],
        'NOISY': ['NOISE', 'NOISY', 'SOUND', 'AUDIBLE'],
        'SMOKE': ['SMOKE', 'SMOKING', 'FUMES'],
    }
    
    for mode, keywords in failure_modes.items():
        if any(keyword in desc for keyword in keywords):
            return mode
    
    # If no specific mode found, try to extract from common phrases
    if 'TEMPERATURE' in desc:
        if 'HIGH' in desc or 'ELEVATED' in desc:
            return 'High Temperature'
        elif 'LOW' in desc:
            return 'Low Temperature'
    elif 'PRESSURE' in desc:
        if 'HIGH' in desc:
            return 'High Pressure'
        elif 'LOW' in desc:
            return 'Low Pressure'
    
    return 'Other'


# ============================================================================
# STEP 1: CREATE MASTER EQUIPMENT TABLE
# Section 4.1: Create Master Equipment Table
# ============================================================================

def create_master_equipment():
    """
    Section 4.1: Create Master Equipment Table
    Merge Make & Model files and normalize to long format
    """
    print("=" * 60)
    print("STEP 1: Creating Master Equipment Table")
    print("=" * 60)
    
    equipment_list = []
    
    # 1. Process Aux Engine file (normalize wide to long)
    aux_file = DATA_FOLDER / "Make & Model - Aux Engine -New.xlsx"
    if aux_file.exists():
        print(f"Processing: {aux_file.name}")
        try:
            df_aux = pd.read_excel(aux_file, sheet_name="Make _ Model - Aux Engine -New")
            df_aux = standardize_column_names(df_aux)
            
            # Normalize AE1, AE2, AE3, etc. to long format
            for engine_num in [1, 2, 3, 4, 5, 6]:
                make_col = f"AE{engine_num}_Make"
                model_col = f"AE{engine_num}_Model"
                make_type_col = f"AE{engine_num}_Make type" if engine_num <= 3 else ("Make type" if engine_num == 5 else f"Make type (1)")
                engine_type_col = f"AE{engine_num}_Engine type" if engine_num <= 3 else ("Engine type" if engine_num == 5 else f"Engine type (1)")
                
                # Check if columns exist
                if make_col in df_aux.columns and model_col in df_aux.columns:
                    df_engine = df_aux[['vessel_name', 'imo_number', make_col, model_col]].copy()
                    df_engine = df_engine[df_engine[make_col].notna()]  # Only rows with data
                    
                    df_engine['equipment_type'] = 'Aux Engine'
                    df_engine['component_id'] = f'AE{engine_num}'
                    df_engine['make'] = df_engine[make_col]
                    df_engine['model'] = df_engine[model_col]
                    
                    if make_type_col in df_aux.columns:
                        df_engine['make_type'] = df_aux[make_type_col]
                    if engine_type_col in df_aux.columns:
                        df_engine['engine_type'] = df_aux[engine_type_col]
                    
                    # Keep only standardized columns
                    df_engine = df_engine[['vessel_name', 'imo_number', 'equipment_type', 
                                          'component_id', 'make', 'model', 'make_type', 'engine_type']].copy()
                    equipment_list.append(df_engine)
        except Exception as e:
            print(f"  Warning: Error processing Aux Engine file: {e}")
    
    # 2. Process Main Engine file
    main_file = DATA_FOLDER / "Make & Model - Main Engine - New.xlsx"
    if main_file.exists():
        print(f"Processing: {main_file.name}")
        try:
            df_main = pd.read_excel(main_file, sheet_name="Make _ Model - Main Engine - Ne")
            df_main = standardize_column_names(df_main)
            
            df_main['equipment_type'] = 'Main Engine'
            df_main['component_id'] = 'ME1'
            df_main['make'] = df_main.get('ME1_Make', None)
            df_main['model'] = df_main.get('ME1_Model', None)
            df_main['make_type'] = df_main.get('ME1_Make_Type', None)
            df_main['engine_type'] = df_main.get('ME1_Engine_Type', None)
            
            df_main = df_main[['vessel_name', 'imo_number', 'equipment_type', 
                              'component_id', 'make', 'model', 'make_type', 'engine_type']].copy()
            df_main = df_main[df_main['make'].notna()]  # Only rows with data
            equipment_list.append(df_main)
        except Exception as e:
            print(f"  Warning: Error processing Main Engine file: {e}")
    
    # 3. Process BWTS file
    bwts_file = DATA_FOLDER / "Make & Model - BWTS.xlsx"
    if bwts_file.exists():
        print(f"Processing: {bwts_file.name}")
        try:
            df_bwts = pd.read_excel(bwts_file, sheet_name="Make _ Model - BWTS")
            df_bwts = standardize_column_names(df_bwts)
            
            df_bwts['equipment_type'] = 'BWTS'
            df_bwts['component_id'] = 'BWTS1'
            df_bwts['make'] = df_bwts.get('BWTS1_Make', None)
            df_bwts['model'] = df_bwts.get('BWTS1_Model', None)
            df_bwts['make_type'] = None
            df_bwts['engine_type'] = None
            
            df_bwts = df_bwts[['vessel_name', 'imo_number', 'equipment_type', 
                              'component_id', 'make', 'model', 'make_type', 'engine_type']].copy()
            df_bwts = df_bwts[df_bwts['make'].notna()]  # Only rows with data
            equipment_list.append(df_bwts)
        except Exception as e:
            print(f"  Warning: Error processing BWTS file: {e}")
    
    # Combine all equipment
    if equipment_list:
        master_equipment = pd.concat(equipment_list, ignore_index=True)
        
        # Standardize vessel names
        master_equipment['vessel_name'] = master_equipment['vessel_name'].apply(standardize_vessel_name)
        
        # Create make_model key for joining
        master_equipment['make_model'] = master_equipment['make'].astype(str) + ' ' + master_equipment['model'].astype(str)
        
        print(f"  Created master_equipment with {len(master_equipment)} records")
        print(f"  Unique vessels: {master_equipment['vessel_name'].nunique()}")
        print(f"  Unique make/models: {master_equipment['make_model'].nunique()}")
        
        return master_equipment
    else:
        print("  Warning: No equipment data found!")
        return pd.DataFrame()


# ============================================================================
# STEP 2: CREATE MASTER INCIDENTS TABLE
# Section 4.2: Create Master Incidents Table
# ============================================================================

def create_master_incidents():
    """
    Section 4.2: Create Master Incidents Table
    Merge all incident sources
    """
    print("\n" + "=" * 60)
    print("STEP 2: Creating Master Incidents Table")
    print("=" * 60)
    
    incidents_list = []
    
    # 1. Process Incident Database (all sheets)
    incident_db_file = DATA_FOLDER / "Incident Database.xlsx"
    if incident_db_file.exists():
        print(f"Processing: {incident_db_file.name}")
        try:
            xls = pd.ExcelFile(incident_db_file)
            for sheet_name in xls.sheet_names:
                if sheet_name.lower() in ['template', 'guidelines', 'comments']:
                    continue  # Skip template/guideline sheets
                
                try:
                    df = pd.read_excel(incident_db_file, sheet_name=sheet_name)
                    df = standardize_column_names(df)
                    
                    # Extract key columns
                    incident_cols = {}
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if 'vessel' in col_lower and 'name' in col_lower:
                            incident_cols['vessel_name'] = col
                        elif 'date' in col_lower and 'time' in col_lower and 'occurrence' in col_lower:
                            incident_cols['failure_date'] = col
                        elif 'details' in col_lower or 'description' in col_lower or 'event' in col_lower:
                            incident_cols['incident_description'] = col
                        elif 'equipment' in col_lower or 'component' in col_lower:
                            incident_cols['component'] = col
                        elif 'category' in col_lower or 'type' in col_lower:
                            incident_cols['incident_category'] = col
                    
                    # Create standardized dataframe
                    df_incident = pd.DataFrame()
                    if 'vessel_name' in incident_cols:
                        df_incident['vessel_name'] = df[incident_cols['vessel_name']]
                    if 'failure_date' in incident_cols:
                        df_incident['failure_date'] = df[incident_cols['failure_date']]
                    if 'incident_description' in incident_cols:
                        df_incident['incident_description'] = df[incident_cols['incident_description']]
                    if 'component' in incident_cols:
                        df_incident['component'] = df[incident_cols['component']]
                    if 'incident_category' in incident_cols:
                        df_incident['incident_category'] = df[incident_cols['incident_category']]
                    
                    df_incident['source_file'] = incident_db_file.name
                    df_incident['source_sheet'] = sheet_name
                    df_incident['doc_office'] = sheet_name  # DOC office from sheet name
                    
                    # Parse dates
                    if 'failure_date' in df_incident.columns:
                        df_incident = parse_date_column(df_incident, 'failure_date')
                    
                    # Standardize vessel names
                    if 'vessel_name' in df_incident.columns:
                        df_incident['vessel_name'] = df_incident['vessel_name'].apply(standardize_vessel_name)
                    
                    # Extract component from description if not available
                    if 'component' not in df_incident.columns or df_incident['component'].isna().all():
                        if 'incident_description' in df_incident.columns:
                            df_incident['component'] = df_incident['incident_description'].apply(extract_component_from_description)
                    
                    # Extract failure mode from description
                    if 'incident_description' in df_incident.columns:
                        df_incident['failure_mode'] = df_incident['incident_description'].apply(extract_failure_mode)
                    
                    # Remove rows with no vessel or date
                    df_incident = df_incident[df_incident['vessel_name'].notna()]
                    df_incident = df_incident[df_incident['failure_date'].notna()]
                    
                    if len(df_incident) > 0:
                        incidents_list.append(df_incident)
                except Exception as e:
                    print(f"  Warning: Error processing sheet '{sheet_name}': {e}")
                    continue
            
            print(f"  Processed {len(incidents_list)} sheets from Incident Database")
        except Exception as e:
            print(f"  Warning: Error processing Incident Database: {e}")
    
    # 2. Process Alert actions (these contain failure/incident data too)
    alert_file = DATA_FOLDER / "Alert actions.xlsx"
    if alert_file.exists():
        print(f"Processing: {alert_file.name} (extracting incidents)")
        try:
            xls = pd.ExcelFile(alert_file)
            monthly_sheets = [s for s in xls.sheet_names if any(month in s.upper() for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'COMBINATION'])]
            
            for sheet_name in monthly_sheets + ['Alert Details']:
                if sheet_name not in xls.sheet_names:
                    continue
                
                try:
                    df = pd.read_excel(alert_file, sheet_name=sheet_name)
                    df = standardize_column_names(df)
                    
                    if 'vessel_name' not in df.columns or 'failure_date' not in df.columns:
                        continue
                    
                    df_incident = pd.DataFrame()
                    df_incident['vessel_name'] = df['vessel_name'].apply(standardize_vessel_name)
                    df_incident['failure_date'] = df['failure_date']
                    df_incident['incident_description'] = df.get('alert_description', df.get('incident_description', ''))
                    df_incident['component'] = df.get('alert_description', '').apply(extract_component_from_description)
                    df_incident['failure_mode'] = df_incident['incident_description'].apply(extract_failure_mode)
                    df_incident['incident_category'] = df.get('alert_category', None)
                    df_incident['source_file'] = alert_file.name
                    df_incident['source_sheet'] = sheet_name
                    df_incident['doc_office'] = None
                    
                    # Parse dates
                    df_incident = parse_date_column(df_incident, 'failure_date')
                    
                    # Remove rows with no vessel or date
                    df_incident = df_incident[df_incident['vessel_name'].notna()]
                    df_incident = df_incident[df_incident['failure_date'].notna()]
                    
                    if len(df_incident) > 0:
                        incidents_list.append(df_incident)
                except Exception as e:
                    print(f"  Warning: Error processing sheet '{sheet_name}': {e}")
                    continue
        except Exception as e:
            print(f"  Warning: Error processing Alert actions: {e}")
    
    # Combine all incidents
    if incidents_list:
        master_incidents = pd.concat(incidents_list, ignore_index=True)
        
        # Remove duplicates (same vessel + date + description)
        master_incidents = master_incidents.drop_duplicates(
            subset=['vessel_name', 'failure_date', 'incident_description'],
            keep='first'
        )
        
        # Add incident ID
        master_incidents['incident_id'] = range(1, len(master_incidents) + 1)
        
        print(f"  Created master_incidents with {len(master_incidents)} records")
        print(f"  Date range: {master_incidents['failure_date'].min()} to {master_incidents['failure_date'].max()}")
        
        return master_incidents
    else:
        print("  Warning: No incident data found!")
        return pd.DataFrame()


# ============================================================================
# STEP 3: CREATE MASTER ACTIONS TABLE
# Section 4.3: Create Master Actions Table
# ============================================================================

def create_master_actions():
    """
    Section 4.3: Create Master Actions Table
    Merge all action sheets and parse action types
    """
    print("\n" + "=" * 60)
    print("STEP 3: Creating Master Actions Table")
    print("=" * 60)
    
    actions_list = []
    
    alert_file = DATA_FOLDER / "Alert actions.xlsx"
    if alert_file.exists():
        print(f"Processing: {alert_file.name}")
        try:
            xls = pd.ExcelFile(alert_file)
            # Process monthly sheets and Alert Details
            sheets_to_process = [s for s in xls.sheet_names if any(month in s.upper() for month in 
                ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'COMBINATION'])] + ['Alert Details']
            
            for sheet_name in sheets_to_process:
                if sheet_name not in xls.sheet_names:
                    continue
                
                try:
                    df = pd.read_excel(alert_file, sheet_name=sheet_name)
                    df = standardize_column_names(df)
                    
                    if 'vessel_name' not in df.columns:
                        continue
                    
                    df_action = pd.DataFrame()
                    df_action['vessel_name'] = df['vessel_name'].apply(standardize_vessel_name)
                    df_action['action_date'] = df.get('failure_date', df.get('action_date', None))  # Use failure_date as action_date
                    df_action['action_taken'] = df.get('action_taken', '')
                    df_action['remarks'] = df.get('remarks', '')
                    df_action['alert_category'] = df.get('alert_category', None)
                    df_action['fleet'] = df.get('fleet', None)
                    df_action['source_sheet'] = sheet_name
                    
                    # Parse action type
                    df_action['action_type'] = df_action['action_taken'].apply(parse_action_type)
                    
                    # Extract root cause
                    df_action['root_cause'] = df_action['remarks'].apply(categorize_root_cause)
                    
                    # Extract component
                    df_action['component'] = df.get('alert_description', '').apply(extract_component_from_description)
                    
                    # Parse dates
                    df_action = parse_date_column(df_action, 'action_date')
                    
                    # Remove rows with no vessel or date
                    df_action = df_action[df_action['vessel_name'].notna()]
                    df_action = df_action[df_action['action_date'].notna()]
                    
                    if len(df_action) > 0:
                        actions_list.append(df_action)
                except Exception as e:
                    print(f"  Warning: Error processing sheet '{sheet_name}': {e}")
                    continue
            
            print(f"  Processed {len(actions_list)} sheets from Alert actions")
        except Exception as e:
            print(f"  Warning: Error processing Alert actions: {e}")
    
    # Combine all actions
    if actions_list:
        master_actions = pd.concat(actions_list, ignore_index=True)
        
        # Add action ID
        master_actions['action_id'] = range(1, len(master_actions) + 1)
        
        print(f"  Created master_actions with {len(master_actions)} records")
        print(f"  Action types: {master_actions['action_type'].value_counts().to_dict()}")
        
        return master_actions
    else:
        print("  Warning: No action data found!")
        return pd.DataFrame()


# ============================================================================
# STEP 4: JOIN ALL MASTERS AND CALCULATE METRICS
# Section 4.5: Join All Masters
# Section 5: Calculation Methodology
# ============================================================================

def calculate_metrics_and_join(master_equipment, master_incidents, master_actions, master_parts=None):
    """
    Section 4.5 & 5: Join all masters and calculate metrics
    """
    print("\n" + "=" * 60)
    print("STEP 4: Joining Data and Calculating Metrics")
    print("=" * 60)
    
    if master_equipment.empty or master_incidents.empty:
        print("  Error: Missing required master tables!")
        return pd.DataFrame()
    
    if master_parts is None:
        master_parts = pd.DataFrame()
    
    # Join equipment with incidents (on vessel_name + component)
    print("  Joining equipment with incidents...")
    unified = master_equipment.merge(
        master_incidents,
        on='vessel_name',
        how='left',
        suffixes=('', '_incident')
    )
    
    # Join with actions (on vessel_name + component + date range)
    if not master_actions.empty:
        print("  Joining with actions...")
        # Match actions to failures (action_date within 90 days of failure_date)
        unified = unified.merge(
            master_actions,
            on='vessel_name',
            how='left',
            suffixes=('', '_action')
        )
        # Filter actions within 90 days of failure
        if 'action_date' in unified.columns and 'failure_date' in unified.columns:
            unified['days_diff'] = (unified['action_date'] - unified['failure_date']).dt.days
            unified = unified[(unified['days_diff'].isna()) | ((unified['days_diff'] >= 0) & (unified['days_diff'] <= 90))]
    
    # Parts data will be used in metrics calculation, not joined here
    # (to avoid data explosion from many-to-many relationships)
    
    # Calculate metrics per make/model/component
    print("  Calculating metrics...")
    
    # Group by make_model and component_id for metric calculation
    metrics_list = []
    
    for (make_model, component_id), group in unified.groupby(['make_model', 'component_id']):
        if pd.isna(make_model) or pd.isna(component_id):
            continue
        
        # Get failures for this make/model/component
        failures = group[group['failure_date'].notna()].copy()
        
        if len(failures) == 0:
            continue
        
        # Sort by failure date
        failures = failures.sort_values('failure_date')
        
        # Calculate MTBF (Section 5.1)
        mtbf_hours = None
        if len(failures) > 1:
            time_diffs = failures['failure_date'].diff().dropna()
            time_diffs_hours = time_diffs.dt.total_seconds() / 3600
            mtbf_hours = time_diffs_hours.mean() if len(time_diffs_hours) > 0 else None
        
        # Calculate failure rate (Section 5.4 - estimate from date range)
        failure_rate_per_1000h = None
        if len(failures) > 0:
            date_range = (failures['failure_date'].max() - failures['failure_date'].min()).days
            if date_range > 0:
                # Estimate run hours (assume 24/7 operation)
                estimated_run_hours = date_range * 24
                failure_rate_per_1000h = (len(failures) / estimated_run_hours) * 1000 if estimated_run_hours > 0 else None
        
        # Get actions for this group
        actions = group[group['action_type'].notna()].copy()
        repair_actions = actions[actions['action_type'] == 'Repair']
        
        # Calculate MTBR (Section 5.2)
        mtbr_hours = None
        if len(repair_actions) > 1:
            repair_actions = repair_actions.sort_values('action_date')
            time_diffs = repair_actions['action_date'].diff().dropna()
            time_diffs_hours = time_diffs.dt.total_seconds() / 3600
            mtbr_hours = time_diffs_hours.mean() if len(time_diffs_hours) > 0 else None
        
        # Calculate MTTR (Section 5.3)
        mttr_hours = None
        if len(failures) > 0 and len(repair_actions) > 0:
            mttr_list = []
            for idx, failure in failures.iterrows():
                failure_date = failure['failure_date']
                # Find first repair after this failure
                repairs_after = repair_actions[repair_actions['action_date'] >= failure_date]
                if len(repairs_after) > 0:
                    first_repair = repairs_after.iloc[0]
                    mttr = (first_repair['action_date'] - failure_date).total_seconds() / 3600
                    if mttr >= 0:  # Only positive MTTR
                        mttr_list.append(mttr)
            
            if len(mttr_list) > 0:
                mttr_hours = np.mean(mttr_list)
        
        # Root cause analytics (Section 5.5)
        root_cause_counts = group['root_cause'].value_counts().to_dict() if 'root_cause' in group.columns else {}
        total_failures = len(failures)
        root_cause_percentage = (total_failures - root_cause_counts.get('Unknown', 0)) / total_failures * 100 if total_failures > 0 else 0
        
        # Failure mode analytics (NEW) - Calculate failure rate per mode
        failure_mode_data = {}
        if 'failure_mode' in group.columns:
            for failure_mode, fm_group in group.groupby('failure_mode'):
                fm_failures = fm_group[fm_group['failure_date'].notna()]
                if len(fm_failures) > 0:
                    # Calculate failure rate for this mode
                    fm_date_range = (fm_failures['failure_date'].max() - fm_failures['failure_date'].min()).days
                    fm_estimated_hours = fm_date_range * 24 if fm_date_range > 0 else 1
                    fm_failure_rate = (len(fm_failures) / fm_estimated_hours) * 1000 if fm_estimated_hours > 0 else 0
                    
                    # Calculate MTBF for this mode
                    fm_mtbf = None
                    if len(fm_failures) > 1:
                        fm_sorted = fm_failures.sort_values('failure_date')
                        fm_time_diffs = fm_sorted['failure_date'].diff().dropna()
                        fm_time_diffs_hours = fm_time_diffs.dt.total_seconds() / 3600
                        fm_mtbf = fm_time_diffs_hours.mean() if len(fm_time_diffs_hours) > 0 else None
                    
                    failure_mode_data[failure_mode] = {
                        'count': len(fm_failures),
                        'failure_rate': fm_failure_rate,
                        'mtbf': fm_mtbf,
                        'vessels': fm_group['vessel_name'].nunique()
                    }
        
        # Get top failure mode by failure rate (not just count)
        top_failure_mode = None
        top_failure_mode_count = 0
        top_failure_mode_rate = 0
        if failure_mode_data:
            top_failure_mode = max(failure_mode_data.items(), key=lambda x: x[1]['failure_rate'])[0]
            top_failure_mode_count = failure_mode_data[top_failure_mode]['count']
            top_failure_mode_rate = failure_mode_data[top_failure_mode]['failure_rate']
        
        # Store all failure mode data as JSON string for later use
        failure_mode_distribution = str(failure_mode_data)
        
        # Vessels affected
        vessels_affected = group['vessel_name'].nunique()
        
        # Trend analysis (NEW) - Compare recent 3 months vs previous 3 months
        trend = None
        if len(failures) > 0:
            failures_sorted = failures.sort_values('failure_date')
            mid_point = len(failures_sorted) // 2
            if mid_point > 0:
                recent_failures = len(failures_sorted[mid_point:])
                older_failures = len(failures_sorted[:mid_point])
                if recent_failures > older_failures * 1.1:
                    trend = '↑'  # Increasing
                elif recent_failures < older_failures * 0.9:
                    trend = '↓'  # Decreasing
                else:
                    trend = '→'  # Stable
        
        # Extract stakeholder communications from remarks
        stakeholder_communications = []
        if 'remarks' in group.columns:
            for remark in group['remarks'].dropna():
                comm = extract_stakeholder_communication(remark)
                if comm:
                    stakeholder_communications.append(comm)
        
        stakeholder_summary = ', '.join(set(stakeholder_communications)) if stakeholder_communications else None
        
        # Enhanced Action Analytics (NEW)
        action_type_counts = actions['action_type'].value_counts().to_dict() if len(actions) > 0 else {}
        action_recurrence_rate = None
        action_recurrence_by_type = {}  # NEW: Recurrence rate per action type
        highest_recurrence_action = None  # NEW: Action type with highest recurrence
        highest_recurrence_rate = 0  # NEW: Highest recurrence rate value
        mtbf_before_action = None
        mtbf_after_action = None
        
        # Calculate recurrence rate (failures after same action type)
        if len(actions) > 0 and len(failures) > 0:
            # Group actions by type and check if failures recurred
            recurring_actions = 0
            total_actions = len(actions)
            
            # Calculate recurrence rate per action type
            for action_type, action_group in actions.groupby('action_type'):
                action_dates = action_group['action_date'].dropna()
                recurring_count = 0
                total_for_type = len(action_dates)
                
                if total_for_type > 0:
                    for action_date in action_dates:
                        failures_after = failures[failures['failure_date'] > action_date]
                        if len(failures_after) > 0:
                            recurring_count += 1
                            recurring_actions += 1
                    
                    recurrence_rate_for_type = (recurring_count / total_for_type) * 100
                    action_recurrence_by_type[action_type] = recurrence_rate_for_type
                    
                    # Track highest recurrence
                    if recurrence_rate_for_type > highest_recurrence_rate:
                        highest_recurrence_rate = recurrence_rate_for_type
                        highest_recurrence_action = action_type
            
            if total_actions > 0:
                action_recurrence_rate = (recurring_actions / total_actions) * 100
        
        # MTBF before/after actions (simplified - compare first half vs second half)
        if len(failures) > 4:
            mid_point = len(failures) // 2
            first_half = failures.iloc[:mid_point]
            second_half = failures.iloc[mid_point:]
            
            if len(first_half) > 1:
                time_diffs_1 = first_half['failure_date'].diff().dropna()
                mtbf_before_action = (time_diffs_1.dt.total_seconds() / 3600).mean() if len(time_diffs_1) > 0 else None
            
            if len(second_half) > 1:
                time_diffs_2 = second_half['failure_date'].diff().dropna()
                mtbf_after_action = (time_diffs_2.dt.total_seconds() / 3600).mean() if len(time_diffs_2) > 0 else None
        
        # Create metrics record
        metrics_record = {
            'make_model': make_model,
            'component_id': component_id,
            'equipment_type': group['equipment_type'].iloc[0] if 'equipment_type' in group.columns else None,
            'make': group['make'].iloc[0] if 'make' in group.columns else None,
            'model': group['model'].iloc[0] if 'model' in group.columns else None,
            'failure_count': total_failures,
            'mtbf_hours': mtbf_hours,
            'mtbr_hours': mtbr_hours,
            'mttr_hours': mttr_hours,
            'failure_rate_per_1000h': failure_rate_per_1000h,
            'vessels_affected': vessels_affected,
            'root_cause_human_error': root_cause_counts.get('Human Error', 0),
            'root_cause_operational_error': root_cause_counts.get('Operational Error', 0),
            'root_cause_machinery_failure': root_cause_counts.get('Machinery Failure', 0),
            'root_cause_other': root_cause_counts.get('Other', 0),
            'root_cause_unknown': root_cause_counts.get('Unknown', 0),
            'root_cause_percentage_confirmed': root_cause_percentage,
            'first_failure_date': failures['failure_date'].min(),
            'last_failure_date': failures['failure_date'].max(),
            # NEW: Failure Mode Analytics
            'top_failure_mode': top_failure_mode,
            'top_failure_mode_count': top_failure_mode_count,
            'top_failure_mode_rate': top_failure_mode_rate,
            'failure_mode_distribution': failure_mode_distribution,  # Store as string for CSV
            # NEW: Trend Analysis
            'trend': trend,
            # NEW: Enhanced Action Analytics
            'action_inspect_count': action_type_counts.get('Inspect', 0),
            'action_repair_count': action_type_counts.get('Repair', 0),
            'action_replace_count': action_type_counts.get('Replace', 0),
            'action_temporary_count': action_type_counts.get('Temporary', 0),
            'action_recurrence_rate': action_recurrence_rate,
            'action_recurrence_by_type': str(action_recurrence_by_type),  # Store as string for CSV
            'highest_recurrence_action': highest_recurrence_action,
            'highest_recurrence_rate': highest_recurrence_rate,
            'mtbf_before_action': mtbf_before_action,
            'mtbf_after_action': mtbf_after_action,
            # Flag: Actions that permanently eliminate failures (MTBF after > 2x MTBF before)
            'action_permanently_eliminates': (mtbf_after_action is not None and mtbf_before_action is not None and 
                                             mtbf_after_action > mtbf_before_action * 2) if (mtbf_after_action and mtbf_before_action) else False,
            # NEW: Stakeholder Communication Summary
            'stakeholder_communication_summary': stakeholder_summary,
            # NEW: Cost Impact (will be calculated if parts data available)
            'estimated_cost_per_failure': None,
            'total_cost_impact': None,
            'stock_risk': None,  # Will be calculated: Low/Medium/High
            'parts_per_failure': None,
            'top_parts': None,  # JSON list of top part names
            'top_parts_consumption': None,  # JSON list of consumption counts
        }
        
        # Calculate cost impact if parts data available
        if not master_parts.empty and 'component_id' in master_parts.columns:
            # Match parts to this component_id (AE1, AE2, AE3, AE4, ME1)
            component_parts = master_parts[master_parts['component_id'] == component_id].copy()
            
            if len(component_parts) > 0:
                # Calculate total cost from parts
                if 'part_cost' in component_parts.columns and 'quantity' in component_parts.columns:
                    component_parts['_cost'] = pd.to_numeric(component_parts['part_cost'], errors='coerce')
                    component_parts['_qty'] = pd.to_numeric(component_parts['quantity'], errors='coerce')
                    
                    # Fill NaN quantities with 1
                    component_parts['_qty'] = component_parts['_qty'].fillna(1)
                    
                    total_parts_cost = (component_parts['_cost'].fillna(0) * component_parts['_qty']).sum()
                    total_parts_qty = component_parts['_qty'].sum()
                    
                    if total_failures > 0 and total_parts_cost > 0:
                        metrics_record['estimated_cost_per_failure'] = total_parts_cost / total_failures
                        metrics_record['total_cost_impact'] = total_parts_cost
                    
                    # Calculate parts per failure
                    if total_failures > 0:
                        metrics_record['parts_per_failure'] = total_parts_qty / total_failures
                    
                    # Stock risk assessment (based on failure rate and parts consumption)
                    if total_failures > 0:
                        parts_per_failure = total_parts_qty / total_failures
                        if parts_per_failure > 2 and failure_rate_per_1000h and failure_rate_per_1000h > 100:
                            metrics_record['stock_risk'] = 'High'
                        elif parts_per_failure > 1 and failure_rate_per_1000h and failure_rate_per_1000h > 50:
                            metrics_record['stock_risk'] = 'Medium'
                        else:
                            metrics_record['stock_risk'] = 'Low'
                
                # Store top parts for this component (for Level 3 display)
                top_parts = component_parts.groupby('part_name').agg({
                    'quantity': 'sum',
                    'part_cost': 'sum'
                }).reset_index().sort_values('quantity', ascending=False).head(5)
                
                metrics_record['top_parts'] = str(top_parts['part_name'].tolist())
                metrics_record['top_parts_consumption'] = str(top_parts['quantity'].tolist())
        
        metrics_list.append(metrics_record)
    
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        print(f"  Calculated metrics for {len(metrics_df)} make/model/component combinations")
        
        # Add Make/Model Summary Metrics (Section 5.9)
        print("  Calculating Make/Model summary metrics...")
        make_model_summary = calculate_make_model_summary(metrics_df)
        
        # Identify Bad Actors per Make/Model (Section 5.8)
        print("  Identifying bad actors per Make/Model...")
        bad_actors = identify_bad_actors(metrics_df)
        
        # Add fleet average MTBF to each component for comparison
        fleet_avg_mtbf = metrics_df['mtbf_hours'].mean() if metrics_df['mtbf_hours'].notna().any() else None
        metrics_df['fleet_avg_mtbf_hours'] = fleet_avg_mtbf
        metrics_df['mtbf_vs_fleet'] = metrics_df.apply(
            lambda row: (row['mtbf_hours'] - fleet_avg_mtbf) / fleet_avg_mtbf * 100 if pd.notna(row['mtbf_hours']) and pd.notna(fleet_avg_mtbf) and fleet_avg_mtbf > 0 else None,
            axis=1
        )
        
        # Combine all data
        final_data = metrics_df.merge(
            make_model_summary,
            on='make_model',
            how='left',
            suffixes=('', '_summary')
        )
        
        final_data = final_data.merge(
            bad_actors,
            on=['make_model', 'component_id'],
            how='left',
            suffixes=('', '_bad_actor')
        )
        
        return final_data
    else:
        print("  Warning: No metrics calculated!")
        return pd.DataFrame()


def calculate_make_model_summary(metrics_df):
    """
    Section 5.9: Make/Model Summary Metrics
    Calculate summary metrics for each Make/Model combination
    """
    # Calculate fleet-wide averages for comparison
    fleet_avg_mtbf = metrics_df['mtbf_hours'].mean() if metrics_df['mtbf_hours'].notna().any() else None
    fleet_avg_failure_rate = metrics_df['failure_rate_per_1000h'].mean() if metrics_df['failure_rate_per_1000h'].notna().any() else None
    
    summary_list = []
    
    for make_model, group in metrics_df.groupby('make_model'):
        summary = {
            'make_model': make_model,
            'total_failures': group['failure_count'].sum(),
            'total_vessels': group['vessels_affected'].max() if 'vessels_affected' in group.columns else group['vessels_affected'].sum(),
            'overall_mtbf_hours': group['mtbf_hours'].mean() if group['mtbf_hours'].notna().any() else None,
            'overall_failure_rate_per_1000h': group['failure_rate_per_1000h'].mean() if group['failure_rate_per_1000h'].notna().any() else None,
            'component_count': group['component_id'].nunique(),
            'first_failure_date': group['first_failure_date'].min(),
            'last_failure_date': group['last_failure_date'].max(),
            'equipment_type': group['equipment_type'].iloc[0] if 'equipment_type' in group.columns else None,
            'make': group['make'].iloc[0] if 'make' in group.columns else None,
            'model': group['model'].iloc[0] if 'model' in group.columns else None,
            # NEW: Fleet comparison metrics
            'fleet_avg_mtbf_hours': fleet_avg_mtbf,
            'fleet_avg_failure_rate_per_1000h': fleet_avg_failure_rate,
        }
        summary_list.append(summary)
    
    return pd.DataFrame(summary_list)


def identify_bad_actors(metrics_df):
    """
    Section 5.8: Top 10 Bad Actors per Make/Model
    Identify bad actors by components, failure modes, and parts
    """
    bad_actor_list = []
    
    for make_model, group in metrics_df.groupby('make_model'):
        # Filter components with at least 3 failures
        valid_components = group[group['failure_count'] >= 3].copy()
        
        if len(valid_components) == 0:
            continue
        
        # Rank by failure rate (descending)
        valid_components = valid_components.sort_values('failure_rate_per_1000h', ascending=False, na_position='last')
        
        # Assign rank (Top 10)
        valid_components['bad_actor_rank'] = range(1, len(valid_components) + 1)
        valid_components['is_bad_actor'] = valid_components['bad_actor_rank'] <= 10
        
        # Add make_model for merging
        valid_components['make_model'] = make_model
        
        bad_actor_list.append(valid_components[['make_model', 'component_id', 'bad_actor_rank', 'is_bad_actor']])
    
    if bad_actor_list:
        return pd.concat(bad_actor_list, ignore_index=True)
    else:
        return pd.DataFrame(columns=['make_model', 'component_id', 'bad_actor_rank', 'is_bad_actor'])


# ============================================================================
# STEP 5: PROCESS SPARE PARTS DATA
# ============================================================================

# SSDG to Component mapping for parts
SSDG_TO_COMPONENT = {
    'SSDG1': 'AE1',
    'SSDG2': 'AE2', 
    'SSDG3': 'AE3',
    'SSDG4': 'AE4',
}


def extract_parts_from_sheet2_or_3(parts_file, sheet_name):
    """
    Extract parts data from Sheet2 or Sheet3 of DG RMA Analysis.xlsm
    Headers are in row 2 (0-indexed): SYS, Comp, SubComp, FD Code, Detection Date, etc.
    SubComp column contains the actual PARTS!
    """
    parts_list = []
    
    try:
        # Headers are in row 2
        df = pd.read_excel(parts_file, sheet_name=sheet_name, header=2)
        df.columns = [str(c).strip() for c in df.columns]
        
        # Find columns by exact name
        sys_col = 'SYS' if 'SYS' in df.columns else None
        comp_col = 'Comp' if 'Comp' in df.columns else None
        subcomp_col = 'SubComp' if 'SubComp' in df.columns else None
        mh_col = 'MH' if 'MH' in df.columns else None
        date_col = 'Detection Date' if 'Detection Date' in df.columns else None
        
        if subcomp_col is None:
            print(f"    SubComp column not found in {sheet_name}")
            return parts_list
        
        for idx, row in df.iterrows():
            subcomp_val = row.get(subcomp_col)
            
            if pd.isna(subcomp_val):
                continue
            
            part_name = str(subcomp_val).strip()
            if part_name.upper() in ['NA', 'N/A', 'NAN', '', 'SUBCOMP']:
                continue
            
            # Get system (SSDG1-4)
            system = None
            if sys_col and pd.notna(row.get(sys_col)):
                system = str(row.get(sys_col)).strip().upper()
            
            # Get component (Intake/Exhaust, Cooling System, etc.)
            component = None
            if comp_col and pd.notna(row.get(comp_col)):
                component = str(row.get(comp_col)).strip()
            
            # Get date
            part_date = None
            if date_col:
                part_date = pd.to_datetime(row.get(date_col), errors='coerce')
            
            # Get cost from MH (man-hours * $50/hour)
            part_cost = None
            if mh_col:
                mh_val = pd.to_numeric(row.get(mh_col), errors='coerce')
                if pd.notna(mh_val) and mh_val > 0:
                    part_cost = mh_val * 50
            
            # Map SSDG to component_id
            component_id = SSDG_TO_COMPONENT.get(system)
            
            parts_list.append({
                'part_name': part_name,
                'system': system,
                'component': component,
                'component_id': component_id,
                'part_date': part_date,
                'part_cost': part_cost,
                'quantity': 1,
                'source_sheet': sheet_name,
            })
        
        return parts_list
    
    except Exception as e:
        print(f"    Error processing {sheet_name}: {e}")
        return parts_list


def extract_parts_from_sheet5(parts_file):
    """
    Extract parts data from Sheet5 of DG RMA Analysis.xlsm (wide format)
    - SSDG1, SSDG2, SSDG3, SSDG4 columns = system indicators
    - Part name columns = "Bearings", "Turbocharger", "LO Cooler", etc.
    - Cell values = consumption count
    """
    parts_list = []
    
    try:
        df = pd.read_excel(parts_file, sheet_name='Sheet5')
        df.columns = [str(c).strip() for c in df.columns]
        
        # System columns
        system_cols = ['SSDG1', 'SSDG2', 'SSDG3', 'SSDG4']
        
        # Find part name columns (exclude Unnamed, single letters, numeric)
        part_cols = []
        for col in df.columns:
            col_str = str(col).strip()
            
            # Skip system columns
            if col_str.upper() in ['SSDG1', 'SSDG2', 'SSDG3', 'SSDG4']:
                continue
            # Skip Unnamed
            if col_str.startswith('Unnamed'):
                continue
            # Skip single letters
            if len(col_str) <= 2 and col_str.replace('.', '').isalpha():
                continue
            # Skip NA
            if col_str.upper() == 'NA':
                continue
            # Skip numeric column names
            try:
                float(col_str)
                continue
            except ValueError:
                pass
            
            part_cols.append(col_str)
        
        # For each part column, count total consumption per system
        for part_col in part_cols:
            if part_col not in df.columns:
                continue
            
            # Get the part column values as numeric
            part_values = pd.to_numeric(df[part_col], errors='coerce')
            
            # Count total for this part
            total_consumption = part_values.sum()
            
            if pd.notna(total_consumption) and total_consumption > 0:
                # For each system, check if this part was used
                for system in system_cols:
                    if system not in df.columns:
                        continue
                    
                    # Convert system column to numeric
                    sys_values = pd.to_numeric(df[system], errors='coerce')
                    
                    # Count rows where both system AND part have values
                    mask = (sys_values.notna() & (sys_values > 0)) & \
                           (part_values.notna() & (part_values > 0))
                    
                    count = mask.sum()
                    
                    if count > 0:
                        component_id = SSDG_TO_COMPONENT.get(system)
                        
                        parts_list.append({
                            'part_name': part_col,
                            'system': system,
                            'component': None,
                            'component_id': component_id,
                            'part_date': None,
                            'part_cost': None,
                            'quantity': int(count),
                            'source_sheet': 'Sheet5',
                        })
        
        return parts_list
    
    except Exception as e:
        print(f"    Error processing Sheet5: {e}")
        return parts_list


def create_master_parts():
    """
    Process spare parts/RMA data from DG RMA Analysis.xlsm
    Corrected to properly extract parts from:
    - Sheet2/Sheet3: SubComp column (with header=2)
    - Sheet5: Part names as column headers (wide format)
    """
    print("\n" + "=" * 60)
    print("STEP 5: Processing Spare Parts Data")
    print("=" * 60)
    
    parts_file = DATA_FOLDER / "DG RMA Analysis.xlsm"
    if not parts_file.exists():
        print(f"  Note: {parts_file.name} not found. Spare parts data will be unavailable.")
        return pd.DataFrame()
    
    all_parts = []
    
    try:
        print(f"  Processing: {parts_file.name}")
        
        # Process Sheet2 (SubComp = parts)
        print("  Processing Sheet2...")
        parts_sheet2 = extract_parts_from_sheet2_or_3(parts_file, 'Sheet2')
        print(f"    Extracted {len(parts_sheet2)} parts from Sheet2")
        all_parts.extend(parts_sheet2)
        
        # Process Sheet3 (SubComp = parts)
        print("  Processing Sheet3...")
        parts_sheet3 = extract_parts_from_sheet2_or_3(parts_file, 'Sheet3')
        print(f"    Extracted {len(parts_sheet3)} parts from Sheet3")
        all_parts.extend(parts_sheet3)
        
        # Process Sheet5 (wide format)
        print("  Processing Sheet5 (wide format)...")
        parts_sheet5 = extract_parts_from_sheet5(parts_file)
        print(f"    Extracted {len(parts_sheet5)} parts from Sheet5")
        all_parts.extend(parts_sheet5)
        
        if all_parts:
            master_parts = pd.DataFrame(all_parts)
            
            print(f"\n  ✅ Created master_parts with {len(master_parts)} records")
            print(f"     Unique parts: {master_parts['part_name'].nunique()}")
            print(f"     With component_id: {master_parts['component_id'].notna().sum()}")
            print(f"     With cost: {master_parts['part_cost'].notna().sum()}")
            
            return master_parts
        else:
            print("  Warning: No parts data extracted")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"  Warning: Error processing parts file: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    Implements the complete data pipeline from IMPLEMENTATION_PLAN.txt
    """
    print("\n" + "=" * 60)
    print("ASSET FAILURE ANALYTICS - DATA PIPELINE")
    print("=" * 60)
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Output file: {OUTPUT_FILE}")
    print()
    
    # Step 1: Create master equipment table
    master_equipment = create_master_equipment()
    
    # Step 2: Create master incidents table
    master_incidents = create_master_incidents()
    
    # Step 3: Create master actions table
    master_actions = create_master_actions()
    
    # Step 4: Create master parts table (optional)
    master_parts = create_master_parts()
    
    # Step 5: Join and calculate metrics
    processed_data = calculate_metrics_and_join(master_equipment, master_incidents, master_actions, master_parts)
    
    # Save output
    if not processed_data.empty:
        processed_data.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ SUCCESS: Saved {len(processed_data)} records to {OUTPUT_FILE}")
        print(f"   Columns: {list(processed_data.columns)}")
    else:
        print("\n❌ ERROR: No data processed. Check input files and errors above.")
    
    return processed_data


if __name__ == "__main__":
    result = main()

