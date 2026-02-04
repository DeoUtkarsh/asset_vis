# Asset Failure Analytics - Data Flow & UI Explanation

## Executive Summary

This document explains how multiple Excel files are processed into a unified dataset and visualized through a 3-level interactive dashboard. The system identifies "bad actors" (problematic components) across a maritime fleet and provides actionable insights for maintenance decisions.

---

## 1. Data Sources & Processing

### 1.1 Excel Files Used

#### **Equipment Master Data (3 files):**
1. **Make & Model - Aux Engine -New.xlsx**
   - **Sheet:** `Make _ Model - Aux Engine -New`
   - **Purpose:** Equipment master data for Auxiliary Engines
   - **Records:** 2,667 equipment records (normalized from wide format: AE1, AE2, AE3, etc.)
   - **Key Columns:** Vessel Name, Make, Model, Component IDs

2. **Make & Model - Main Engine - New.xlsx**
   - **Sheet:** `Make _ Model - Main Engine - Ne`
   - **Purpose:** Equipment master data for Main Engines
   - **Key Columns:** Vessel Name, Make, Model, Component IDs

3. **Make & Model - BWTS.xlsx**
   - **Sheet:** `Make _ Model - BWTS`
   - **Purpose:** Equipment master data for Ballast Water Treatment Systems
   - **Key Columns:** Vessel Name, Make, Model, Component IDs

#### **Incident & Action Data (2 files):**
4. **Alert actions.xlsx**
   - **Sheets Used:** 12 monthly sheets (JAN 2025, FEB 2025, MAR 2025, APR 2025, MAY 2025, JUN 2025, JUL 2025, AUG 2025, SEP 2025, OCT 2025, Combination, Alert Details)
   - **Purpose:** Historical failure/incident records and actions taken
   - **Records:** 1,901 incidents, 2,959 actions
   - **Key Columns:** Vessel Name, Date of occurrence, Description, Action Taken, Remarks

5. **Incident Database.xlsx**
   - **Sheets Attempted:** 31 sheets
   - **Status:** Not used (column structure mismatch)
   - **Fallback:** Used Alert actions.xlsx instead

#### **Spare Parts Data (1 file):**
6. **DG RMA Analysis.xlsm**
   - **Sheets Used:** Sheet2, Sheet3, Sheet5
   - **Purpose:** Spare parts consumption and RMA (Return Material Authorization) data
   - **Records:** 377 parts records, 53 unique parts
   - **Key Columns:**
     - **Sheet2/Sheet3:** SYS, Comp, SubComp, Detection Date, MH (man-hours)
     - **Sheet5:** Wide format with part names as column headers (Bearings, Turbocharger, etc.)
   - **Special Logic:** Maps SSDG systems to components (SSDG1→AE1, SSDG2→AE2, SSDG3→AE3, SSDG4→AE4)

---

### 1.2 Data Joining Strategy

#### **Primary Keys:**
- **Vessel Name** (standardized to uppercase, cleaned)
- **Component ID** (extracted from descriptions: AE1, AE2, ME1, etc.)

#### **Join Process:**

1. **Step 1: Create Master Equipment Table**
   - Combine all Make & Model files
   - Normalize wide format (AE1, AE2, AE3 columns) to long format
   - Result: One row per Vessel + Make/Model + Component combination

2. **Step 2: Create Master Incidents Table**
   - Extract incidents from Alert actions.xlsx
   - Extract component IDs from descriptions using regex patterns
   - Extract failure modes (High Temperature, Leak, etc.) from descriptions
   - Standardize vessel names and dates
   - Result: One row per incident with component and failure mode identified

3. **Step 3: Create Master Actions Table**
   - Extract actions from Alert actions.xlsx
   - Categorize action types (Repair, Replace, Inspect, Temporary)
   - Extract root causes from remarks (Human Error, Operational Error, Machinery Failure, Other, Unknown)
   - Match actions to failures (within 90 days)
   - Result: One row per action with type and root cause

4. **Step 4: Create Master Parts Table**
   - Extract parts from DG RMA Analysis.xlsm (Sheet2, Sheet3, Sheet5)
   - Map SSDG systems to AE components
   - Calculate costs from man-hours (MH × $50/hour)
   - Normalize wide format (Sheet5) to long format
   - Result: One row per part consumption with component mapping

5. **Step 5: Join All Tables**
   - Join Equipment + Incidents on Vessel Name + Component
   - Join Actions to Incidents (within 90 days)
   - Join Parts to Components (via component_id mapping)
   - Result: Unified dataset with all metrics

---

### 1.3 Key Calculations

#### **Reliability Metrics:**
- **MTBF (Mean Time Between Failures):** `Total Operating Hours / Number of Failures`
- **Failure Rate:** `(Number of Failures / Total Operating Hours) × 1000` (failures per 1000h)
- **MTBR (Mean Time Between Repairs):** `Total Operating Hours / Number of Repairs` (if repair data available)
- **MTTR (Mean Time To Repair):** `Average(Repair Date - Failure Date)` (if repair dates available)

#### **Root Cause Analysis:**
- Categorize incidents into: Human Error, Operational Error, Machinery Failure, Other, Unknown
- Calculate percentage with confirmed root cause

#### **Failure Mode Analysis:**
- Extract failure modes from descriptions (High Temperature, Leak, Vibration, etc.)
- Calculate failure rate per mode
- Identify top failure mode for each component

#### **Action Analytics:**
- **Recurrence Rate:** Percentage of actions followed by another failure (ineffective actions)
- **MTBF Before vs After:** Compare MTBF before and after actions to measure effectiveness
- **Permanent Elimination:** Flag if MTBF improved >2x after actions

#### **Spare Parts Impact:**
- **Cost per Failure:** `(Parts Cost + Labor Cost) / Total Failures`
- **Total Cost Impact:** `Cost per Failure × Total Failures`
- **Parts per Failure:** `Total Parts Consumed / Total Failures`
- **Stock Risk:** Assessed based on consumption rate, lead time, and failure frequency (High/Medium/Low)

#### **Bad Actor Ranking:**
- Rank components by `failure_rate_per_1000h` (highest = worst)
- Top 10 per Make/Model are flagged as "bad actors"

---

### 1.4 Final Output: `processed_fleet_data.csv`

#### **Structure:**
- **187 rows** (one per Make/Model + Component combination)
- **58 columns** (metrics, root causes, actions, parts, summaries)
- **92 unique Make/Model combinations**
- **6 unique component types** (AE1, AE2, AE3, ME1, etc.)

#### **Key Column Categories:**

1. **Identifiers:** make_model, component_id, equipment_type, make, model
2. **Core Metrics:** failure_count, mtbf_hours, failure_rate_per_1000h, vessels_affected
3. **Root Causes:** root_cause_human_error, root_cause_operational_error, root_cause_machinery_failure, root_cause_other, root_cause_unknown
4. **Failure Modes:** top_failure_mode, top_failure_mode_count, top_failure_mode_rate, failure_mode_distribution
5. **Actions:** action_inspect_count, action_repair_count, action_replace_count, action_temporary_count, action_recurrence_rate, mtbf_before_action, mtbf_after_action
6. **Spare Parts:** estimated_cost_per_failure, total_cost_impact, stock_risk, parts_per_failure, top_parts, top_parts_consumption
7. **Fleet Comparison:** fleet_avg_mtbf_hours, mtbf_vs_fleet
8. **Make/Model Summary:** total_failures, total_vessels, overall_mtbf_hours, overall_failure_rate_per_1000h, component_count
9. **Bad Actor Flags:** bad_actor_rank, is_bad_actor

---

## 2. User Interface (UI) Flow

### 2.1 Overview: 3-Level Hierarchy

The dashboard follows a drill-down structure:
- **Level 1:** Make/Model Overview (92 combinations)
- **Level 2:** Top 10 Bad Actors for Selected Make/Model
- **Level 3:** Detailed Analytics for Selected Component

---

### 2.2 Level 1: Make/Model Selection (Landing Page)

#### **Purpose:**
High-level overview of all Make/Model combinations with summary metrics.

#### **What You See:**
- **Search Bar:** Filter Make/Model by name
- **Filter Dropdown:** Filter by Equipment Type (Aux Engine, Main Engine, BWTS)
- **Sort Dropdown:** Sort by Total Failures, Total Vessels, Failure Rate, or MTBF
- **Table with Columns:**
  - **Make/Model:** Equipment manufacturer and model
  - **Equipment Type:** Aux Engine, Main Engine, or BWTS
  - **Total Failures:** Sum of all failures across components (e.g., "24.85K")
  - **Total Vessels:** Number of vessels with this Make/Model
  - **MTBF (hours):** Average MTBF across all components (e.g., "1.1")
  - **Failure Rate / 1000h:** Average failure rate (e.g., "912.92")
  - **Components:** Number of components with failures (e.g., "3")
  - **Trend:** Arrow indicating trend (↑ increasing, ↓ decreasing, → stable)
  - **Cost Impact:** Total spare parts cost (e.g., "$8.65K")

#### **Data Source:**
- Aggregated from `processed_fleet_data.csv`
- One row per Make/Model (summary columns: total_failures, total_vessels, overall_mtbf_hours, etc.)

#### **User Action:**
- Click a row or select from dropdown → Navigate to Level 2

---

### 2.3 Level 2: Bad Actor Analysis

#### **Purpose:**
Show Top 10 Bad Actors for the selected Make/Model across three categories.

#### **What You See:**

**Header Section:**
- **Summary Metrics:** Total Failures, Total Vessels, Overall MTBF, Failure Rate, Components (for selected Make/Model)
- **Breadcrumb Navigation:** Shows current path (Make/Model List > Selected Make/Model)
- **Back Button:** Return to Level 1

**Three Tabs:**

##### **Tab 1: Components**
- **Table:** Top 10 components ranked by failure rate
- **Columns:**
  - **Component:** Component ID (AE1, AE2, AE3, etc.)
  - **Failure Rate / 1000h:** Failures per 1000 hours
  - **MTBF (hours):** Mean Time Between Failures
  - **MTBF vs Fleet (%):** Comparison to fleet average (negative = worse)
  - **Trend:** Arrow indicator
  - **Cost Impact:** Estimated cost per failure
  - **Vessels Affected:** Number of vessels
  - **Total Failures:** Count of failures
  - **Rank:** Bad actor ranking (1 = worst)

##### **Tab 2: Failure Modes**
- **Table:** Top 10 failure modes ranked by failure rate
- **Columns:**
  - **Failure Mode:** Type of failure (High Temperature, Leak, etc.)
  - **Failure Count:** Number of occurrences
  - **Failure Rate / 1000h:** Rate per 1000 hours
  - **MTBF (hours):** MTBF for this failure mode
  - **Affected Components:** Which components experience this mode
  - **Vessels Affected:** Number of vessels
  - **Trend:** Arrow indicator

##### **Tab 3: Spare Parts**
- **Table:** Top 10 spare parts ranked by consumption
- **Columns:**
  - **Part Name:** Spare part name (Coupling, Bearings, etc.)
  - **Consumption:** Total quantity consumed
  - **Est. Cost/Failure:** Average cost per failure
  - **Parts/Failure:** Average parts consumed per failure
  - **Stock Risk:** Risk level (High/Medium/Low)
- **Summary Metrics:**
  - **Total Parts Cost:** Total cost across all parts
  - **Avg Cost per Failure:** Average cost per failure
  - **Components with High Stock Risk:** Count of high-risk components

**Component Selector:**
- Dropdown to select a component for detailed analysis → Navigate to Level 3

#### **Data Source:**
- Filtered `processed_fleet_data.csv` for selected Make/Model
- Components tab: Filtered by `is_bad_actor == True`, sorted by `failure_rate_per_1000h`
- Failure Modes tab: Extracted from `top_failure_mode` and `failure_mode_distribution`
- Spare Parts tab: Parsed from `top_parts` and `top_parts_consumption` columns

---

### 2.4 Level 3: Detailed Component Analytics

#### **Purpose:**
Comprehensive drill-down for a specific component with all metrics, root causes, actions, and parts.

#### **What You See:**

**Header Section:**
- **Breadcrumb Navigation:** Make/Model List > Selected Make/Model > Selected Component
- **Back Button:** Return to Level 2

**Section 1: Component Analytics (Expanded by Default)**
- **5 Key Metrics:**
  - **MTBF:** Mean Time Between Failures (with visual signal if below peer average)
  - **MTBR:** Mean Time Between Repairs (N/A if no repair data)
  - **MTTR:** Mean Time To Repair (N/A if no repair data)
  - **Failure Rate:** Failures per 1000 hours
  - **Vessels Affected:** Number of vessels
- **Failure Timeline:** Date range from first to last failure

**Section 2: Failure Mode & Root Cause Analysis (Expandable)**
- **Root Cause Snapshot:**
  - **% with Confirmed Root Cause:** Percentage of failures with identified root cause
  - **Warning Banner:** If < 50% have confirmed root causes
- **Root Cause Distribution:**
  - **Pie Chart:** Visual breakdown of root causes (Human Error, Operational Error, Machinery Failure, Other, Unknown)
  - **Table:** Count and percentage for each root cause category
- **Failure Mode Breakdown:**
  - Top failure modes with counts and rates

**Section 3: Action Taken & Spare Parts Impact (Expandable)**
- **Action Taken Analytics:**
  - **Action Type Counts:** Inspect, Repair, Replace, Temporary
  - **Overall Action Recurrence Rate:** Percentage of ineffective actions (warning if > 50%)
  - **Action with Highest Recurrence:** Most problematic action type (critical warning if > 70%)
  - **MTBF Before vs After Actions:**
    - MTBF Before, MTBF After, Change percentage
    - Auto-flags for repeated temporary mitigations or repairs with no improvement
    - Success indicator if actions permanently eliminated failures (>2x MTBF improvement)
- **Spare Parts Impact:**
  - **Cost per Failure:** Average cost
  - **Total Cost Impact:** Total cost across all failures
  - **Parts per Failure:** Average parts consumed
  - **Stock Risk:** Risk level with color indicator (🟢 Low, 🟡 Medium, 🔴 High)
  - **Top Parts Table:** List of top consumed parts with consumption quantities
- **Stakeholder Communication Summary:** Stakeholders mentioned in remarks (supplier, manufacturer, etc.)

**Section 4: Decision & Action Panel (Sticky)**
- **5 Action Buttons:**
  - 🔄 Change PM Strategy
  - 🔧 Engineering Fix
  - 📞 Supplier Escalation
  - 📦 Increase Spares
  - 👁️ Monitor Only
- **Action Tracking Form:** (Appears after button click)
  - Owner, Due Date, Expected Outcome, Status
  - Save button to record action

#### **Data Source:**
- Single row from `processed_fleet_data.csv` for selected Make/Model + Component
- All metrics, root causes, actions, and parts data from this single row

---

## 3. Data Flow Summary

### 3.1 Processing Pipeline

```
Excel Files (6 files, 105+ sheets)
    ↓
[Data Pipeline: data_pipeline.py]
    ├─ Standardize column names
    ├─ Clean vessel names
    ├─ Extract components from descriptions
    ├─ Extract failure modes
    ├─ Categorize root causes
    ├─ Parse action types
    ├─ Match actions to failures
    ├─ Join parts data
    ├─ Calculate metrics (MTBF, MTTR, etc.)
    ├─ Rank bad actors
    └─ Aggregate Make/Model summaries
    ↓
processed_fleet_data.csv (187 rows × 58 columns)
    ↓
[Streamlit Dashboard: app.py]
    ├─ Level 1: Make/Model Overview (aggregate summaries)
    ├─ Level 2: Bad Actors (filter by Make/Model)
    └─ Level 3: Detailed Analytics (filter by Make/Model + Component)
```

### 3.2 Key Data Transformations

1. **Wide to Long Format:** Converted component columns (AE1, AE2, AE3) to rows
2. **Component Extraction:** Used regex to identify components from text descriptions
3. **Fuzzy Matching:** Standardized vessel names for consistent joining
4. **Date Matching:** Matched actions to failures within 90-day window
5. **Parts Mapping:** Mapped SSDG systems to AE components
6. **Aggregation:** Calculated component-level and Make/Model-level summaries

---

## 4. Key Metrics Explained

### 4.1 Reliability Metrics

- **MTBF (Mean Time Between Failures):** Average operating time between consecutive failures. Higher = better reliability.
- **Failure Rate:** Number of failures per 1,000 operating hours. Lower = better reliability.
- **MTBR (Mean Time Between Repairs):** Average time between repair actions. Lower = more frequent repairs needed.
- **MTTR (Mean Time To Repair):** Average time to complete a repair. Lower = faster repairs.

### 4.2 Action Effectiveness Metrics

- **Recurrence Rate:** Percentage of actions followed by another failure. Lower = more effective actions.
- **MTBF Before vs After:** Comparison of reliability before and after actions. Positive change = improvement.
- **Permanent Elimination:** Flag indicating if actions resulted in >2x MTBF improvement (sustained success).

### 4.3 Cost & Parts Metrics

- **Cost per Failure:** Average cost of spare parts and labor per failure event.
- **Total Cost Impact:** Total financial impact across all failures for a component.
- **Stock Risk:** Assessment of risk for running out of critical spare parts (High/Medium/Low).

---

## 5. Files Not Used (and Why)

1. **Consolidated Incidents.xlsx:** Duplicate of Incident Database.xlsx
2. **defects-report-2025-11-18.xlsx:** Single-date snapshot (Alert actions.xlsx covers longer period)
3. **Fleet Wise Vessel Issue.xlsx:** Fleet-level summaries, not component-level failures
4. **FMECA Documents.xlsx:** Design/risk assessment, not historical failure data
5. **SMARTShip - Use Cases.xlsx:** Use case tracking, not failure records
6. **summary_of the incidents.xlsx:** Aggregated summaries, not raw incident data
7. **Vessel Incident and Injury.xlsx:** Safety incidents, not equipment failures

---

## 6. Technical Stack

- **Data Processing:** Python, pandas, numpy
- **Dashboard:** Streamlit
- **Visualization:** Plotly
- **Data Storage:** CSV (processed_fleet_data.csv)
- **Deployment:** Streamlit Cloud (hosted at: https://assetvis-wuqn4tlse9cghjwuf84jdh.streamlit.app/)

---

## 7. Summary

### What the System Does:
1. **Ingests** 6 Excel files (105+ sheets) containing equipment, incidents, actions, and parts data
2. **Processes** and standardizes data, extracts components and failure modes, categorizes root causes
3. **Calculates** reliability metrics (MTBF, MTTR, failure rates), action effectiveness, and cost impact
4. **Outputs** a unified CSV with 187 component-level records and 58 calculated metrics
5. **Visualizes** data through a 3-level interactive dashboard for drill-down analysis

### Key Benefits:
- **Identifies Bad Actors:** Quickly find problematic components, failure modes, and spare parts
- **Root Cause Analysis:** Understand why failures occur (Human Error, Operational Error, Machinery Failure)
- **Action Effectiveness:** Measure if maintenance actions are working (recurrence rates, MTBF improvement)
- **Cost Visibility:** Track spare parts consumption and costs per component
- **Data-Driven Decisions:** Support maintenance strategy with quantitative metrics

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Total Lines:** ~500

