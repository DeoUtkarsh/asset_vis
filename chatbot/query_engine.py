"""
Query Engine Module
===================
Implements the Router that handles both metric and knowledge queries.

Uses:
- NVIDIA Nemotron LLM for generation
- Pandas Query Engine for metrics (Type A)
- Vector Query Engine for knowledge (Type B)
- Router to decide which engine to use
"""

import pandas as pd
import chromadb
import os
import ast
import json
from pathlib import Path
from dotenv import load_dotenv

from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# Load environment
load_dotenv()

# Configuration
CSV_FILE = "../processed_fleet_data.csv"
CHROMA_DB_PATH = "./chroma_db"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")


class AssetQueryEngine:
    """
    Main query engine that routes questions to appropriate sub-engines
    """
    
    def __init__(self, test_mode=False):
        """Initialize the query engine with LLM and sub-engines. test_mode=True skips Vector/ChromDB for validation."""
        
        if not test_mode and not NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY not found! Set it in .env file")
        
        print("Initializing Asset Query Engine...")
        
        # Initialize LLM (required for pandas engine even in test mode)
        if not NVIDIA_API_KEY:
            self.llm = None
        else:
            self.llm = NVIDIA(
                model="meta/llama-3.1-70b-instruct",
                api_key=NVIDIA_API_KEY,
                temperature=0.1,  # Low temperature for more consistent answers
                max_tokens=1024
            )
        
        # Load CSV for metrics
        self.df = self._load_csv()
        
        # Build engines
        self.pandas_engine = self._build_pandas_engine()
        if test_mode:
            self.embed_model = None
            self.vector_engine = None
            self.router_engine = None
            print("Query engine ready (test mode - CSV only)")
        else:
            self.embed_model = NVIDIAEmbedding(
                model="nvidia/nv-embedqa-e5-v5",
                api_key=NVIDIA_API_KEY,
                truncate="END"
            )
            self.vector_engine = self._build_vector_engine()
            self.router_engine = self._build_router()
            print("Query engine ready!")
    
    def _load_csv(self):
        """Load processed fleet data CSV"""
        csv_path = Path(CSV_FILE)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path.absolute()}")
        
        df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def _build_pandas_engine(self):
        """
        Build simple CSV query engine for metrics queries (Type A)
        Uses LLM to interpret question and format response from DataFrame
        """
        print("   🔢 Building CSV metrics engine...")
        
        # Create a simple wrapper that uses LLM to query CSV
        class SimpleCSVQueryEngine:
            """Simple engine that queries CSV data using LLM"""
            
            def __init__(self, df, llm):
                self._df = df
                self._llm = llm
                self._result_df = None  # Store filtered DataFrame for visualization
            
            def query(self, query_bundle):
                """Query the CSV data"""
                # Extract query string from QueryBundle if needed
                if hasattr(query_bundle, 'query_str'):
                    query_str = query_bundle.query_str
                else:
                    query_str = str(query_bundle)
                
                # Reset result DataFrame
                self._result_df = None
                
                # Create a focused context based on query type
                context = self._create_context(query_str)
                
                # Create improved prompt
                prompt = f"""You are analyzing a fleet failure analytics dataset.

CRITICAL DISTINCTIONS:
- COMPONENTS are physical parts: "Boiler", "Engine (General)", "Turbocharger", "Fuel System", etc.
- FAILURE MODES are HOW components fail: "HIGH TEMPERATURE", "LEAK", "STOPPED", "CORROSION", "LOW PRESSURE", etc.

When asked about "failure modes", ONLY mention failure modes (not components).
When asked about "components" or "bad actors", ONLY mention components (not failure modes).

Data:
{context}

User Question: {query_str}

Provide a clear, concise answer based ONLY on the data above. If data is not available, say so clearly.
Format numbers nicely and be specific."""
                
                # Get response from LLM
                from llama_index.core.base.response.schema import Response
                response_text = self._llm.complete(prompt)
                return Response(response=str(response_text))

            
            def _match_make_model(self, query_lower):
                """Match make/model with exact and partial matching (e.g. 'Daihatsu' -> DAIHATSU DIESEL MFG...)"""
                for mm in self._df['make_model'].unique():
                    mm_lower = str(mm).lower()
                    if mm_lower in query_lower:
                        return mm
                # Partial: query terms in make_model (e.g. "daihatsu" -> "ANQING DAIHATSU 6DK-20")
                for mm in self._df['make_model'].unique():
                    mm_lower = str(mm).lower()
                    words = [w for w in query_lower.replace(',', ' ').split() if len(w) > 2]
                    for w in words:
                        if w in mm_lower and len(w) > 3:
                            return mm
                return None
            
            def _get_equipment_type_filter(self, query_lower):
                """Detect Main Engine vs Auxiliary Engine from query."""
                if 'main engine' in query_lower or 'main engines' in query_lower:
                    return 'Main Engine'
                if 'auxiliary engine' in query_lower or 'aux engine' in query_lower or 'aux engines' in query_lower:
                    return 'Aux Engine'
                return None
            
            def _create_context(self, query_str):
                """Create relevant context based on query - full UI parity (Level 1, 2, 3)."""
                query_lower = str(query_str).lower()
                
                # Match make/model (exact + partial)
                matched_make_model = self._match_make_model(query_lower)
                
                # Match component (for Level 3)
                matched_component = None
                for comp in self._df['component_id'].unique():
                    if comp and str(comp).lower() in query_lower:
                        matched_component = str(comp)
                        break
                
                # When component is specified, prefer make_model that has that component (e.g. Turbocharger on Daihatsu 6DK-20E)
                if matched_component and matched_make_model:
                    rows_with_comp = self._df[self._df['component_id'] == matched_component]
                    query_words = [w for w in query_lower.replace(',', ' ').split() if len(w) > 3]
                    for _, row in rows_with_comp.iterrows():
                        mm = row['make_model']
                        mm_lower = str(mm).lower()
                        if any(w in mm_lower for w in query_words):
                            matched_make_model = mm
                            break
                
                # Equipment type filter (Main Engine vs Aux Engine)
                equipment_filter = self._get_equipment_type_filter(query_lower)
                
                # ----- FAILURE PROBABILITY (30/60/90 days) -----
                if any(k in query_lower for k in ['probability of failure', 'highest probability', '30 days', '60 days', '90 days', 'next 30', 'next 60', 'next 90']):
                    df_subset = self._df.copy()
                    if equipment_filter:
                        df_subset = df_subset[df_subset['equipment_type'] == equipment_filter]
                    df_subset = df_subset[df_subset['failure_rate_per_1000h'].notna()]
                    df_subset = df_subset.nlargest(15, 'failure_rate_per_1000h')
                    
                    # Store for visualization
                    self._result_df = df_subset[['make_model', 'component_id', 'failure_rate_per_1000h', 'equipment_type']].copy()
                    self._result_df['prob_30d'] = self._result_df['failure_rate_per_1000h'] * 720 / 1000
                    self._result_df['prob_60d'] = self._result_df['failure_rate_per_1000h'] * 1440 / 1000
                    self._result_df['prob_90d'] = self._result_df['failure_rate_per_1000h'] * 2160 / 1000
                    hrs_30, hrs_60, hrs_90 = 720, 1440, 2160
                    lines = [f"Probability of failure = (Failure rate per 1000h) x (Hours) / 1000",
                             f"Hours: 30d={hrs_30}, 60d={hrs_60}, 90d={hrs_90}"]
                    if equipment_filter:
                        lines.append(f"Filtered by equipment type: {equipment_filter}")
                    for _, r in df_subset.iterrows():
                        rate = r.get('failure_rate_per_1000h', 0) or 0
                        p30 = rate * hrs_30 / 1000
                        p60 = rate * hrs_60 / 1000
                        p90 = rate * hrs_90 / 1000
                        mm = r.get('make_model', '')
                        comp = r.get('component_id', '')
                        lines.append(f"{mm} {comp}: 30d={p30:.2%} | 60d={p60:.2%} | 90d={p90:.2%} (rate={rate:.2f}/1000h)")
                    return "\n".join(lines)
                
                # ----- HIGHEST COST FAILURES -----
                if any(k in query_lower for k in ['highest cost', 'highest-cost', 'costliest', 'most expensive', 'cost repairs', 'cost downtime', 'cost spares']):
                    df_subset = self._df.copy()
                    if equipment_filter:
                        df_subset = df_subset[df_subset['equipment_type'] == equipment_filter]
                    df_subset = df_subset[df_subset['total_cost_impact'].notna()]
                    df_subset = df_subset[df_subset['total_cost_impact'] > 0]
                    df_subset = df_subset.nlargest(15, 'total_cost_impact')
                    
                    # Store for visualization
                    self._result_df = df_subset[['make_model', 'component_id', 'total_cost_impact', 'estimated_cost_per_failure', 'failure_count', 'equipment_type']].copy()
                    cols = [c for c in ['make_model', 'component_id', 'total_cost_impact', 'estimated_cost_per_failure', 'failure_count', 'equipment_type'] if c in df_subset.columns]
                    return f"""Highest Cost Failures (by total cost impact):
{df_subset[cols].to_string(index=False)}"""
                
                # ----- 14-DAY RISK HEURISTIC -----
                if any(k in query_lower for k in ['14 days', 'next 14 days', 'operational risk', 'highest risk']) and 'maintenance' in query_lower:
                    df_subset = self._df.copy()
                    df_subset['cost'] = df_subset['total_cost_impact'].fillna(0)
                    df_subset['rate'] = df_subset['failure_rate_per_1000h'].fillna(0)
                    df_subset['risk_proxy'] = df_subset['rate'] * (1 + df_subset['cost'] / 1000)
                    df_subset = df_subset.nlargest(15, 'risk_proxy')
                    cols = [c for c in ['make_model', 'component_id', 'failure_rate_per_1000h', 'total_cost_impact', 'equipment_type'] if c in df_subset.columns]
                    return f"""Highest Operational Risk (heuristic: failure rate x cost impact, next 14 days):
{df_subset[cols].to_string(index=False)}"""
                
                # ----- TOP BAD ACTORS BY EQUIPMENT TYPE (Main Engine / Aux Engine) -----
                # Note: Q9 multi-part (root causes + actions + parts) takes precedence - checked above
                if equipment_filter and ('bad actor' in query_lower or ('top' in query_lower and 'component' in query_lower)) and 'root cause' not in query_lower and 'corrective action' not in query_lower:
                    df_subset = self._df[self._df['equipment_type'] == equipment_filter].copy()
                    df_subset = df_subset.nlargest(10, 'failure_rate_per_1000h')
                    
                    # Store for visualization
                    cols = [c for c in ['make_model', 'component_id', 'failure_rate_per_1000h', 'mtbf_hours', 'failure_count', 'equipment_type'] if c in df_subset.columns]
                    self._result_df = df_subset[cols].copy()
                    cols = [c for c in ['make_model', 'component_id', 'failure_rate_per_1000h', 'mtbf_hours', 'failure_count', 'root_cause_human_error', 'root_cause_operational_error', 'root_cause_machinery_failure', 'root_cause_other'] if c in df_subset.columns]
                    root_cols = [c for c in cols if c.startswith('root_cause')]
                    lines = [f"Top 10 Bad Actor Components for {equipment_filter}:", df_subset[cols].to_string(index=False)]
                    if root_cols:
                        lines.append("\nRoot cause breakdown (counts): human_error, operational_error, machinery_failure, other")
                    return "\n".join(lines)
                
                # ----- TOP SPARE PARTS BY EQUIPMENT TYPE (Main Engine / Aux Engine) -----
                if equipment_filter and ('spare parts' in query_lower or 'parts consumed' in query_lower or ('parts' in query_lower and 'top' in query_lower)):
                    df_subset = self._df[self._df['equipment_type'] == equipment_filter]
                    parts_agg = {}
                    for _, row in df_subset.iterrows():
                        pdj = row.get('parts_detail_json')
                        if pdj and pd.notna(pdj):
                            try:
                                plist = json.loads(pdj) if isinstance(pdj, str) else pdj
                                for p in plist:
                                    name = p.get('part_name', '')
                                    cnt = p.get('count', 0) or 0
                                    if name:
                                        parts_agg[name] = parts_agg.get(name, 0) + cnt
                            except Exception:
                                pass
                    sorted_parts = sorted(parts_agg.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    # Store for visualization
                    if sorted_parts:
                        self._result_df = pd.DataFrame(sorted_parts, columns=['Part Name', 'Consumption'])
                    
                    lines = [f"Top 10 Spare Parts consumed for {equipment_filter}:"]
                    for name, cnt in sorted_parts:
                        lines.append(f"  {name}: {cnt}")
                    return "\n".join(lines)
                
                # ----- TOP BAD ACTORS FOR MAIN ENGINE + ROOT CAUSES + CORRECTIVE ACTIONS + SPARE PARTS (Q9 multi-part) -----
                if equipment_filter and ('root cause' in query_lower or 'corrective action' in query_lower) and ('top' in query_lower or 'bad actor' in query_lower):
                    df_subset = self._df[self._df['equipment_type'] == equipment_filter].copy()
                    df_subset = df_subset.nlargest(10, 'failure_rate_per_1000h')
                    lines = [f"Top 10 Bad Actor Components for {equipment_filter} with root causes, corrective actions, and spare parts:"]
                    for _, r in df_subset.iterrows():
                        mm = r.get('make_model', '')
                        comp = r.get('component_id', '')
                        rc = f"Root causes: Human={r.get('root_cause_human_error',0)} Operational={r.get('root_cause_operational_error',0)} Machinery={r.get('root_cause_machinery_failure',0)} Other={r.get('root_cause_other',0)}"
                        actions = f"Actions: Inspect={r.get('action_inspect_count',0)} Repair={r.get('action_repair_count',0)} Replace={r.get('action_replace_count',0)}"
                        pdj = r.get('parts_detail_json')
                        top_parts = []
                        if pdj and pd.notna(pdj):
                            try:
                                plist = json.loads(pdj) if isinstance(pdj, str) else pdj
                                top_parts = [f"{p.get('part_name','')}({p.get('count',0)})" for p in plist[:5]]
                            except Exception:
                                pass
                        parts_str = " | ".join(top_parts) if top_parts else "N/A"
                        lines.append(f"\n{mm} - {comp}: {rc} | {actions} | Top parts: {parts_str}")
                    return "\n".join(lines)
                
                # ----- LEVEL 3: Component deep-dive (make/model + component) -----
                if matched_make_model and matched_component:
                    row = self._df[(self._df['make_model'] == matched_make_model) & (self._df['component_id'] == matched_component)]
                    if len(row) > 0:
                        r = row.iloc[0]
                        lines = [f"Component: {matched_component} for {matched_make_model}",
                                 f"Failure count: {r.get('failure_count', 'N/A')} | MTBF (hours): {r.get('mtbf_hours', 'N/A')} | Failure rate/1000h: {r.get('failure_rate_per_1000h', 'N/A')}",
                                 f"MTBF vs Fleet: {r.get('mtbf_vs_fleet', 'N/A')}% | Vessels affected: {r.get('vessels_affected', 'N/A')}"]
                        # Root causes
                        rc = [f"  Human error: {r.get('root_cause_human_error', 0)}", f"  Operational: {r.get('root_cause_operational_error', 0)}",
                              f"  Machinery: {r.get('root_cause_machinery_failure', 0)}", f"  Other: {r.get('root_cause_other', 0)}", f"  Unknown: {r.get('root_cause_unknown', 0)}"]
                        lines.append("Root causes: " + " | ".join(rc))
                        # Failure mode
                        lines.append(f"Top failure mode: {r.get('top_failure_mode', 'N/A')} (count: {r.get('top_failure_mode_count', 'N/A')})")
                        # Parts
                        pdj = r.get('parts_detail_json')
                        if pdj and pd.notna(pdj) and str(pdj) != '[]':
                            try:
                                parts_list = json.loads(pdj) if isinstance(pdj, str) else pdj
                                parts_list = parts_list[:10]
                                lines.append("Spare parts (top): " + ", ".join([f"{p.get('part_name', '')}({p.get('count', 0)})" for p in parts_list]))
                            except Exception:
                                pass
                        lines.append(f"Estimated cost per failure: {r.get('estimated_cost_per_failure', 'N/A')} | Total cost impact: {r.get('total_cost_impact', 'N/A')} | Stock risk: {r.get('stock_risk', 'N/A')}")
                        return "\n".join(lines)
                
                # ----- LEVEL 2: Top 10 Failure Modes for make/model -----
                if 'failure mode' in query_lower or 'failure modes' in query_lower:
                    # Check if fleet-wide query
                    is_fleet_wide = any(keyword in query_lower for keyword in ['across all', 'fleet', 'all make', 'all model', 'entire fleet', 'fleet-wide'])
                    
                    # Check if asking for specific component across fleet
                    component_filter = None
                    component_keywords = {
                        'engine (general)': 'Engine',
                        'engine(general)': 'Engine',
                        'turbocharger': 'Turbo',
                        'boiler': 'Boiler',
                        'fuel system': 'Fuel',
                        'starting air': 'Starting',
                        'lubrication': 'Lubrication',
                        'cooling': 'Cooling'
                    }
                    
                    for keyword, filter_term in component_keywords.items():
                        if keyword in query_lower:
                            component_filter = filter_term
                            break
                    
                    if is_fleet_wide or component_filter:
                        # Aggregate failure modes across entire fleet (or filtered by component)
                        fm_dict = {}
                        filtered_df = self._df
                        
                        # Filter by component if specified
                        if component_filter:
                            filtered_df = self._df[self._df['component_id'].str.contains(component_filter, case=False, na=False)]
                        
                        for _, row in filtered_df.iterrows():
                            top_fm = row.get('top_failure_mode')
                            fm_count = row.get('top_failure_mode_count', 0)
                            fm_rate = row.get('top_failure_mode_rate', 0)
                            
                            if top_fm and str(top_fm) not in ('Unknown', 'nan') and pd.notna(top_fm):
                                if top_fm not in fm_dict:
                                    fm_dict[top_fm] = {'count': 0, 'rates': []}
                                fm_dict[top_fm]['count'] += fm_count if pd.notna(fm_count) else 0
                                if pd.notna(fm_rate) and fm_rate < 1000:  # Filter extreme rates
                                    fm_dict[top_fm]['rates'].append(fm_rate)
                        
                        # Sort by count
                        sorted_modes = sorted(fm_dict.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
                        
                        header = f"Fleet-wide Failure Modes for {component_filter} (Top 10):" if component_filter else "Fleet-wide Failure Modes (Top 10 by occurrence):"
                        lines = [header]
                        lines.append(f"Total components analyzed: {len(filtered_df)}")
                        lines.append("")
                        for mode, data in sorted_modes:
                            avg_rate = sum(data['rates']) / len(data['rates']) if data['rates'] else 0
                            lines.append(f"  {mode}: {data['count']} occurrences, avg rate {avg_rate:.2f} per 1000h")
                        
                        return "\n".join(lines)
                    
                    elif matched_make_model:
                        # Failure modes for specific make/model
                        filtered = self._df[self._df['make_model'] == matched_make_model]
                        fm_list = []
                        for _, row in filtered.iterrows():
                            top_fm = row.get('top_failure_mode')
                            if top_fm and str(top_fm) not in ('Unknown', 'Other', 'nan'):
                                fm_list.append({'Failure Mode': top_fm, 'Count': row.get('top_failure_mode_count', 0),
                                               'Rate': row.get('top_failure_mode_rate', 0) or row.get('failure_rate_per_1000h', 0),
                                               'Component': row.get('component_id', '')})
                        if not fm_list:
                            for _, row in filtered.iterrows():
                                fm_list.append({'Failure Mode': str(row.get('component_id', '')) + ' - High Failure Rate',
                                               'Count': row.get('failure_count', 0), 'Rate': row.get('failure_rate_per_1000h', 0), 'Component': row.get('component_id', '')})
                        if fm_list:
                            fm_df = pd.DataFrame(fm_list).groupby('Failure Mode').agg({'Count': 'sum', 'Rate': 'mean'}).reset_index()
                            fm_df = fm_df.sort_values('Rate', ascending=False).head(10)
                            return f"""Top 10 Failure Modes for {matched_make_model}:
{fm_df.to_string(index=False)}"""

                
                # ----- LEVEL 2: Top 10 Spare Parts for make/model -----
                if matched_make_model and ('spare parts' in query_lower or 'parts consumption' in query_lower or ('parts' in query_lower and 'top' in query_lower)):
                    filtered = self._df[self._df['make_model'] == matched_make_model]
                    parts_data = []
                    for _, row in filtered.iterrows():
                        top_parts_str = row.get('top_parts')
                        top_consumption_str = row.get('top_parts_consumption')
                        if not top_parts_str or pd.isna(top_parts_str): continue
                        try:
                            top_parts = ast.literal_eval(top_parts_str) if isinstance(top_parts_str, str) else top_parts_str
                            top_consumption = ast.literal_eval(top_consumption_str) if top_consumption_str and pd.notna(top_consumption_str) else [0] * len(top_parts)
                            for i, name in enumerate(top_parts):
                                consumption = top_consumption[i] if i < len(top_consumption) else 0
                                parts_data.append({'Part Name': name, 'Consumption': int(consumption)})
                        except Exception:
                            pass
                    if parts_data:
                        parts_agg = pd.DataFrame(parts_data).groupby('Part Name').agg({'Consumption': 'sum'}).reset_index().sort_values('Consumption', ascending=False).head(10)
                        return f"""Top 10 Spare Parts by consumption for {matched_make_model}:
{parts_agg.to_string(index=False)}"""
                
                # ----- Top 10 Bad Actor Components FOR [make/model] -----
                if matched_make_model and ('top' in query_lower or 'bad actor' in query_lower or 'list' in query_lower or 'component' in query_lower):
                    df_subset = self._df[self._df['make_model'] == matched_make_model].copy()
                    df_subset = df_subset.nlargest(10, 'failure_rate_per_1000h')
                    
                    # Store for visualization
                    cols = [c for c in ['component_id', 'failure_rate_per_1000h', 'mtbf_hours', 'failure_count', 'mtbf_vs_fleet', 'bad_actor_rank'] if c in df_subset.columns]
                    self._result_df = df_subset[cols].copy()
                    cols = [c for c in ['component_id', 'failure_rate_per_1000h', 'mtbf_hours', 'failure_count', 'mtbf_vs_fleet', 'bad_actor_rank'] if c in df_subset.columns]
                    df_display = df_subset[cols]
                    return f"""Top 10 Bad Actor Components for {matched_make_model} (ranked by failure rate):
{df_display.to_string(index=False)}

These are the components within this Make/Model, ranked by failure rate per 1000h (highest = worst performer)."""
                
                # ----- LEVEL 1: Summary for make/model (MTBF, total failures, vessels, etc.) -----
                if matched_make_model and not ('top' in query_lower or 'bad actor' in query_lower or 'failure mode' in query_lower or 'spare parts' in query_lower):
                    row = self._df[self._df['make_model'] == matched_make_model].iloc[0]
                    return f"""Summary for {matched_make_model}:
- Total Failures: {row.get('total_failures', 'N/A')}
- Total Vessels: {row.get('total_vessels', 'N/A')}
- Overall MTBF (hours): {row.get('overall_mtbf_hours', 'N/A')}
- Overall Failure Rate per 1000h: {row.get('overall_failure_rate_per_1000h', 'N/A')}
- Number of Components tracked: {row.get('component_count', 'N/A')}
- Equipment Type: {row.get('equipment_type_summary', 'N/A')}"""
                
                # ----- Generic top N / bad actors (whole fleet, optionally by equipment type) -----
                if 'top' in query_lower or 'list' in query_lower or 'bad actor' in query_lower:
                    if 'bad actor' in query_lower:
                        df_subset = self._df[self._df['is_bad_actor'] == True].copy()
                    else:
                        df_subset = self._df.copy()
                    if equipment_filter:
                        df_subset = df_subset[df_subset['equipment_type'] == equipment_filter]
                    df_subset = df_subset.nlargest(15, 'failure_rate_per_1000h')
                    
                    # Store for visualization
                    cols = [c for c in ['make_model', 'component_id', 'failure_rate_per_1000h', 'mtbf_hours', 'failure_count', 'mtbf_vs_fleet', 'equipment_type'] if c in df_subset.columns]
                    self._result_df = df_subset[cols].copy()
                    cols = [c for c in ['make_model', 'component_id', 'failure_rate_per_1000h', 'mtbf_hours', 'failure_count', 'mtbf_vs_fleet', 'equipment_type'] if c in df_subset.columns]
                    df_display = df_subset[cols]
                    scope = f"for {equipment_filter}" if equipment_filter else "across entire fleet"
                    return f"""Top components by failure rate ({scope}):
{df_display.to_string(index=False)}"""
                
                # ----- Make/model component list (no ranking) -----
                if matched_make_model:
                    df_subset = self._df[self._df['make_model'] == matched_make_model]
                    cols = [c for c in ['component_id', 'failure_rate_per_1000h', 'mtbf_hours', 'failure_count', 'is_bad_actor'] if c in df_subset.columns]
                    return f"""Data for {matched_make_model}:
{df_subset[cols].to_string(index=False)}

Summary: Total failures {df_subset['failure_count'].sum()} | Components: {len(df_subset)} | Bad actors: {df_subset['is_bad_actor'].sum()}"""
                
                # Default
                return f"""Dataset Overview:
- Total records: {len(self._df)} | Unique make/models: {self._df['make_model'].nunique()}
- Equipment types: {', '.join(self._df['equipment_type'].unique())}

Top 10 by failure rate:
{self._df.nlargest(10, 'failure_rate_per_1000h')[['make_model', 'component_id', 'failure_rate_per_1000h']].to_string(index=False)}"""
        
        engine = SimpleCSVQueryEngine(df=self.df, llm=self.llm)
        return engine
    
    def _build_vector_engine(self):
        """
        Build Vector Query Engine for knowledge queries (Type B)
        """
        print("   📚 Building Vector engine...")
        
        chroma_path = Path(CHROMA_DB_PATH)
        if not chroma_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {chroma_path.absolute()}! "
                f"Run rag_ingest.py first."
            )
        
        # Load ChromaDB
        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        chroma_collection = db.get_or_create_collection("asset_knowledge")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create index from existing store
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self.embed_model
        )
        
        # Create query engine with custom prompt
        engine = index.as_query_engine(
            llm=self.llm,
            similarity_top_k=5,  # Retrieve top 5 most relevant chunks
            response_mode="compact",  # Compact response synthesis
        )
        
        return engine
    
    def _build_router(self):
        """
        Build Router Query Engine that selects between Pandas and Vector
        """
        print("   🧠 Building Router...")
        
        # Define tools for the router
        query_engine_tools = [
            QueryEngineTool(
                query_engine=self.pandas_engine,
                metadata=ToolMetadata(
                    name="fleet_metrics_database",
                    description=(
                        "Use this for questions about NUMBERS, STATISTICS, and CALCULATIONS. "
                        "Good for: "
                        "- 'Probability of failure in 30/60/90 days' "
                        "- 'Top bad actors for main engines / auxiliary engines' "
                        "- 'Highest cost failures' "
                        "- 'Highest operational risk in 14 days' "
                        "- 'Top spare parts for main/auxiliary engines' "
                        "- 'List top N bad actors' "
                        "- 'What is the MTBF for [Make/Model]?' "
                        "- 'Which component has highest failure rate?' "
                        "- 'How many failures for [component]?' "
                        "- Any question asking for counts, averages, comparisons, rankings, costs."
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=self.vector_engine,
                metadata=ToolMetadata(
                    name="technical_knowledge_base",
                    description=(
                        "Use this for questions about PROCEDURES, TROUBLESHOOTING, and HISTORY. "
                        "Good for: "
                        "- 'How do I fix [problem]?' "
                        "- 'What causes [failure mode]?' "
                        "- 'What are symptoms of [issue]?' "
                        "- 'What actions were taken for [incident]?' "
                        "- 'Recommended checks for [component]' "
                        "- 'Historical incidents involving [vessel/component]' "
                        "- Any question asking for explanations, procedures, or past examples."
                    ),
                ),
            ),
        ]
        
        # Create router with LLM selector
        router = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(llm=self.llm),
            query_engine_tools=query_engine_tools,
            verbose=True,
            llm=self.llm
        )
        
        return router
    
    def query(self, question: str):
        """
        Main query method
        
        Args:
            question: User's natural language question
        
        Returns:
            tuple: (answer_text, query_type, result_data)
        """
        try:
            print(f"\n🔍 Query: {question}")
            
            # Route and execute query
            response = self.router_engine.query(question)
            
            # Extract result
            answer_text = str(response)
            
            # Try to detect which engine was used and get result DataFrame
            query_type = "unknown"
            result_data = None
            
            # Check if CSV engine was used (it returns our SimpleCSVQueryEngine response)
            if hasattr(self.pandas_engine, '_result_df') and self.pandas_engine._result_df is not None:
                query_type = "metrics"
                result_data = self.pandas_engine._result_df
            elif 'pandas' in str(type(response)).lower() or hasattr(response, 'metadata'):
                query_type = "metrics"
                # Fallback: return None so no viz is generated
                result_data = None
            else:
                query_type = "knowledge"
            
            return answer_text, query_type, result_data
        
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(f"❌ Error: {e}")
            return error_msg, "error", None
    
    def query_metrics_direct(self, make_model=None, component_id=None):
        """
        Direct metrics query (bypass router)
        Useful for pre-formatted queries from UI buttons
        """
        filtered = self.df.copy()
        
        if make_model:
            filtered = filtered[filtered['make_model'] == make_model]
        
        if component_id:
            filtered = filtered[filtered['component_id'] == component_id]
        
        return filtered
    
    def get_csv_context(self, question: str) -> str:
        """Get raw CSV context for a question (for testing). Does not call LLM."""
        return self.pandas_engine._create_context(question)


# Singleton instance
_engine = None

def get_query_engine():
    """Get or create query engine singleton"""
    global _engine
    if _engine is None:
        _engine = AssetQueryEngine()
    return _engine
