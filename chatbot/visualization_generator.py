"""
Visualization Generator
=======================
Auto-generates Plotly charts based on query type and result data.

Uses LLM to decide what visualization is appropriate and generates the code.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from llama_index.llms.nvidia import NVIDIA
import os
from dotenv import load_dotenv

load_dotenv()


class VisualizationGenerator:
    """
    Generates appropriate visualizations for chatbot responses
    """
    
    def __init__(self, llm=None):
        """Initialize with LLM for decision making"""
        if llm is None:
            api_key = os.getenv("NVIDIA_API_KEY")
            self.llm = NVIDIA(
                model="meta/llama-3.1-70b-instruct",
                api_key=api_key,
                temperature=0.1
            )
        else:
            self.llm = llm
    
    def should_visualize(self, question: str, answer: str) -> bool:
        """
        Determine if a visualization would be helpful
        
        Args:
            question: User's question
            answer: Generated answer text
        
        Returns:
            bool: True if visualization should be generated
        """
        # Keywords that typically benefit from visualization
        viz_keywords = [
            'top', 'list', 'compare', 'show', 'trend', 'distribution',
            'breakdown', 'ranking', 'vs', 'versus', 'chart', 'graph'
        ]
        
        question_lower = question.lower()
        
        # Check for visualization keywords
        for keyword in viz_keywords:
            if keyword in question_lower:
                return True
        
        # Check if answer contains numerical lists or comparisons
        if any(char.isdigit() for char in answer) and len(answer) > 50:
            return True
        
        return False
    
    def generate_visualization(self, question: str, answer: str, df: pd.DataFrame = None, query_type: str = "unknown"):
        """
        Generate appropriate visualization
        
        Args:
            question: User's question
            answer: Generated answer
            df: DataFrame with result data (if available)
            query_type: "metrics" or "knowledge"
        
        Returns:
            Plotly figure object or None
        """
        print(f"[VIZ DEBUG] Question: {question[:50]}...")
        print(f"[VIZ DEBUG] Should visualize: {self.should_visualize(question, answer)}")
        print(f"[VIZ DEBUG] DF is None: {df is None}")
        
        if not self.should_visualize(question, answer):
            print("[VIZ DEBUG] should_visualize returned False")
            return None
        
        if df is None or len(df) == 0:
            return None
        
        try:
            # Determine chart type based on question
            question_lower = question.lower()
            
            # Top N / Ranking queries → Bar chart
            if 'top' in question_lower or 'list' in question_lower or 'bad actor' in question_lower:
                return self._create_ranking_chart(df, question)
            
            # Comparison queries → Grouped bar chart
            elif 'compare' in question_lower or 'vs' in question_lower or 'versus' in question_lower:
                return self._create_comparison_chart(df, question)
            
            # Distribution queries → Pie chart
            elif 'distribution' in question_lower or 'breakdown' in question_lower:
                return self._create_distribution_chart(df, question)
            
            # MTBF or failure rate queries → Bar chart
            elif 'mtbf' in question_lower or 'failure rate' in question_lower:
                return self._create_metric_chart(df, question)
            
            # Parts/cost queries → Pie or bar
            elif 'parts' in question_lower or 'cost' in question_lower:
                return self._create_parts_chart(df, question)
            
            # Default: Bar chart if we have numeric data
            else:
                return self._create_default_chart(df)
        
        except Exception as e:
            print(f"[VIZ ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_ranking_chart(self, df, question):
        """Create bar chart for ranking/top N queries"""
        
        # Determine what to show
        if 'bad actor' in question.lower() and 'is_bad_actor' in df.columns:
            # Filter bad actors only if column exists
            plot_df = df[df['is_bad_actor'] == True].copy()
            if len(plot_df) == 0:
                plot_df = df.copy()
        else:
            plot_df = df.copy()
        
        # Limit to top 10
        if len(plot_df) > 10:
            plot_df = plot_df.nlargest(10, 'failure_rate_per_1000h')
        
        # Create label combining make_model and component
        if 'make_model' in plot_df.columns and 'component_id' in plot_df.columns:
            plot_df['label'] = plot_df['make_model'] + ' - ' + plot_df['component_id']
        elif 'component_id' in plot_df.columns:
            plot_df['label'] = plot_df['component_id']
        elif 'make_model' in plot_df.columns:
            plot_df['label'] = plot_df['make_model']
        else:
            return None
        
        # Create bar chart
        if 'failure_rate_per_1000h' in plot_df.columns:
            fig = px.bar(
                plot_df,
                x='failure_rate_per_1000h',
                y='label',
                orientation='h',
                title='Failure Rate per 1000 Hours',
                labels={'failure_rate_per_1000h': 'Failure Rate', 'label': ''},
                color='failure_rate_per_1000h',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400, showlegend=False)
            return fig
        
        return None
    
    def _create_comparison_chart(self, df, question):
        """Create grouped bar chart for comparisons"""
        
        if len(df) < 2:
            return None
        
        # Create comparison bar chart
        if 'mtbf_hours' in df.columns and 'component_id' in df.columns:
            fig = px.bar(
                df,
                x='component_id',
                y='mtbf_hours',
                title='MTBF Comparison',
                labels={'mtbf_hours': 'MTBF (hours)', 'component_id': 'Component'},
                color='mtbf_hours',
                color_continuous_scale='RdYlGn'
            )
            
            # Add fleet average line if available
            if 'fleet_avg_mtbf_hours' in df.columns:
                fleet_avg = df['fleet_avg_mtbf_hours'].iloc[0]
                if pd.notna(fleet_avg):
                    fig.add_hline(y=fleet_avg, line_dash="dash", 
                                  annotation_text="Fleet Average",
                                  line_color="blue")
            
            fig.update_layout(height=400)
            return fig
        
        return None
    
    def _create_distribution_chart(self, df, question):
        """Create pie chart for distributions"""
        
        # Root cause distribution
        if 'root_cause' in question.lower():
            root_cause_cols = [col for col in df.columns if 'root_cause_' in col and col != 'root_cause_percentage_confirmed']
            if root_cause_cols:
                values = [df[col].sum() for col in root_cause_cols]
                labels = [col.replace('root_cause_', '').replace('_', ' ').title() for col in root_cause_cols]
                
                fig = px.pie(
                    values=values,
                    names=labels,
                    title='Root Cause Distribution'
                )
                fig.update_layout(height=400)
                return fig
        
        return None
    
    def _create_metric_chart(self, df, question):
        """Create chart for MTBF or failure rate queries"""
        
        if len(df) == 0:
            return None
        
        # MTBF vs Fleet percentage chart
        if 'mtbf_vs_fleet' in df.columns and 'mtbf vs fleet' in question.lower():
            plot_df = df.copy()
            
            if 'make_model' in plot_df.columns and 'component_id' in plot_df.columns:
                plot_df['label'] = plot_df['component_id']
            elif 'component_id' in plot_df.columns:
                plot_df['label'] = plot_df['component_id']
            else:
                return None
            
            fig = px.bar(
                plot_df,
                x='mtbf_vs_fleet',
                y='label',
                orientation='h',
                title='MTBF vs Fleet Average (%)',
                labels={'mtbf_vs_fleet': 'MTBF vs Fleet (%)', 'label': 'Component'},
                color='mtbf_vs_fleet',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0
            )
            
            # Add zero line
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=400)
            return fig
        
        # Default MTBF bar chart
        elif 'mtbf_hours' in df.columns:
            return self._create_comparison_chart(df, question)
        
        return None
    
    def _create_parts_chart(self, df, question):
        """Create chart for parts/cost queries"""
        
        # If we have Part Name / Consumption columns (from spare parts query)
        if 'Part Name' in df.columns and 'Consumption' in df.columns:
            fig = px.bar(
                df,
                x='Consumption',
                y='Part Name',
                orientation='h',
                title='Top Spare Parts by Consumption',
                labels={'Consumption': 'Consumption Count', 'Part Name': ''},
                color='Consumption',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500, showlegend=False)
            return fig
        
        # If we have cost columns
        if 'total_cost_impact' in df.columns or 'estimated_cost_per_failure' in df.columns:
            # Create label
            if 'make_model' in df.columns and 'component_id' in df.columns:
                df = df.copy()
                df['label'] = df['make_model'] + ' - ' + df['component_id']
            elif 'component_id' in df.columns:
                df = df.copy()
                df['label'] = df['component_id']
            else:
                return None
            
            # Choose cost column
            cost_col = 'total_cost_impact' if 'total_cost_impact' in df.columns else 'estimated_cost_per_failure'
            
            fig = px.bar(
                df.nlargest(10, cost_col),
                x=cost_col,
                y='label',
                orientation='h',
                title=f'Top Failures by {"Total Cost Impact" if cost_col == "total_cost_impact" else "Cost per Failure"}',
                labels={cost_col: 'Cost ($)', 'label': ''},
                color=cost_col,
                color_continuous_scale='Oranges'
            )
            fig.update_layout(height=500, showlegend=False)
            return fig
        
        # If we have parts_detail_json
        if 'parts_detail_json' in df.columns:
            import json
            
            # Aggregate parts across components
            all_parts = []
            for parts_json in df['parts_detail_json'].dropna():
                try:
                    parts_list = json.loads(parts_json)
                    all_parts.extend(parts_list)
                except:
                    continue
            
            if all_parts:
                parts_df = pd.DataFrame(all_parts)
                parts_agg = parts_df.groupby('part_name').agg({
                    'count': 'sum',
                    'total_cost': 'sum'
                }).reset_index().nlargest(10, 'count')
                
                fig = px.pie(
                    parts_agg,
                    values='count',
                    names='part_name',
                    title='Top Parts by Consumption'
                )
                fig.update_layout(height=400)
                return fig
        
        return None
    
    def _create_default_chart(self, df):
        """Create default visualization if data is available"""
        
        if len(df) == 0:
            return None
        
        # Try to create a simple bar chart
        if 'failure_rate_per_1000h' in df.columns and 'component_id' in df.columns:
            return self._create_ranking_chart(df, "default")
        
        return None
