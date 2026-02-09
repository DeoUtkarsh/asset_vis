"""
Asset Failure Analytics Chatbot
================================
Interactive AI assistant for fleet maintenance insights.

Run with: streamlit run chatbot_app.py
"""

import streamlit as st
from pathlib import Path
import sys
import uuid
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from query_engine import get_query_engine
from visualization_generator import VisualizationGenerator


# Page configuration - sidebar expanded for ChatGPT-like layout
st.set_page_config(
    page_title="Asset Analytics Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ChatGPT-like layout: FIXED sidebar, scrollable main content
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    body {
        margin: 0;
        padding: 0;
    }
    
    /* Sidebar: FIXED - does NOT scroll with page */
    [data-testid="stSidebar"] {
        position: fixed !important;
        left: 0 !important;
        top: 0 !important;
        height: 100vh !important;
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
        background-color: #171717 !important;
        overflow-y: auto !important;
        z-index: 999;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 1rem 0.75rem;
        height: 100%;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ECECEC;
    }
    
    [data-testid="stSidebar"] [data-testid="stDecoration"] {
        display: none;
    }
    
    /* Main content: scrolls when it grows */
    section.main {
        margin-left: 280px !important;
        padding-left: 0 !important;
        padding-bottom: 8rem !important;
        min-height: 100vh !important;
    }
    
    .main .block-container {
        padding: 1.5rem 2rem 2rem 2rem;
        max-width: 900px;
        margin-left: auto !important;
        margin-right: auto;
        min-height: 100vh !important;
    }
    
    /* Root: allow page scroll */
    .stApp, [data-testid="stAppViewContainer"] {
        overflow: visible !important;
    }
    
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Header - offset for fixed sidebar, subtle border */
    [data-testid="stHeader"] {
        margin-left: 280px !important;
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #E5E5E5;
    }
    
    /* User message bubble */
    .user-message {
        background-color: #007AFF;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        max-width: 70%;
        align-self: flex-end;
        margin-left: auto;
        word-wrap: break-word;
    }
    
    /* Assistant message bubble */
    .assistant-message {
        background-color: #F4F4F4;
        color: #212529;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        max-width: 70%;
        align-self: flex-start;
        word-wrap: break-word;
        border: 1px solid #E5E5E5;
    }
    
    .message-meta {
        font-size: 0.75rem;
        opacity: 0.6;
        margin-top: 0.25rem;
    }
    
    /* Chat input - fixed at bottom of viewport */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 280px !important;
        right: 0 !important;
        z-index: 100 !important;
        background: #FFFFFF !important;
        border-top: 1px solid #E5E5E5;
        padding: 1rem 2rem !important;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05) !important;
    }
    
    
    [data-testid="stChatInput"] textarea {
        border: 1px solid #E5E5E5 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.9375rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    }
    
    [data-testid="stChatInput"] textarea:focus {
        border-color: #007AFF !important;
        box-shadow: 0 0 0 2px rgba(0,122,255,0.15) !important;
    }
    
    /* Sidebar chat list - list-item style */
    [data-testid="stSidebar"] button {
        justify-content: flex-start;
        text-align: left;
        border-radius: 8px;
    }
    
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
    }
    
    /* Ensure chat message content is fully visible */
    [data-testid="stChatMessage"] {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session retention: 30 minutes
SESSION_RETENTION_MINUTES = 30


def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}
    
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    
    if 'viz_generator' not in st.session_state:
        st.session_state.viz_generator = None
    
    if 'engine_loaded' not in st.session_state:
        st.session_state.engine_loaded = False
    
    # Ensure we have at least one session
    if st.session_state.current_session_id is None or st.session_state.current_session_id not in st.session_state.chat_sessions:
        _create_new_session()


def _create_new_session():
    """Create a new chat session and set it as current"""
    now = datetime.now()
    session_id = str(uuid.uuid4())
    st.session_state.chat_sessions[session_id] = {
        "messages": [],
        "created_at": now,
        "title": "New chat"
    }
    st.session_state.current_session_id = session_id
    return session_id


def _get_current_messages():
    """Get messages for current session"""
    sid = st.session_state.current_session_id
    if sid and sid in st.session_state.chat_sessions:
        return st.session_state.chat_sessions[sid]["messages"]
    return []


def _set_current_messages(messages):
    """Set messages for current session"""
    sid = st.session_state.current_session_id
    if sid and sid in st.session_state.chat_sessions:
        st.session_state.chat_sessions[sid]["messages"] = messages


def _cleanup_old_sessions():
    """Remove sessions older than SESSION_RETENTION_MINUTES"""
    now = datetime.now()
    cutoff = now - timedelta(minutes=SESSION_RETENTION_MINUTES)
    to_remove = [
        sid for sid, data in st.session_state.chat_sessions.items()
        if data["created_at"] < cutoff
    ]
    for sid in to_remove:
        del st.session_state.chat_sessions[sid]
    # If current was removed, create new one
    if st.session_state.current_session_id not in st.session_state.chat_sessions:
        _create_new_session()


def _update_session_title(session_id, first_query: str):
    """Set session title from first user message"""
    if session_id in st.session_state.chat_sessions:
        title = (first_query[:50] + "…") if len(first_query) > 50 else first_query
        st.session_state.chat_sessions[session_id]["title"] = title


def load_engines():
    """Load query engine and visualization generator (cached)"""
    if not st.session_state.engine_loaded:
        with st.spinner("Initializing AI engines..."):
            try:
                st.session_state.query_engine = get_query_engine()
                st.session_state.viz_generator = VisualizationGenerator(
                    llm=st.session_state.query_engine.llm
                )
                st.session_state.engine_loaded = True
                return True
            except FileNotFoundError as e:
                st.error(f" {str(e)}")
                st.info("Run `python rag_ingest.py` first to build the vector store.")
                return False
            except Exception as e:
                st.error(f"Failed to load engines: {e}")
                return False
    return True


def handle_query(question: str):
    """Handle user query and generate response"""
    messages = _get_current_messages()
    
    # Add user message
    messages.append({"role": "user", "content": question})
    
    # Update session title from first message
    if len(messages) == 1:
        _update_session_title(st.session_state.current_session_id, question)
    
    # Display user message
    with st.chat_message("user"):
        st.write(question)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, query_type, result_df = st.session_state.query_engine.query(question)
                st.write(answer)
                
                # Debug logging for visualization
                print(f"\n[DEBUG] Query type: {query_type}")
                print(f"[DEBUG] Result DF: {type(result_df)}")
                if result_df is not None:
                    print(f"[DEBUG] DF shape: {result_df.shape}")
                    print(f"[DEBUG] DF columns: {list(result_df.columns)}")
                else:
                    print("[DEBUG] No DataFrame returned - visualization will be skipped")
                
                viz_fig = None
                if result_df is not None:
                    print("[DEBUG] Calling visualization generator...")
                    viz_fig = st.session_state.viz_generator.generate_visualization(
                        question, answer, result_df, query_type
                    )
                    if viz_fig is not None:
                        print("[DEBUG] Visualization generated successfully")
                        st.plotly_chart(viz_fig, use_container_width=True)
                    else:
                        print("[DEBUG] Visualization generator returned None")
                
                messages.append({"role": "assistant", "content": answer, "viz_fig": viz_fig})
            
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                print(f"[ERROR] {e}")
                import traceback
                traceback.print_exc()
                messages.append({"role": "assistant", "content": error_msg})
    
    _set_current_messages(messages)


def main():
    """Main application"""
    initialize_session_state()
    _cleanup_old_sessions()
    
    # Load engines
    if not load_engines():
        st.stop()
    
    # ----- LEFT SIDEBAR: New Chat + Chat history -----
    with st.sidebar:
        if st.button("New Chat", type="primary", use_container_width=True, key="sidebar_new_chat"):
            _create_new_session()
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Chat history**")
        
        session_ids = list(st.session_state.chat_sessions.keys())
        session_ids.sort(
            key=lambda sid: st.session_state.chat_sessions[sid]["created_at"],
            reverse=True
        )
        
        # Only show sessions that have messages (exclude empty "New chat" session)
        session_ids_with_messages = [
            sid for sid in session_ids
            if len(st.session_state.chat_sessions[sid]["messages"]) > 0
        ]
        if session_ids_with_messages:
            for sid in session_ids_with_messages:
                title = st.session_state.chat_sessions[sid]["title"]
                is_current = sid == st.session_state.current_session_id
                if st.button(
                    title,
                    key=f"session_{sid}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    if sid != st.session_state.current_session_id:
                        st.session_state.current_session_id = sid
                        st.rerun()
        else:
            st.caption("No chats yet")
        
        st.markdown("---")
        if st.session_state.engine_loaded:
            df = st.session_state.query_engine.df
            st.caption(f"Records: {len(df)} | Make/Models: {df['make_model'].nunique()}")
    
    # ----- MAIN CONTENT: Header + Messages + Input -----
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 1.5rem; font-weight: 600; color: #171717; margin-bottom: 0.25rem;">Asset Failure Analytics Chatbot</h1>
        <p style="font-size: 0.875rem; color: #6B7280;">Ask questions about fleet maintenance, failures, and troubleshooting.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ----- Chat messages with auto-scroll to newest message -----
    messages = _get_current_messages()
    message_count = len(messages)
    
    # Display all messages (no height constraint)
    for i, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("viz_fig") is not None:
                st.plotly_chart(message["viz_fig"], use_container_width=True, key=f"viz_{i}")
    
    # Scroll to bottom by adding element at end
    if message_count > 0:
        st.markdown('<div id="chat-end" style="height: 1px;"></div>', unsafe_allow_html=True)
    
    # ----- Chat input with stable key (fixes icon disappearing) -----
    question = st.chat_input("Ask me anything about fleet failures...", key="main_chat_input")
    if question:
        handle_query(question)
        st.rerun()


if __name__ == "__main__":
    main()
