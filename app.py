# app.py
"""
Main Streamlit application entry point.
Run with: streamlit run app.py
"""

import streamlit as st

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="Multilingual RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import UI components
from ui.session import init_session_state
from ui.components import show_header
from ui.sidebar import render_sidebar
from ui.chat import render_chat


def main():
    """Main application function."""
    
    # Initialize session state
    init_session_state()
    
    # Custom CSS for better RTL support
    _add_custom_css()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    show_header()
    render_chat()


def _add_custom_css():
    """Add custom CSS for RTL support and styling."""
    st.markdown("""
        <style>
        /* RTL support for Arabic text */
        [dir="rtl"] {
            text-align: right;
            direction: rtl;
        }
        
        /* Chat message styling */
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        
        /* Make chat input wider */
        .stChatInput {
            max-width: 100%;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            padding: 1rem;
        }
        
        /* Document card styling */
        .document-card {
            padding: 0.5rem;
            border-radius: 0.25rem;
            background-color: #f0f2f6;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()