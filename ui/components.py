# ui/components.py
"""
Reusable UI components for the Streamlit app.
"""

import streamlit as st


def show_header():
    """Display the app header."""
    st.title("🔍 Multilingual RAG System")
    st.markdown("*Ask questions about your documents in English or Arabic*")
    st.divider()


def show_message(role: str, content: str):
    """
    Display a chat message with appropriate styling.
    
    Args:
        role: "user" or "assistant"
        content: Message text
    """
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # Check if content might be Arabic (RTL)
            if _contains_arabic(content):
                st.markdown(f'<div dir="rtl">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(content)


def show_error(message: str):
    """Display an error message."""
    st.error(f"❌ {message}")


def show_success(message: str):
    """Display a success message."""
    st.success(f"✅ {message}")


def show_info(message: str):
    """Display an info message."""
    st.info(f"ℹ️ {message}")


def show_warning(message: str):
    """Display a warning message."""
    st.warning(f"⚠️ {message}")


def show_document_card(doc: dict):
    """
    Display a document info card.
    
    Args:
        doc: Dictionary with 'name', 'chunks', 'language'
    """
    lang_emoji = "🇸🇦" if doc["language"] == "ar" else "🇺🇸"
    
    st.markdown(f"""
    **📄 {doc['name']}**  
    {lang_emoji} {doc['language'].upper()} • {doc['chunks']} chunks
    """)


def show_stats(stats: dict):
    """
    Display system statistics.
    
    Args:
        stats: Dictionary with 'total_chunks'
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Chunks", stats.get("total_chunks", 0))
    
    with col2:
        st.metric("Documents", len(st.session_state.get("documents", [])))


def show_spinner(text: str):
    """
    Show a spinner with text.
    
    Usage:
        with show_spinner("Processing..."):
            do_something()
    """
    return st.spinner(text)


def _contains_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    for char in text:
        if '\u0600' <= char <= '\u06FF':
            return True
    return False