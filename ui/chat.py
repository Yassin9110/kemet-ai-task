# ui/chat.py
"""
Chat interface component.
"""

import streamlit as st
from .session import (
    get_rag,
    add_message,
    get_chat_history,
    clear_chat,
    has_documents
)
from .components import (
    show_message,
    show_warning,
    show_spinner
)


def render_chat():
    """Render the chat interface."""
    
    # Chat header with clear button
    col1, col2 = st.columns([6, 1])
    
    with col1:
        st.subheader("💬 Chat")
    
    with col2:
        if st.button("🔄", help="Clear chat"):
            clear_chat()
            st.rerun()
    
    # Display chat messages
    _render_messages()
    
    # Chat input
    _render_input()


def _render_messages():
    """Render all chat messages."""
    
    for message in st.session_state.messages:
        show_message(message["role"], message["content"])


def _render_input():
    """Render chat input box."""
    
    # Check if documents are uploaded
    if not has_documents():
        st.chat_input("Upload a document first...", disabled=True)
        show_warning("Please upload a document before asking questions.")
        return
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        _handle_user_input(prompt)


def _handle_user_input(prompt: str):
    """
    Handle user input and generate response.
    
    Args:
        prompt: User's question
    """
    # Add user message
    add_message("user", prompt)
    show_message("user", prompt)
    
    # Generate response
    with show_spinner("Thinking..."):
        # try:
        rag = get_rag()
        chat_history = get_chat_history()
        
        # Query RAG system
        result = rag.query(prompt, chat_history)
        
        # Add assistant message
        add_message("assistant", result.answer)
        show_message("assistant", result.answer)
            
        # except Exception as e:
        #     error_msg = f"Sorry, an error occurred: {str(e)}"
        #     add_message("assistant", error_msg)
        #     show_message("assistant", error_msg)