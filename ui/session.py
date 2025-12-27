# ui/session.py
"""
Session state management for Streamlit.
Handles chat history, document state, and RAG orchestrator.
"""

import streamlit as st
from src.orchestrator import RAGOrchestrator
from src.config import settings


def init_session_state():
    """
    Initialize all session state variables.
    Call this at the start of your app.
    """
    # Chat messages: list of {"role": "user/assistant", "content": "..."}
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Uploaded documents info
    if "documents" not in st.session_state:
        st.session_state.documents = []
    
    # RAG orchestrator (the main engine)
    if "rag" not in st.session_state:
        st.session_state.rag = None
    
    # Processing state
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False


def get_rag() -> RAGOrchestrator:
    """
    Get or create the RAG orchestrator.
    
    Returns:
        RAGOrchestrator instance
    """
    if st.session_state.rag is None:
        st.session_state.rag = RAGOrchestrator()
    
    return st.session_state.rag


def add_message(role: str, content: str):
    """
    Add a message to chat history.
    
    Args:
        role: "user" or "assistant"
        content: Message text
    """
    st.session_state.messages.append({
        "role": role,
        "content": content
    })


def get_chat_history() -> list[dict]:
    """
    Get recent chat history for context.
    
    Returns:
        List of recent messages (limited by max_conversation_turns)
    """
    max_messages = settings.max_conversation_turns * 2  # user + assistant pairs
    return st.session_state.messages[-max_messages:]


def clear_chat():
    """Clear chat history."""
    st.session_state.messages = []


def clear_all():
    """Clear everything and start fresh."""
    st.session_state.messages = []
    st.session_state.documents = []
    
    # Clear vector store if RAG exists
    if st.session_state.rag is not None:
        st.session_state.rag.clear_documents()


def add_document(doc_name: str, num_chunks: int, language: str):
    """
    Track an uploaded document.
    
    Args:
        doc_name: Document filename
        num_chunks: Number of chunks created
        language: Detected language
    """
    st.session_state.documents.append({
        "name": doc_name,
        "chunks": num_chunks,
        "language": language
    })


def get_documents() -> list[dict]:
    """Get list of uploaded documents."""
    return st.session_state.documents


def has_documents() -> bool:
    """Check if any documents are uploaded."""
    return len(st.session_state.documents) > 0