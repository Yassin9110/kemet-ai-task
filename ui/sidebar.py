# ui/sidebar.py
"""
Sidebar component for document upload and management.
"""

import streamlit as st
from src.config import settings
from .session import (
    get_rag,
    add_document,
    get_documents,
    clear_all,
    has_documents
)
from .components import (
    show_error,
    show_success,
    show_document_card,
    show_spinner,
    show_warning
)


def render_sidebar():
    """Render the sidebar with upload and document management."""
    
    with st.sidebar:
        st.header("📁 Documents")
        
        # File uploader
        _render_uploader()
        
        st.divider()
        
        # Uploaded documents list
        _render_document_list()
        
        st.divider()
        
        # Actions
        _render_actions()
        
        st.divider()
        
        # Help section
        _render_help()


def _render_uploader():
    """Render the file upload section."""
    st.subheader("Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help=f"Maximum file size: {settings.max_file_size_mb}MB"
    )
    
    if uploaded_file is not None:
        _process_uploaded_file(uploaded_file)


def _process_uploaded_file(uploaded_file):
    """Process an uploaded file."""
    
    # Check file size
    file_size = len(uploaded_file.getvalue())
    max_size = settings.max_file_size_bytes
    
    if file_size > max_size:
        show_error(f"File too large! Max size: {settings.max_file_size_mb}MB")
        return
    
    # Check if already uploaded
    existing_docs = [d["name"] for d in get_documents()]
    if uploaded_file.name in existing_docs:
        show_warning("This document is already uploaded.")
        return
    
    # Process button
    if st.button("📥 Process Document", type="primary", use_container_width=True):
        _ingest_document(uploaded_file)


def _ingest_document(uploaded_file):
    """Ingest a document into the RAG system."""
    
    with show_spinner("Processing document... This may take a moment."):
        try:
            # Get RAG orchestrator
            rag = get_rag()
            
            # Read file bytes
            file_bytes = uploaded_file.getvalue()
            
            # Ingest document
            result = rag.ingest_document(uploaded_file.name, file_bytes)
            
            if result.success:
                # Track document
                add_document(
                    doc_name=result.document_name,
                    num_chunks=result.total_chunks,
                    language=result.language
                )
                
                show_success(
                    f"Document processed! "
                    f"Created {result.total_chunks} chunks "
                    f"in {result.processing_time_ms:.0f}ms"
                )
            else:
                show_error(f"Failed to process: {result.error_message}")
                
        except Exception as e:
            show_error(f"Error: {str(e)}")


def _render_document_list():
    """Render the list of uploaded documents."""
    st.subheader("Uploaded Documents")
    
    documents = get_documents()
    
    if not documents:
        st.caption("No documents uploaded yet.")
    else:
        for doc in documents:
            show_document_card(doc)


def _render_actions():
    """Render action buttons."""
    st.subheader("Actions")
    
    if st.button("🗑️ Clear All", use_container_width=True):
        clear_all()
        st.rerun()


def _render_help():
    """Render help section."""
    with st.expander("ℹ️ Help"):
        st.markdown("""
        **How to use:**
        1. Upload a PDF or TXT document
        2. Click "Process Document"
        3. Ask questions in the chat
        
        **Supported languages:**
        - 🇺🇸 English
        - 🇸🇦 Arabic
        
        **Tips:**
        - Ask questions in the same language as your document
        - Be specific in your questions
        - Use follow-up questions for more details
        """)