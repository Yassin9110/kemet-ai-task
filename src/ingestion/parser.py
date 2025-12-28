# src/ingestion/parser.py
"""
Document parser for PDF and TXT files.
- PDF: Uses LlamaParse for intelligent parsing
- TXT: Direct file reading (simple and fast)
"""

from pathlib import Path
from llama_parse import LlamaParse
from src.config import settings
# from src.core.logging import  get_logger

#logger = get_#logger(__name__, settings.log_level)


class DocumentParser:
    """
    Parses documents and extracts text content.
    
    Example:
        parser = DocumentParser()
        result = parser.parse("document.pdf", file_bytes)
        print(result["text"])
        print(result["pages"])
    """
    
    def __init__(self):
        """Initialize the parser with LlamaParse for PDFs."""
        self.llama_parser = LlamaParse(
            api_key=settings.llama_cloud_api_key,
            result_type="text",  # We want plain text output
        )
        #logger.info("DocumentParser initialized")
    
    def parse(self, filename: str, file_bytes: bytes) -> dict:
        """
        Parse a document and extract text.
        
        Args:
            filename: Name of the file (used to detect type)
            file_bytes: Raw file content as bytes
            
        Returns:
            Dictionary with:
                - "text": Extracted text content
                - "pages": Number of pages (None for TXT)
                - "file_type": "pdf" or "txt"
        """
        # Get file extension
        file_type = Path(filename).suffix.lower().strip(".")
        
        #logger.info(f"Parsing file: {filename} (type: {file_type})")
        
        # Route to the right parser
        if file_type == "pdf":
            return self._parse_pdf(filename, file_bytes)
        elif file_type == "txt":
            return self._parse_txt(filename, file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _parse_pdf(self, filename: str, file_bytes: bytes) -> dict:
        """
        Parse a PDF file using LlamaParse.
        
        Args:
            filename: Name of the PDF file
            file_bytes: PDF content as bytes
            
        Returns:
            Dictionary with text, pages, and file_type
        """
        # try:
        # LlamaParse expects extra_info for metadata
        documents = self.llama_parser.load_data(
            file_bytes,
            extra_info={"file_name": filename}
        )
        
        # Combine all pages into one text
        # Each document in the list is typically one page
        all_text = ""
        page_texts = []
        
        for doc in documents:
            page_texts.append(doc.text)
            all_text += doc.text + "\n\n"
        
        num_pages = len(documents)
        
        #logger.info(f"PDF parsed: {num_pages} pages extracted")
        
        return {
            "text": all_text.strip(),
            "pages": num_pages,
            "page_texts": page_texts,  # Individual page content
            "file_type": "pdf"
        }
            
        # except Exception as e:
        #     #logger.error(f"PDF parsing failed: {str(e)}")
        #     raise
    
    def _parse_txt(self, filename: str, file_bytes: bytes) -> dict:
        """
        Parse a TXT file by direct reading.
        
        Args:
            filename: Name of the TXT file
            file_bytes: TXT content as bytes
            
        Returns:
            Dictionary with text, pages (None), and file_type
        """
        # try:
        # Try UTF-8 first, then fallback to other encodings
        text = self._decode_text(file_bytes)
        
        #logger.info(f"TXT parsed: {len(text)} characters")
        
        return {
            "text": text,
            "pages": None,  # TXT files don't have pages
            "page_texts": [text],  # Treat whole file as one "page"
            "file_type": "txt"
        }
            
        # except Exception as e:
        #     #logger.error(f"TXT parsing failed: {str(e)}")
        #     raise
    
    def _decode_text(self, file_bytes: bytes) -> str:
        """
        Decode bytes to string, trying multiple encodings.
        
        Args:
            file_bytes: Raw bytes to decode
            
        Returns:
            Decoded string
        """
        # Common encodings to try (UTF-8 handles most Arabic text)
        encodings = ["utf-8", "utf-16", "cp1256", "iso-8859-6"]
        
        for encoding in encodings:
            # try:
            return file_bytes.decode(encoding)
            # except UnicodeDecodeError:
            #     continue
        
        # Last resort: decode with replacement characters
        return file_bytes.decode("utf-8", errors="replace")