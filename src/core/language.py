# src/core/language.py
"""
Language detection for English and Arabic.
Uses the 'langdetect' library to identify the language of text.
"""

from langdetect import detect, LangDetectException
from .models import Language


class LanguageDetector:
    """
    Simple language detector for English and Arabic.
    
    Example usage:
        detector = LanguageDetector()
        language = detector.detect("Hello, how are you?")
        print(language)  # Language.ENGLISH
    """
    
    def detect(self, text: str) -> Language:
        """
        Detect the language of the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Language.ENGLISH, Language.ARABIC, or Language.UNKNOWN
        """
        # Step 1: Check if text is empty or too short
        if not text or len(text.strip()) < 3:
            return Language.UNKNOWN
        
        # Step 2: Try to detect the language
        # try:
        detected = detect(text)
        
        # Step 3: Map the result to our Language enum
        if detected == "en":
            return Language.ENGLISH
        elif detected == "ar":
            return Language.ARABIC
        else:
            # Other languages default to English for response
            return Language.ENGLISH
                
        # except LangDetectException:
        #     # If detection fails, return unknown
        #     return Language.UNKNOWN
    
    def is_arabic(self, text: str) -> bool:
        """
        Quick check if text is Arabic.
        
        Args:
            text: The text to check
            
        Returns:
            True if Arabic, False otherwise
        """
        return self.detect(text) == Language.ARABIC
    
    def is_english(self, text: str) -> bool:
        """
        Quick check if text is English.
        
        Args:
            text: The text to check
            
        Returns:
            True if English, False otherwise
        """
        return self.detect(text) == Language.ENGLISH


# Create a single instance to reuse (optional convenience)
language_detector = LanguageDetector()