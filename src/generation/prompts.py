# src/generation/prompts.py
"""
Prompt templates for the RAG system.
Separate prompts for English and Arabic responses.
"""


# English system prompt
ENGLISH_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided documents.

RULES:
1. Answer ONLY using the information in the CONTEXT below
2. If the answer is not in the context, say "I don't have information about that in the documents"
3. Be clear and concise
4. When citing sources, mention page numbers like [Page 1] or [Page 2]
5. Always respond in English

CONTEXT:
{context}"""


# Arabic system prompt
ARABIC_SYSTEM_PROMPT = """أنت مساعد مفيد يجيب على الأسئلة بناءً على المستندات المقدمة.

القواعد:
1. أجب فقط باستخدام المعلومات الموجودة في السياق أدناه
2. إذا لم تكن الإجابة موجودة في السياق، قل "لا تتوفر لدي معلومات حول ذلك في المستندات"
3. كن واضحاً ومختصراً
4. عند الاستشهاد بالمصادر، اذكر أرقام الصفحات مثل [صفحة 1] أو [صفحة 2]
5. أجب دائماً باللغة العربية

السياق:
{context}"""


# No answer found - English
NO_ANSWER_ENGLISH = "I don't have information about that in the uploaded documents."

# No answer found - Arabic
NO_ANSWER_ARABIC = "لا تتوفر لدي معلومات حول ذلك في المستندات المرفوعة."


def get_system_prompt(language: str, context: str) -> str:
    """
    Get the system prompt for a specific language.
    
    Args:
        language: "en" for English, "ar" for Arabic
        context: The retrieved context to include
        
    Returns:
        Formatted system prompt
    """
    if language == "ar":
        return ARABIC_SYSTEM_PROMPT.format(context=context)
    else:
        return ENGLISH_SYSTEM_PROMPT.format(context=context)


def get_no_answer_message(language: str) -> str:
    """
    Get the "no answer found" message for a specific language.
    
    Args:
        language: "en" for English, "ar" for Arabic
        
    Returns:
        No answer message in the appropriate language
    """
    if language == "ar":
        return NO_ANSWER_ARABIC
    else:
        return NO_ANSWER_ENGLISH