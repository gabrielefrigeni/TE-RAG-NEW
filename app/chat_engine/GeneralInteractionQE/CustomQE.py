from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.base.response.schema import StreamingResponse
from typing import Optional
from common.prompts_templates.PromptTemplates import (
    MAPPING_SEARCH_RESPONSE, 
    SYSTEM_PROMPT_ASSISTANT
)
from models.models import gemini_flash


class ProblemsReportingQueryEngine(CustomQueryEngine):
    """
    Tool for reporting issues.
    """
    retriever: Optional[BaseRetriever]
    response_synthesizer: Optional[BaseSynthesizer]

    def custom_query(self, query_str: str):
        return StreamingResponse(response_gen=(char for char in "Grazie per la segnalazione. La problematica riscontrata verrà inviata a un assistente che provvederà alla verifica manuale."))


class GeneralInteractionQueryEngine(CustomQueryEngine):
    """
    Tool for out of scope user queries.
    """
    retriever: Optional[BaseRetriever]
    response_synthesizer: Optional[BaseSynthesizer]

    def custom_query(self, query_str: str):
        return StreamingResponse(response_gen=(char for char in "Mi dispiace, la tua domanda non sembra riguardare alcun asset nel vecchio o nel nuovo DWH."))


class ChatbotInfoQueryEngine(CustomQueryEngine):
    """
    Tool for information about the chatbot.
    """
    retriever: Optional[BaseRetriever]
    response_synthesizer: Optional[BaseSynthesizer]

    def custom_query(self, query_str: str):
        model = gemini_flash
        model._system_instruction = SYSTEM_PROMPT_ASSISTANT
        response = model.generate_content(query_str)
        response = response.text.strip()
        return StreamingResponse(response_gen=(char for char in response))
