from llama_index.core import (
    Settings, 
    get_response_synthesizer, 
    PromptTemplate
)
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.callbacks import CallbackManager
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.selectors.llm_selectors import LLMSingleSelector

from common.prompts_templates.PromptTemplates import (
    SINGLE_SELECT_PROMPT, 
    CONDENSE_QUESTION_PROMPT
)

from chat_engine.LogHandler.shell import shell_colors
from chat_engine.LogHandler.LogHandler import CustomLlamaIndexCallbackHandler
from chat_engine.SemanticSearchQE.SemanticSearchQETool import build_SemanticSearchQETool
from chat_engine.GeneralInteractionQE.GeneralInteractionQETool import (
    build_AssetMappingQueryEngine, 
    build_ProblemsReportingQueryEngine,
    build_GeneralInteractionQueryEngine,
    build_ChatbotInfoQueryEngine
)
from chat_engine.LoadIndex.load_vector_indices import load_vector_indices


def load_chat_engine() -> CondenseQuestionChatEngine:
    """
    Initialize the core component of the whole RAG application.
    The chat engine manages the router, the vector indices, and the tools.
    This uses the global LLM and embedding models set previously.

    Returns:
        - the chat engine instance.
    """
    callback_manager = CallbackManager([CustomLlamaIndexCallbackHandler()])
    node_parser = SentenceSplitter(
        separator=" ",
        chunk_size=256,
        chunk_overlap=0,
        paragraph_separator="\n",
        secondary_chunking_regex="[^.]+[.]?",
        include_metadata=True, 
        include_prev_next_rel=True,
    )

    # Set them as global tools
    Settings.node_parser = node_parser
    Settings.callback_manager = callback_manager

    # Build the tools
    nodes, vector_indices = load_vector_indices()        
    semantic_search_query_engine_tools = build_SemanticSearchQETool(nodes, vector_indices)
    asset_mapping_query_engine_tool = build_AssetMappingQueryEngine()
    problems_reporting_query_engine_tool = build_ProblemsReportingQueryEngine()
    general_interaction_query_engine_tool = build_GeneralInteractionQueryEngine()
    chatbot_info_query_engine_tool = build_ChatbotInfoQueryEngine()

    # Define the router, using one route for each of the tools defined above
    router_query_engine = RouterQueryEngine.from_defaults(
        selector=LLMSingleSelector.from_defaults(prompt_template_str=SINGLE_SELECT_PROMPT),
        summarizer=get_response_synthesizer(response_mode=ResponseMode.COMPACT, verbose=True, streaming=True, use_async=False),
        query_engine_tools=list(semantic_search_query_engine_tools.values()) + \
                            [asset_mapping_query_engine_tool] + \
                            [problems_reporting_query_engine_tool] + \
                            [general_interaction_query_engine_tool] + \
                            [chatbot_info_query_engine_tool],
        )

    print(f"\n{shell_colors['BOLD']}{shell_colors['HEADER']}QueryEngine Metadatas: {shell_colors['ENDC']}{shell_colors['ENDC']}", "\n".join([f"\t- {shell_colors['BOLD']}TOOL {idx}{shell_colors['ENDC']}: {shell_colors['OKBLUE']}\"{x.name}\"{shell_colors['ENDC']} - {x.description}" for idx,x in enumerate(router_query_engine._metadatas)]), sep="\n")

    # Return the chat engine
    # See https://docs.llamaindex.ai/en/stable/api_reference/chat_engines/condense_question/ for reference
    return CondenseQuestionChatEngine.from_defaults(
        query_engine=router_query_engine,
        condense_question_prompt=PromptTemplate(CONDENSE_QUESTION_PROMPT),
        memory=ChatMemoryBuffer.from_defaults(chat_history=[], token_limit=1024, tokenizer_fn=node_parser._tokenizer),
        streaming=True,
        use_async=False,
        verbose=True
        )
