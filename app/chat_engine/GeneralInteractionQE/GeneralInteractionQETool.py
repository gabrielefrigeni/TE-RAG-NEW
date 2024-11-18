from llama_index.core.tools import QueryEngineTool
from common.prompts_templates.PromptTemplates import TOOL_DESCRIPTIONS
from chat_engine.GeneralInteractionQE.CustomQE import ( 
    ProblemsReportingQueryEngine,
    GeneralInteractionQueryEngine,
    ChatbotInfoQueryEngine
) 



def build_ProblemsReportingQueryEngine():    
    qe = ProblemsReportingQueryEngine(
        retriever=None,
        response_synthesizer=None,
    )

    return QueryEngineTool.from_defaults(
        query_engine=qe,
        description=TOOL_DESCRIPTIONS["problems_reporting"],
        name="problems_reporting"
    )


def build_GeneralInteractionQueryEngine():    
    qe = GeneralInteractionQueryEngine(
        retriever=None,
        response_synthesizer=None,
    )

    return QueryEngineTool.from_defaults(
        query_engine=qe,
        description=TOOL_DESCRIPTIONS["general_interaction"],
        name="general_interaction"
    )


def build_ChatbotInfoQueryEngine():    
    qe = ChatbotInfoQueryEngine(
        retriever=None,
        response_synthesizer=None,
    )

    return QueryEngineTool.from_defaults(
        query_engine=qe,
        description=TOOL_DESCRIPTIONS["chatbot_info"],
        name="chatbot_info"
    )
