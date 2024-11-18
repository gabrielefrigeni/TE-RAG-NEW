from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.indices.utils import default_parse_choice_select_answer_fn
import Stemmer
from chat_engine.LogHandler.shell import shell_colors
from common.prompts_templates.PromptTemplates import TOOL_DESCRIPTIONS
from chat_engine.SemanticSearchQE.HybridRetriever import HybridRetriever


def custom_print_choice_select_answer_fn(answer: str, num_choices: int):
    """
    Custom print of reranker response.
    """
    
    print(shell_colors['OKCYAN'], end='')
    print(f'==> Reranker answer: {answer}')
    print(f'Number of choices: {num_choices}')
    print(shell_colors['ENDC'], end='')

    return default_parse_choice_select_answer_fn(answer, num_choices)


def build_SemanticSearchQETool(nodes, vector_indices):
    """
    Builds the tool to perform semantic search within the vector database.

    Args:
        nodes: the list of nodes.
        vector_indices: the vector database.

    Returns:
        - LlamaIndex query engine function.
    """
    semantic_search_query_engine = {}

    for key_name in vector_indices.keys():

        # Initialize a vector similarity retriever
        vector_retriever = VectorIndexRetriever(
            index=vector_indices[key_name],
            similarity_top_k=10,
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
            )
        
        # Initialize a BM25 retriever
        # Reference: https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/
        bm25_retriever = BM25Retriever(
            nodes=nodes[key_name], 
            stemmer=Stemmer.Stemmer("italian"), # Removes stop words and stems each word
            verbose=True,
            similarity_top_k=10,
            language="italian"
            )    
        
        # Combine two retrieval methods into an hybrid retriever
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

        # Initialize a query engine using the hybrid retriever and a re-ranker
        query_engine = RetrieverQueryEngine(
            retriever=hybrid_retriever,
            node_postprocessors=[
                LLMRerank(
                   choice_batch_size=20, 
                   top_n=10,
                   parse_choice_select_answer_fn=custom_print_choice_select_answer_fn,
                   )
                ],
            response_synthesizer=get_response_synthesizer(
                response_mode=ResponseMode.COMPACT, 
                verbose=True, 
                streaming=True, 
                use_async=False),
        )

        semantic_search_query_engine[key_name] = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            description=TOOL_DESCRIPTIONS["semantic_search"].format(key_name=" ".join([c.capitalize() for c in key_name.split("_")])),
            name=f"semantic_search_{key_name}"
        )

    return semantic_search_query_engine
