import os
from models.gcp_client import init_gcp_client
from llama_index.llms.vertex import Vertex
from llama_index.embeddings.vertex import VertexTextEmbedding
from vertexai.generative_models import GenerationConfig, GenerativeModel
from llama_index.core import Settings

# Get the credentials to be passed to the LlamaIndex Vertex models
credentials = init_gcp_client()

def load_index_models(credentials) -> None:
    """
    Load the LlamaIndex LLM and embedding model, and give them global scope within the codebase.
    See https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/ for further information.

    Args:
        credentials: the GCP credentials.
    """
    # Initialize the Gemini LLM
    llm = Vertex(
        model=os.getenv("LLM_MODEL"), 
        project=credentials.project_id, 
        credentials=credentials, 
        temperature=os.getenv("LLM_TEMPERATURE"),
        max_tokens=os.getenv("LLM_MAX_TOKENS"), 
        system_prompt="Rispondi sempre in italiano." # This system prompt is the same for all the LlamaIndex components using the LLM
    )
    # Initialize the embedding model
    embeddings_model = VertexTextEmbedding(
            model_name=os.getenv("EMBEDDING_MODEL"),
            project=credentials.project_id,
            location=os.getenv("GCP_REGION"),
            credentials=credentials,
        )

    Settings.llm = llm
    Settings.embed_model = embeddings_model

# Load the models and give them global scope
load_index_models(credentials)

# Initialise also a separate Gemini instance for secondary tasks
generation_config = GenerationConfig(
    max_output_tokens=int(os.getenv("LLM_MAX_TOKENS")),
    temperature=float(os.getenv("LLM_TEMPERATURE")),
    top_p=float(os.getenv("LLM_TOP_P"))
)

gemini_flash = GenerativeModel(
    os.getenv("LLM_MODEL"),
    generation_config=generation_config
)
