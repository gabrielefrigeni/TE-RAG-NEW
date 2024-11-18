from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from chat_engine.LoadIndex.chroma_utils import get_chroma_nodes
from chat_engine.LogHandler.shell import shell_colors
import chromadb
from chromadb.config import Settings
import os

def load_vector_indices():
    """
    Load the Chroma vector database.

    Returns:
        - the list of nodes for each collection.
        - the vector indices for each collection.
    """
    chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH"), settings = Settings(anonymized_telemetry=True))

    vector_indices = {}
    nodes = {}
    for collection in chroma_client.list_collections():
        chroma_collection = chroma_client.get_or_create_collection(collection.name, metadata={"hnsw:space": "cosine"})
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        vector_indices[collection.name] = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )

        nodes[collection.name] = get_chroma_nodes(chroma_collection)

    print(f"\n{shell_colors['BOLD']}{shell_colors['HEADER']}Available Documents: {shell_colors['ENDC']}{shell_colors['ENDC']}", "\n".join([f"\t- {shell_colors['OKBLUE']}\"{key}\"{shell_colors['ENDC']} - {len(val)} nodes" for key,val in nodes.items()]), sep="\n")
        
    return nodes, vector_indices
