from llama_index.core.schema import TextNode
import json 

def parse_chroma_node(chroma_record):
    node = json.loads(chroma_record[0]["_node_content"])
    text = chroma_record[1]
    
    node["text"] = text
    return TextNode(**node)

def get_chroma_nodes(chroma_collection):
    collection_nodes = chroma_collection.get()
    return [parse_chroma_node(x) for x in zip(collection_nodes["metadatas"], collection_nodes["documents"])]