from llama_index.core.retrievers import BaseRetriever

class HybridRetriever(BaseRetriever):
    """
    Builds an hybrid retriever, which uses both a vector similarity metric (e.g., cosine) and the BM25 algorithm to retrieve docs.
    """
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__(None)

    def _retrieve(self, query, **kwargs):
        # Retrieve the top nodes for each algorithm
        bm25_nodes = self.bm25_retriever.retrieve(query.query_str, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # Combine the two lists of nodes excluding duplicates
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)

        return all_nodes
