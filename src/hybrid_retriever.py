"""
Hybrid Retriever Module
-----------------------
Performs vector-based anchor discovery followed by topological expansion.
"""

import networkx as nx

class HybridRetriever:
    """Executes search queries against a populated KnowledgeBase."""
    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def retrieve(self, query, top_k=1, hop_depth=1):
        """
        Executes Hybrid Retrieval: 
        1. Semantic Search for anchor nodes using FAISS.
        2. Context Expansion via Breadth-First Search (BFS) in NetworkX.
        """
        # Step 1: Anchor Discovery
        query_emb = self.kb.encoder.encode([query]).astype('float32')
        distances, indices = self.kb.index.search(query_emb, top_k)
        
        retrieved_context = []
        retrieved_nodes = set()
        anchor_nodes = [self.kb.node_id_map[idx] for idx in indices[0] if idx != -1]
        
        # Step 2: Topological Context Expansion
        for anchor in anchor_nodes:
            anchor_content = self.kb.graph.nodes[anchor]['content']
            retrieved_context.append(f"[Anchor] {anchor_content}")
            retrieved_nodes.add(anchor)
            
            # BFS Traversal
            edges = list(nx.bfs_edges(self.kb.graph, source=anchor, depth_limit=hop_depth))
            for u, v in edges:
                relation = self.kb.graph.edges[u, v]['relation']
                target_content = self.kb.graph.nodes[v]['content']
                retrieved_context.append(
                    f"[Hop] {self.kb.graph.nodes[u]['content']} --({relation})--> {target_content}"
                )
                retrieved_nodes.update([u, v])
                
        return list(set(retrieved_context)), list(retrieved_nodes)