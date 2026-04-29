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

    def retrieve(self, query, top_k=1, hop_depth=1, include_trace=False):
        """
        Executes Hybrid Retrieval: 
        1. Semantic Search for anchor nodes using FAISS.
        2. Context Expansion via Breadth-First Search (BFS) in NetworkX.
        """
        # Step 1: Anchor Discovery
        query_emb = self.kb.encoder.encode([query]).astype('float32')
        distances, indices = self.kb.index.search(query_emb, top_k)
        
        retrieved_context = []
        retrieved_nodes = []
        seen_nodes = set()
        trace = {
            "query": query,
            "top_k": top_k,
            "hop_depth": hop_depth,
            "anchors": [],
            "hops": [],
        }

        anchor_nodes = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            node_id = self.kb.node_id_map[idx]
            distance = float(distances[0][rank])
            anchor_nodes.append(node_id)
            trace["anchors"].append({
                "rank": rank + 1,
                "node_id": node_id,
                "distance": distance,
                "content": self.kb.graph.nodes[node_id]["content"],
            })
        
        # Step 2: Topological Context Expansion
        for anchor in anchor_nodes:
            anchor_content = self.kb.graph.nodes[anchor]['content']
            retrieved_context.append(f"[Anchor] {anchor_content}")
            if anchor not in seen_nodes:
                retrieved_nodes.append(anchor)
                seen_nodes.add(anchor)
            
            # BFS Traversal
            edges = list(nx.bfs_edges(self.kb.graph, source=anchor, depth_limit=hop_depth))
            for u, v in edges:
                relation = self.kb.graph.edges[u, v]['relation']
                target_content = self.kb.graph.nodes[v]['content']
                retrieved_context.append(
                    f"[Hop] {self.kb.graph.nodes[u]['content']} --({relation})--> {target_content}"
                )
                for node_id in (u, v):
                    if node_id not in seen_nodes:
                        retrieved_nodes.append(node_id)
                        seen_nodes.add(node_id)
                trace["hops"].append({
                    "source": u,
                    "target": v,
                    "relation": relation,
                    "source_content": self.kb.graph.nodes[u]["content"],
                    "target_content": target_content,
                })
                
        # Preserve retrieval order while removing duplicate context strings.
        ordered_context = list(dict.fromkeys(retrieved_context))

        if include_trace:
            return ordered_context, retrieved_nodes, trace

        return ordered_context, retrieved_nodes