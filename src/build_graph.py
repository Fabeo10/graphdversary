"""
Graph Builder Module
--------------------
Handles the ingestion of JSON corpora, instantiation of the NetworkX 
topological graph, and building of the FAISS vector index.
"""

import json
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer
import warnings

# Suppress verbose HuggingFace warnings
warnings.filterwarnings("ignore")

_ENCODER_CACHE = {}


class KnowledgeBase:
    """Stores the graph, vector index, and mappings for the RAG system."""
    def __init__(self, embedding_model='all-MiniLM-L6-v2', verbose=True):
        self.graph = nx.DiGraph()
        self.encoder = self._load_encoder(embedding_model)
        self.index = None
        self.node_id_map = {}
        self.id_node_map = {}
        self.verbose = verbose

    @staticmethod
    def _load_encoder(embedding_model):
        if embedding_model not in _ENCODER_CACHE:
            _ENCODER_CACHE[embedding_model] = SentenceTransformer(embedding_model)
        return _ENCODER_CACHE[embedding_model]

    def build_from_json(self, filepath):
        """Loads entities and edges from a JSON file into the knowledge base."""
        if self.verbose:
            print(f"[*] Loading data from {filepath}...")
        with open(filepath, 'r') as f:
            data = json.load(f)

        entities = data.get('entities', [])
        edges = data.get('edges', [])
        self.graph.clear()
        self.node_id_map = {}
        self.id_node_map = {}

        # 1. Build Topological Graph
        for entity in entities:
            self.graph.add_node(
                entity['id'], 
                content=entity['content'], 
                type=entity.get('type', 'generic')
            )
            
        for edge in edges:
            self.graph.add_edge(
                edge['source'], 
                edge['target'], 
                relation=edge['relation']
            )

        # 2. Build Semantic Vector Index
        node_ids = list(self.graph.nodes())
        contents = [self.graph.nodes[nid]['content'] for nid in node_ids]
        
        embeddings = self.encoder.encode(contents).astype('float32')
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        for idx, nid in enumerate(node_ids):
            self.node_id_map[idx] = nid
            self.id_node_map[nid] = idx
            
        if self.verbose:
            print(f"[*] Ingestion complete. Indexed {len(node_ids)} nodes and {len(edges)} edges.")
        return data.get('metadata', {})

    def rebuild_index(self):
        """Rebuild the vector index after defensive graph mutations."""
        node_ids = list(self.graph.nodes())
        contents = [self.graph.nodes[nid]['content'] for nid in node_ids]
        embeddings = self.encoder.encode(contents).astype('float32')
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.node_id_map = {idx: nid for idx, nid in enumerate(node_ids)}
        self.id_node_map = {nid: idx for idx, nid in enumerate(node_ids)}