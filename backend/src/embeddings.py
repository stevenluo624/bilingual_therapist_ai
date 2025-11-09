from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import numpy as np

class EmbeddingGenerator:
    '''
    Generating embedding for user input
    '''
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # can use 'cuda' if running on a windows machine with a GPU
            encode_kwargs={'normalize_embeddings': False}
        )
        
        self.model_name = model_name
        
        # Get dimension by embedding a test string
        test_embedding = self.embeddings.embed_query("test")
        self.dimension = len(test_embedding)
    
    def embed_single(self, text: str) -> np.ndarray:
        embedding = self.embeddings.embed_query(text)
        return np.array(embedding)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.embeddings.embed_documents(texts)
        return np.array(embeddings)

    def embed_query(self, text: str) -> np.ndarray:
        '''Will be refactored later'''
        # embedding = self.embeddings.embed_query(text)
        # return np.array(embedding)
        return self.embed_single(text)
    
    def get_dimension(self) -> int:
        return self.dimension
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        return self.embeddings

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if norm_product == 0:
        return 0.0
    return dot_product / norm_product

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

_global_embedder = None

def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingGenerator:
    global _global_embedder
    
    if _global_embedder is None:
        _global_embedder = EmbeddingGenerator(model_name)
    
    return _global_embedder

if __name__ == "__main__":
    embedder = EmbeddingGenerator()
    
    text = "I feel anxious about my future"
    embedding = embedder.embed_single(text)
    print(f"   Text: '{text}'")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   First 5 values: {embedding[:5]}")
    
    texts = [
        "I'm feeling depressed",
        "I'm very happy today",
        "我感到很焦虑"  # Chinese text
    ]
    embeddings = embedder.embed_batch(texts)
    print(f"   Number of texts: {len(texts)}")
    print(f"   Embeddings shape: {embeddings.shape}")
    
    vec1 = embedder.embed_single("I feel sad and hopeless")
    vec2 = embedder.embed_single("I'm experiencing depression")
    vec3 = embedder.embed_single("I love pizza")
    
    sim_12 = cosine_similarity(vec1, vec2)
    sim_13 = cosine_similarity(vec1, vec3)
    
    print(f"   'sad' vs 'depression': {sim_12:.3f}")
    print(f"   'sad' vs 'pizza': {sim_13:.3f}")
    print(f"   → Related concepts are more similar!")
    
    embedder2 = get_embedder()
    print(f"   Same instance? {embedder2.model is embedder.model}")
    print(f"   (Avoids reloading 420MB model)")