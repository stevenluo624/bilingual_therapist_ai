from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pickle

@dataclass
class SearchResult:
    text: str
    score: float
    metadata: Dict[str, Any]
    index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'score': self.score,
            'metadata': self.metadata,
            'index': self.index
        }

class VectorStore:
    def __init__(self, faiss_path: Path, name: str = "VectorStore"):
        """
        Initialize by loading LangChain FAISS vector store.
        
        MAJOR CHANGE: Now loads LangChain FAISS instead of raw FAISS + pickle
        
        Args:
            faiss_path: Path to LangChain FAISS directory (not .faiss file)
            name: Human-readable name
        
        Note: LangChain FAISS stores everything in a directory with:
        - index.faiss (vectors)
        - index.pkl (metadata + documents)
        """
        self.name = name
        self.faiss_path = faiss_path
        
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS directory not found: {faiss_path}")
        
        from embeddings import get_embedder
        embedder = get_embedder()
        
        self.vector_store = FAISS.load_local(
            str(faiss_path),
            embedder.get_langchain_embeddings(),
            allow_dangerous_deserialization=True
        )
    
    def search(self, query: str, k: int = 5, 
               filter_dict: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        docs_and_scores = self.vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=filter_dict
        )
        
        results = []
        for i, (doc, score) in enumerate(docs_and_scores):
            results.append(SearchResult(
                text=doc.page_content,
                score=float(score),
                metadata=doc.metadata,
                index=i
            ))
        
        return results
    
    def search_by_vector(self, query_vector: List[float], k: int = 5) -> List[SearchResult]:
        import numpy as np
        
        query_array = np.array([query_vector], dtype=np.float32)
        distances, indices = self.vector_store.index.search(query_array, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            # Get document from docstore
            doc_id = self.vector_store.index_to_docstore_id[idx]
            doc = self.vector_store.docstore.search(doc_id)
            
            results.append(SearchResult(
                text=doc.page_content,
                score=float(dist),
                metadata=doc.metadata,
                index=int(idx)
            ))
        
        return results
    
    def size(self) -> int:
        return self.vector_store.index.ntotal
    
    def dimension(self) -> int:
        return self.vector_store.index.d
    
    def get_langchain_retriever(self, k: int = 4):
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
class TherapeuticRetriever:
    def __init__(self, 
                 therapeutic_faiss_dir: Path,
                 cbt_faiss_dir: Path):
        
        self.therapeutic_store = VectorStore(
            therapeutic_faiss_dir,
            name="Therapeutic Examples"
        )
        
        self.cbt_store = VectorStore(
            cbt_faiss_dir,
            name="CBT Techniques"
        )
    
    def retrieve(self, 
                query_text: str,
                k_examples: int = 3,
                k_techniques: int = 2,
                language_filter: Optional[str] = None) -> Dict[str, List[SearchResult]]:
        filter_dict = {'language': language_filter} if language_filter else None
        
        examples = self.therapeutic_store.search(
            query_text, 
            k=k_examples,
            filter_dict=filter_dict
        )
        
        techniques = self.cbt_store.search(
            query_text,
            k=k_techniques
        )
        
        return {
            'examples': examples,
            'techniques': techniques
        }

    def retrieve_by_vector(self,
                          query_vector: List[float],
                          k_examples: int = 3,
                          k_techniques: int = 2) -> Dict[str, List[SearchResult]]:
        examples = self.therapeutic_store.search_by_vector(query_vector, k=k_examples)
        techniques = self.cbt_store.search_by_vector(query_vector, k=k_techniques)
        
        return {
            'examples': examples,
            'techniques': techniques
        }
    
    def get_retrievers(self, k_examples: int = 3, k_techniques: int = 2):
        therapeutic_retriever = self.therapeutic_store.get_langchain_retriever(k=k_examples)
        cbt_retriever = self.cbt_store.get_langchain_retriever(k=k_techniques)
        
        return therapeutic_retriever, cbt_retriever

    def get_stats(self) -> Dict[str, Any]:
        return {
            'therapeutic_examples': {
                'count': self.therapeutic_store.size(),
                'dimension': self.therapeutic_store.dimension()
            },
            'cbt_techniques': {
                'count': self.cbt_store.size(),
                'dimension': self.cbt_store.dimension()
            }
        }
    
if __name__ == "__main__":
    """
    Demo script showing LangChain FAISS usage.
    Run: python vectorstore.py
    """
    print("=" * 70)
    print("LangChain FAISS Vector Store Demo")
    print("=" * 70)
    
    data_dir = Path("../data/processed_langchain")
    therapeutic_dir = data_dir / "therapeutic_examples"
    cbt_dir = data_dir / "cbt_techniques"
    
    # Check if directories exist
    if not therapeutic_dir.exists():
        print("❌ LangChain FAISS directories not found!")
        print("   Please run build_index_langchain.py first")
        exit(1)
    
    # Initialize retriever
    retriever = TherapeuticRetriever(therapeutic_dir, cbt_dir)
    
    # Test queries
    test_queries = [
        "I feel very anxious about my job",
        "我感到很沮丧",  # Chinese
        "How do I deal with negative thoughts?"
    ]
    
    print("\n" + "=" * 70)
    print("Running Test Queries (LangChain)")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        results = retriever.retrieve(query, k_examples=2, k_techniques=1)
        
        print(f"\n   📚 Examples ({len(results['examples'])}):")
        for i, result in enumerate(results['examples'], 1):
            print(f"      {i}. Score: {result.score:.3f} | Lang: {result.metadata.get('language', 'N/A')}")
            print(f"         {result.text[:100]}...")
        
        print(f"\n   🧠 Techniques ({len(results['techniques'])}):")
        for i, result in enumerate(results['techniques'], 1):
            print(f"      {i}. Score: {result.score:.3f}")
            print(f"         {result.text[:100]}...")
    
    # Show stats
    print("\n" + "=" * 70)
    print("Vector Store Statistics")
    print("=" * 70)
    stats = retriever.get_stats()
    print(f"Therapeutic Examples: {stats['therapeutic_examples']['count']} vectors")
    print(f"CBT Techniques: {stats['cbt_techniques']['count']} vectors")
    
    print("\n" + "=" * 70)
    print("LangChain Integration")
    print("=" * 70)
    therapeutic_ret, cbt_ret = retriever.get_langchain_retrievers()
    print(f"✅ Therapeutic retriever: {type(therapeutic_ret).__name__}")
    print(f"✅ CBT retriever: {type(cbt_ret).__name__}")
    print("   These can be used directly in LangChain chains!")
    
    print("\n✅ LangChain FAISS demo complete!")
    print("\nKEY CHANGES:")
    print("  - Uses LangChain FAISS (not raw FAISS + pickle)")
    print("  - Search takes text directly (auto-embedding)")
    print("  - Integrated metadata filtering")
    print("  - Can return LangChain retrievers for chains")
