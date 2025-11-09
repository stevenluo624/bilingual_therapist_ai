import json
import os
from pathlib import Path
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from tqdm import tqdm

# File paths
DATA_DIR = Path("../data")
CHINESE_JSON = DATA_DIR / "chinese_therapy.json"
ENGLISH_JSON = DATA_DIR / "english_therapy.json"
CBT_PDF = DATA_DIR / "cbt_manual.pdf"

# Output paths
OUTPUT_DIR = DATA_DIR / "processed_langchain"
THERAPEUTIC_DIR = OUTPUT_DIR / "therapeutic_examples"
CBT_DIR = OUTPUT_DIR / "cbt_techniques"

# Embedding model
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Chunking parameters for CBT manual
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def load_chinese_data(file_path: Path) -> List[Document]:
    
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    documents = []
    
    if isinstance(raw_data, dict):
        data_list = raw_data.get('data', raw_data.get('conversations', list(raw_data.values())[0]))
    else:
        data_list = raw_data
    
    for item in data_list:
        question = item.get('question', '')
        answers = item.get('answers', item.get('answer', []))
        label = item.get('label', 'general')
        
        if isinstance(answers, str):
            answers = [answers]
        
        for answer in answers:
            # Format as Q&A
            content = f"Question: {question}\nAnswer: {answer}"
            
            # Create Document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    'question': question,
                    'answer': answer,
                    'language': 'zh',
                    'label': label,
                    'source': 'chinese_dataset'
                }
            )
            documents.append(doc)
    
    return documents


def load_english_data(file_path: Path) -> List[Document]:

    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    documents = []
    
    if isinstance(raw_data, dict):
        data_list = raw_data.get('data', raw_data.get('conversations', list(raw_data.values())[0]))
    else:
        data_list = raw_data
    
    if len(data_list) > 0:
        print(f"   Sample item keys: {list(data_list[0].keys())}")
    
    for item in data_list:
        question = (item.get('Context') or item.get('context') or 
                   item.get('question') or item.get('Question') or '')
        answer = (item.get('Response') or item.get('response') or 
                 item.get('answer') or item.get('Answer') or '')
        
        if not question or not answer:
            continue
        
        # Format as Q&A
        content = f"Question: {question}\nAnswer: {answer}"
        
        doc = Document(
            page_content=content,
            metadata={
                'question': question,
                'answer': answer,
                'language': 'en',
                'label': 'general',
                'source': 'huggingface'
            }
        )
        documents.append(doc)
    
    print(f"✅ Created {len(documents)} English Documents")
    return documents


def extract_text_from_pdf(file_path: Path) -> str:

    try:
        reader = PdfReader(str(file_path))
        text_content = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
        
        full_text = "\n\n".join(text_content)
        return full_text
    
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return ""

def chunk_text_to_documents(text: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Try to split on paragraphs first
    )
    
    # Split and create Documents
    chunks = text_splitter.split_text(text)
    
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                'chunk_id': i,
                'source': 'cbt_manual',
                'language': 'en'
            }
        )
        documents.append(doc)
    
    return documents

def build_langchain_faiss(documents: List[Document], 
                         embeddings: HuggingFaceEmbeddings,
                         save_path: Path) -> FAISS:
   
    vector_store = FAISS.from_documents(
        documents,
        embeddings,
    )
    
    vector_store.save_local(str(save_path))
    return vector_store

def main():
    """
    Main execution pipeline.
    """
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    # Load both datasets as Documents
    chinese_docs = load_chinese_data(CHINESE_JSON)
    english_docs = load_english_data(ENGLISH_JSON)
    
    # Combine all documents
    all_therapeutic_docs = chinese_docs + english_docs
    
    therapeutic_store = build_langchain_faiss(
        all_therapeutic_docs,
        embeddings,
        THERAPEUTIC_DIR
    )
    
    # PROCESS CBT MANUAL
    cbt_text = extract_text_from_pdf(CBT_PDF)
    
    # Chunk into Documents using LangChain text splitter
    cbt_docs = chunk_text_to_documents(cbt_text)
    
    # Build LangChain FAISS store
    cbt_store = build_langchain_faiss(
        cbt_docs,
        embeddings,
        CBT_DIR
    )

if __name__ == "__main__":
    main()