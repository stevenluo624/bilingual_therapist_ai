from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
from google.cloud import firestore
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_firestore import FirestoreChatMessageHistory

# Load environment variables
load_dotenv()

# Setup Firestore Database
PROJECT_ID = "thera-86f04"
SESSION_ID = "user1_session" # dummy
COLLECTION_NAME = "chat_history"

# Firestore setup (commented out for now)
# client = firestore.Client(project=PROJECT_ID)
# chat_history = FirestoreChatMessageHistory(
#     session_id=SESSION_ID,
#     collection=COLLECTION_NAME,
#     client=client,
# )
# use chat_history.add_ai_message() and add_user_message() to log messages

# Initialize FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, vectorstore, embeddings, prompts

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
        convert_system_message_to_human=True  # Gemini requirement
    )
    print("✅ Gemini LLM initialized (FREE API!)")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    try:
        vectorstore = FAISS.load_local(
            "vectorstore/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except TypeError:
        vectorstore = FAISS.load_local(
            "vectorstore/faiss_index",
            embeddings
        )

    with open('data/sample_prompts.json', 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    try:
        yield
    finally:
        sessions.clear()


app = FastAPI(title="CBT Chatbot API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    language: str = "en"  # 'en' or 'zh'

class ChatResponse(BaseModel):
    response: str
    technique_used: Optional[Dict] = None
    language: str

# Global variables
llm = None
vectorstore = None
embeddings = None
sessions = {}  # Store conversation memories
prompts = {}

def get_or_create_session(session_id: str) -> ConversationBufferMemory:
    """Get existing session memory or create new one"""
    if session_id not in sessions:
        sessions[session_id] = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
    return sessions[session_id]

def detect_language(text: str) -> str:
    """Simple language detection based on character set"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return 'zh' if chinese_chars > len(text) * 0.3 else 'en'

@app.get("/")
async def root():
    return {
        "message": "CBT Chatbot API - Powered by FREE Google Gemini", 
        "endpoints": ["/chat", "/health", "/techniques"],
        "model": "gemini-2.0-flash"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "llm": "Gemini Pro",
        "llm_loaded": llm is not None,
        "vectorstore_loaded": vectorstore is not None,
        "embeddings": "HuggingFace (FREE)"
    }

@app.get("/techniques")
async def get_techniques():
    """Get all available CBT techniques"""
    with open('data/cbt_techniques.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['techniques']

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with CBT techniques - Using FREE Gemini"""
    try:
        # Auto-detect language if not specified
        if request.language == "auto":
            request.language = detect_language(request.message)
        
        # Get or create session memory
        memory = get_or_create_session(request.session_id)
        
        # Search for relevant CBT technique
        search_results = vectorstore.similarity_search_with_score(
            request.message,
            k=1,
            filter={"language": request.language}
        )
        
        technique = None
        technique_context = ""
        
        if search_results and search_results[0][1] < 0.5:  # Relevance threshold
            technique_doc = search_results[0][0]
            technique = technique_doc.metadata
            technique_context = f"""
            
            Apply this CBT Technique in your response:
            - Name: {technique['name']}
            - Description: {technique['description']}
            - Example: {technique['example']}
            """
        
        # Create prompt
        system_prompt = (prompts['system_prompt_zh'] if request.language == 'zh' 
                        else prompts['system_prompt_en'])
        
        # Combine system prompt with technique context
        full_system_prompt = system_prompt + technique_context
        
        # Create prompt template that includes history
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("human", full_system_prompt + "\n\nUser: {input}\nTherapist:"),
        ])

        # Compose chain manually to control memory input keys
        chain = prompt | llm

        chain_inputs = {
            "history": memory.load_memory_variables({}).get("history", []),
            "input": request.message,
        }

        llm_result = chain.invoke(chain_inputs)
        response = getattr(llm_result, "content", llm_result)
        memory.save_context({"input": request.message}, {"output": response})
        
        return ChatResponse(
            response=response,
            technique_used=technique,
            language=request.language
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    if session_id in sessions:
        del sessions[session_id]
    return {"message": f"Session {session_id} cleared"}

# Simple test endpoint
@app.get("/test")
async def test():
    """Test endpoint to verify Gemini works"""
    try:
        # Simple test without retrieval
        test_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        test_result = test_llm.invoke("Say 'Gemini is working!' if you can see this.")
        response = getattr(test_result, "content", test_result)
        return {
            "status": "success",
            "response": response,
            "message": "Gemini API is working correctly!"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "tip": "Check your GOOGLE_API_KEY in .env file"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)