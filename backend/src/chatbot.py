import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

# LangChain imports - NEW
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Our modules
from embeddings import get_embedder
from vectorstore import TherapeuticRetriever
from prompt_builder import PromptBuilder, get_prompt_template
from utils import (
    detect_language,
    validate_message,
    is_crisis_message,
    get_crisis_resources,
    clean_text
)

@dataclass
class ChatbotConfig:
    """
    Configuration for the chatbot.
    UNCHANGED: Same config structure
    """
    # Retrieval parameters
    k_examples: int = 3
    k_techniques: int = 2
    use_language_filter: bool = False
    
    # Generation parameters
    temperature: float = 0.7
    max_output_tokens: int = 1000
    top_p: float = 0.9
    top_k: int = 40
    
    # Conversation parameters
    max_history_turns: int = 5
    
    # Safety parameters
    check_crisis: bool = True
    
    # API parameters
    gemini_model: str = "gemini-2.5-pro"
    
    # Debug
    debug_mode: bool = False

@dataclass
class ChatSession:
    session_id: str
    language: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _memory: Optional[ConversationBufferMemory] = None

def get_langchain_memory(self) -> ConversationBufferMemory:
        if self._memory is None:
            self._memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Load existing history into memory
            for turn in self.history:
                if turn['role'] == 'user':
                    self._memory.chat_memory.add_user_message(turn['content'])
                else:
                    self._memory.chat_memory.add_ai_message(turn['content'])
        
        return self._memory
    
def add_turn(self, role: str, content: str):
    self.history.append({
        'role': role,
        'content': content
    })
    
    if self._memory is not None:
        if role == 'user':
            self._memory.chat_memory.add_user_message(content)
        else:
            self._memory.chat_memory.add_ai_message(content)

def get_recent_history(self, max_turns: int = 5) -> List[Dict[str, str]]:
    return self.history[-max_turns:] if len(self.history) > max_turns else self.history

def clear_history(self):
    self.history = []
    if self._memory is not None:
        self._memory.clear()

class TherapeuticChatbot:
    '''The main chatbot'''
    def __init__(self,
                 data_dir: Path,
                 gemini_api_key: str,
                 config: Optional[ChatbotConfig] = None):
        self.config = config or ChatbotConfig()
        self._setup_model(gemini_api_key)
        self.embedder = get_embedder()
        therapeutic_dir = data_dir / "therapeutic_examples"
        cbt_dir = data_dir / "cbt_techniques"

        self.retriever = TherapeuticRetriever(therapeutic_dir, cbt_dir)
        self.prompt_builder = PromptBuilder()

        stats = self.retriever.get_stats()

    def _setup_model(self, api_key: str):
        if not api_key:
            raise ValueError("An API Key is Required")
        
        self.model = ChatGoogleGenerativeAI(
            model=self.config.gemini_model,
            google_api_key=api_key,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            convert_system_message_to_human=True  # Gemini compatibility
        )
    
    def create_chain(self, session: ChatSession) -> LLMChain:
        prompt_template = get_prompt_template()

        chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            memory=session.get_langchain_memory(),
            verbose=self.config.debug_mode
        )
        
        return chain
    
    def chat(self, user_message: str, session: ChatSession) -> Dict[str, Any]:
        is_valid, error = validate_message(user_message)
        if not is_valid:
            return self._error_response(error)
        
        user_message = clean_text(user_message)
        language = detect_language(user_message)
        session.language = language

        if self.config.debug_mode:
            print(f"Detected Language: {language}")
        
        is_crisis = False
        crisis_resources = None
        if self.config.check_crisis and is_crisis_message(user_message):
            is_crisis = True
            crisis_resources = get_crisis_resources(language)
            if self.config.debug_mode:
                print("Alert: crisis keyword detected")
        
        try:
            if self.config.use_language_filter:
                retrieval_results = self.retriever.retrieve(
                    user_message,
                    k_examples=self.config.k_examples,
                    k_techniques=self.config.k_techniques,
                    language_filter=language
                )
            else:
                retrieval_results = self.retriever.retrieve(
                    user_message,
                    k_examples=self.config.k_examples,
                    k_techniques=self.config.k_techniques
                )
            
            if self.config.debug_mode:
                print(f"Retrieved {len(retrieval_results['examples'])} Examples")
                print(f"Retrieved {len(retrieval_results['techniques'])} Techniques")
            
        except Exception as e:
            print(f"Retrieval failed: {e}")
            retrieval_results = {'examples': [], 'techniques': []}
        
        try:
            prompt = self.prompt_builder.build_prompt(
                user_message=user_message,
                language=language,
                retrieved_examples=retrieval_results['examples'],
                retrieved_techniques=retrieval_results['techniques'],
                conversation_history=session.get_recent_history(self.config.max_history_turns),
                include_crisis_check=is_crisis
            )
        except Exception as e:
            return self._error_response(f"Prompt building failed: {str(e)}")
        
        try: 
            response = self.llm.invoke(prompt)

            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            response_text = response_text.strip()
        
        except Exception as e:
            return self._error_response(f"Generation failed: {str(e)}")

        session.add_turn('user', user_message)
        session.add_turn('assistant', response_text)

        return {
            'response': response_text,
            'language': language,
            'is_crisis': is_crisis,
            'crisis_resources': crisis_resources,
            'metadata': {
                'num_examples_retrieved': len(retrieval_results.get('examples', [])),
                'num_techniques_retrieved': len(retrieval_results.get('techniques', [])),
                'session_id': session.session_id,
                'turn_count': len(session.history) // 2
            }
        }
    
    def chat_stream(self, prompt: str):
        for chunk in self.llm.stream(prompt):
            if hasattr(chunk, 'content'):
                yield chunk.content
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            'response': "I apologize, but I encountered an error. Please try again.",
            'language': 'en',
            'is_crisis': False,
            'crisis_resources': None,
            'metadata': {'error': error_message}
        }
    
    def create_session(self, session_id: str) -> ChatSession:
        return ChatSession(session_id=session_id)

if __name__ == "__main__":
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n Error: GEMINI_API_KEY environment variable not set")
        exit(1)
    
    # Check for LangChain data directory
    data_dir = Path("../data/processed_langchain")
    if not (data_dir / "therapeutic_examples").exists():
        print("\n Error: LangChain FAISS stores not found")
        print("   Run build_index.py first")
        exit(1)
    
    # Initialize chatbot
    config = ChatbotConfig(debug_mode=True)
    
    try:
        chatbot = TherapeuticChatbot(
            data_dir=data_dir,
            gemini_api_key=api_key,
            config=config
        )
    except Exception as e:
        print(f"\n❌Initialization failed: {e}")
        exit(1)
    
    # Create session
    session = chatbot.create_session("demo")
    
    # Test queries
    test_queries = [
        "I feel anxious about my job interview",
        "Thank you, that helps"
    ]
    for i, query in enumerate(test_queries, 1):
        print(f"Turn {i}")
        print(f"User: {query}")
        
        result = chatbot.chat(query, session)
        
        print(f"\nTherapist: {result['response']}")
        print(f"\nLanguage: {result['language']}")
