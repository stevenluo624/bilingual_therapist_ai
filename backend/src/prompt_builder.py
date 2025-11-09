from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Dict, List, Any, Optional
from utils import detect_language

# System message template
SYSTEM_TEMPLATE = """You are a compassionate, bilingual therapeutic chatbot trained in Cognitive Behavioral Therapy (CBT) principles.

Your role:
- Provide empathetic, supportive responses
- Use CBT techniques appropriately
- Never replace professional therapy
- Recognize crisis situations and provide resources

Your capabilities:
- Communicate fluently in English and Chinese
- Apply evidence-based therapeutic techniques
- Maintain appropriate boundaries
- Show warmth while remaining professional"""

# Main chat template
CHAT_TEMPLATE = """
LANGUAGE INSTRUCTION
You MUST respond in: {language_instruction}
The user is communicating in {language_full}, so match their language exactly.

RELEVANT CBT TECHNIQUES
Use these evidence-based techniques as guidance for your response:

{cbt_techniques}

EXAMPLE THERAPEUTIC CONVERSATIONS
These examples show the empathetic, supportive style you should emulate:

{example_conversations}

CONVERSATION HISTORY
{conversation_history}

CURRENT USER MESSAGE
User: {user_message}

{crisis_section}

RESPONSE INSTRUCTIONS
Now provide your response following these guidelines:

1. **Language**: Respond entirely in {language_instruction}
2. **CBT Integration**: Naturally incorporate relevant CBT techniques from above
3. **Style**: Mirror the empathetic, validating tone from the example conversations
4. **Brevity**: Keep responses concise but warm (2-4 paragraphs typically)
5. **Boundaries**: You're a supportive chatbot, not a replacement for professional therapy
6. **Validation**: Acknowledge the user's feelings before offering guidance
7. **Questions**: Ask thoughtful questions to encourage reflection when appropriate

Begin your response now:

Therapist:"""

class PromptBuilder:
    def __init__(self):
        self.chat_prompt = PromptTemplate(
            template=CHAT_TEMPLATE,
            input_variables=[
                "language_instruction",
                "language_full",
                "cbt_techniques",
                "example_conversations",
                "conversation_history",
                "user_message",
                "crisis_section",
            ]
        )

        system_message = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
        human_message = HumanMessagePromptTemplate.from_template(CHAT_TEMPLATE)

        self.chat_prompt_messages = ChatPromptTemplate.from_messages([
            system_message,
            human_message
        ])
    
    def build_prompt(self,
                    user_message: str,
                    language: str,
                    retrieved_examples: List[Any],
                    retrieved_techniques: List[Any],
                    conversation_history: Optional[List[Dict[str, str]]] = None,
                    include_crisis_check: bool = False) -> str:
        language_instruction = "English" if language == "en" else "中文 (Chinese)"
        language_full = detect_language(language)

        examples_text = self._format_examples(retrieved_examples, language)
        techniques_text = self._format_techniques(retrieved_techniques)
        history_text = self._format_history(conversation_history)
        crisis_section = self._format_crisis_section(language) if include_crisis_check else ""

        prompt = self.chat_prompt.format(
            language_instruction=language_instruction,
            language_full=language_full,
            cbt_techniques=techniques_text,
            example_conversations=examples_text,
            conversation_history=history_text,
            user_message=user_message,
            crisis_section=crisis_section
        )

        return prompt
    
    def build_prompt_messages(self, **kwargs) -> List[Dict[str, str]]:
        messages = self.chat_prompt_messages.format_prompt(**kwargs)
        return [
            {"role": msg.type, "content": msg.content}
            for msg in messages.to_messages()
        ]
    
    def _format_examples(self, examples: List[Any], preferred_language: str) -> str:
        if not examples:
            return "No similar conversations available."
        
        formatted = []
        for i, example in enumerate(examples, 1):
            lang = example.metadata.get('language', 'en')
            question = example.metadata.get('question', '')
            answer = example.metadata.get('answer', '')
            
            lang_tag = "[English]" if lang == "en" else "[中文]"
            
            formatted.append(f"""Example {i} {lang_tag}:
                Client: {question}
                Therapist: {answer}
                """)
        
        return '\n'.join(formatted)
    
    def _format_techniques(self, techniques: List[Any]) -> str:
        if not techniques:
            return "General CBT principles apply."
        
        formatted = []
        for i, technique in enumerate(techniques, 1):
            text = technique.text if hasattr(technique, 'text') else technique.page_content
            formatted.append(f"{i}. {text}")
        
        return '\n\n'.join(formatted)
    
    def _format_history(self, history: List[Dict[str, str]], max_turns: int = 5) -> str:
        if not history:
            return "This is the start of the conversation."
        
        recent = history[-max_turns:] if len(history) > max_turns else history
        
        formatted_turns = []
        for turn in recent:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            
            if role == 'user':
                formatted_turns.append(f"User: {content}")
            else:
                formatted_turns.append(f"Therapist: {content}")
        
        return '\n'.join(formatted_turns)
    
    def _build_crisis_section(self, language: str) -> str:
        """In the case of emergency, we build a fallback for user's who needs serious medical attention"""
        if language == 'zh':
            return f"""
                危机情况处理
                如果用户表达了自杀想法、自残意图或严重危机：
                1. 认真对待并表达关心
                2. 提供危机热线资源：
                - 希望热线: 400-161-9995
                - 紧急情况: 110 或 120
                3. 强烈建议寻求专业帮助
                4. 不要试图单独处理危机情况
            """
        else:
            return f"""
                CRISIS SITUATION PROTOCOL
                If the user expresses suicidal thoughts, self-harm intentions, or severe crisis:
                1. Take it seriously and express genuine concern
                2. Provide crisis resources:
                - National Suicide Prevention Lifeline: 988
                - Crisis Text Line: Text HOME to 741741
                - Emergency: 911
                3. Strongly encourage seeking professional help
                4. Do not attempt to handle crisis alone
            """

def get_prompt_template() -> PromptTemplate:
    builder = PromptBuilder()
    return builder.chat_prompt

def get_chat_prompt_tempalte() -> ChatPromptTemplate:
    builder = PromptBuilder()
    return builder.chat_prompt_messages

def build_chat_prompt(*args, **kwargs) -> str:
    builder = PromptBuilder()
    return builder.build_prompt(*args, **kwargs)

def validate_prompt(prompt: str, max_tokens: int = 30000) -> tuple[bool, Optional[str]]:
    if not prompt or not prompt.strip():
        return False, "Prompt is empty"
    
    estimated_tokens = len(prompt) // 4
    
    if estimated_tokens > max_tokens:
        return False, f"Prompt too long (~{estimated_tokens} tokens, max {max_tokens})"
    
    required_sections = ["LANGUAGE INSTRUCTION", "USER MESSAGE", "Therapist:"]
    for section in required_sections:
        if section not in prompt:
            return False, f"Missing required section: {section}"
    
    return True, None

if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class MockResult:
        text: str
        page_content: str
        score: float
        metadata: Dict[str, Any]
        index: int
    
    # Mock data
    mock_examples = [
        MockResult(
            text="Q: I feel anxious\nA: I understand...",
            page_content="Q: I feel anxious\nA: I understand...",
            score=0.15,
            metadata={'question': 'I feel anxious', 'answer': 'I understand...', 'language': 'en'},
            index=0
        )
    ]
    
    mock_techniques = [
        MockResult(
            text="Cognitive restructuring helps...",
            page_content="Cognitive restructuring helps...",
            score=0.25,
            metadata={'technique_name': 'Cognitive Restructuring'},
            index=0
        )
    ]
    
    # Initialize builder
    builder = PromptBuilder()
    
    # Build prompt
    prompt = builder.build_prompt(
        user_message="I feel anxious about my job",
        language="en",
        retrieved_examples=mock_examples,
        retrieved_techniques=mock_techniques,
        conversation_history=None,
        include_crisis_check=False
    )
    
    # Validate
    is_valid, error = validate_prompt(prompt)
    print(f"\n Validation: {'Passed' if is_valid else f'Failed - {error}'}")
    
    # Show preview
    print(prompt[:300] + "...")
    
    lc_template = get_prompt_template()
    print(f"   Template type: {type(lc_template).__name__}")
    print(f"   Input variables: {lc_template.input_variables}")