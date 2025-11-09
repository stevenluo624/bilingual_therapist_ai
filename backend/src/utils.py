import re
from typing import Tuple, Optional, Dict, Any


def validate_message(message: str, max_length: int = 5000) -> Tuple[bool, Optional[str]]:
    if not message or not message.strip():
        return False, "Message cannot be empty"
    
    if len(message) > max_length:
        return False, f"Message too long (max {max_length} characters)"
    
    # Check for suspicious patterns (potential injection attempt
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'eval\(',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return False, "Message contains invalid content"
    
    return True, None


def clean_text(text: str) -> str:
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def detect_language(text: str) -> str:
    # Count Chinese characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    
    # Count ASCII letters
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # If more than 30% Chinese characters, consider it Chinese
    total_chars = len(text)
    if total_chars == 0:
        return 'en'
    
    chinese_ratio = chinese_chars / total_chars
    
    if chinese_ratio > 0.3:
        return 'zh'
    else:
        return 'en'


def is_crisis_message(message: str) -> bool:
    message_lower = message.lower()
    
    # English crisis keywords
    en_crisis_keywords = [
        'suicide', 'kill myself', 'end my life', 'want to die',
        'self-harm', 'hurt myself', 'no point living'
    ]
    
    # Chinese crisis keywords
    zh_crisis_keywords = [
        '自杀', '自殺', '想死', '结束生命', '結束生命',
        '自残', '自殘', '伤害自己', '傷害自己', '活着没意义', '活著沒意義'
    ]
    
    all_keywords = en_crisis_keywords + zh_crisis_keywords
    
    for keyword in all_keywords:
        if keyword in message_lower:
            return True
    
    return False


def get_crisis_resources(language: str = 'en') -> Dict[str, Any]:
    if language == 'zh':
        return {
            'message': '如果您正在经历危机，请立即寻求专业帮助。',
            'hotlines': [
                {'name': '生命热线', 'number': '1-800-273-8255'},
                {'name': '危机短信热线', 'number': 'Text "HELLO" to 741741'}
            ],
            'emergency': '如果这是紧急情况，请拨打 911 或前往最近的急诊室。'
        }
    else:
        return {
            'message': 'If you are in crisis, please seek professional help immediately.',
            'hotlines': [
                {'name': 'National Suicide Prevention Lifeline', 'number': '1-800-273-8255'},
                {'name': 'Crisis Text Line', 'number': 'Text "HELLO" to 741741'}
            ],
            'emergency': 'If this is an emergency, please call 911 or go to your nearest emergency room.'
        }


def format_context_for_prompt(context_chunks: list, max_chunks: int = 5) -> str:
    if not context_chunks:
        return "No relevant context found."
    
    formatted = []
    for i, chunk in enumerate(context_chunks[:max_chunks], 1):
        # Handle both Document objects and dicts
        if hasattr(chunk, 'page_content'):
            content = chunk.page_content
        elif isinstance(chunk, dict):
            content = chunk.get('content', chunk.get('text', str(chunk)))
        else:
            content = str(chunk)
        
        formatted.append(f"[Context {i}]: {content}")
    
    return "\n\n".join(formatted)


def truncate_history(history: list, max_turns: int = 10) -> list:
    if len(history) <= max_turns:
        return history
    
    return history[-max_turns:]


def calculate_response_stats(response_text: str) -> Dict[str, int]:
    return {
        'char_count': len(response_text),
        'word_count': len(response_text.split()),
        'sentence_count': len(re.split(r'[.!?。！？]+', response_text))
    }
