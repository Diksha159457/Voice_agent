# Import JSON to parse model output
import json

# Import model config (centralized)
from config import MODEL_NAME

# Function that returns Groq client (you already have this in your code)
from utils.client import _get_client  


def detect_intent(text):
    """
    Detects user intent using LLM and returns structured JSON
    """
    try:
        # Call Groq LLM API
        response = _get_client().chat.completions.create(
            
            # Use centralized model name
            model=MODEL_NAME,

            # Conversation messages sent to LLM
            messages=[
                {
                    "role": "system",  # System prompt defines behavior
                    "content": """You are an intent classifier.
Return ONLY JSON like:
{"intent": "general_chat", "entities": {}}

Possible intents:
- general_chat
- open_app
- search
- reminder
- system_control
"""
                },
                {
                    "role": "user",  # Actual user input
                    "content": text
                }
            ],

            # Keep temperature 0 → deterministic output (important for JSON)
            temperature=0,

            # Limit tokens since output is small JSON
            max_tokens=100
        )

        # Extract model response text
        content = response.choices[0].message.content

        # Convert JSON string → Python dictionary
        return json.loads(content)

    except Exception as e:
        # Fallback if anything fails (VERY important for stability)
        return {
            "intent": "general_chat",  # default fallback
            "error": str(e)           # useful for debugging
        }