# Import centralized model name
from config import MODEL_NAME

# Import Groq client
from utils.client import _get_client  


def general_chat(text):
    """
    Handles normal conversation with the assistant
    """

    try:
        # Call LLM
        response = _get_client().chat.completions.create(

            # Use centralized model
            model=MODEL_NAME,

            # Messages sent to model
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],

            # Slight creativity
            temperature=0.7
        )

        # Return response
        return response.choices[0].message.content

    except Exception as e:
        # Never crash
        return f"⚠️ Error: {str(e)}"
def streaming_chat(text):
    """
    Streams response token-by-token (ChatGPT typing effect)
    """
    try:
        # Get Groq client
        client = _get_client()

        # Call API with streaming enabled
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ],
            stream=True  # 🔥 Enables streaming
        )

        full_response = ""  # Store final response

        # Loop through streamed chunks
        for chunk in response:

            # Extract partial text
            delta = chunk.choices[0].delta.content or ""

            # Print live (terminal output)
            print(delta, end="", flush=True)

            # Save complete response
            full_response += delta

        print()  # newline after completion

        return full_response
    except Exception as e:
        # Return a readable message to the UI instead of crashing the Flask route.
        return f"⚠️ Error: {str(e)}"
