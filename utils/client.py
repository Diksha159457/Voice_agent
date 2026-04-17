import os

from groq import Groq


_client = None


def _get_client() -> Groq:
    """Return a shared Groq client instance."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your environment or .env file."
            )
        _client = Groq(api_key=api_key)
    return _client
