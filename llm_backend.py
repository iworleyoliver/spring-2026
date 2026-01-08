"""
Light wrapper for LLM backends. Supports:
1. Ollama (local open source models - preferred if running)
2. OpenRouter (cloud-based, requires API key as fallback)
3. Deterministic fallback (no dependencies)

Environment variables:
- OLLAMA_URL: Ollama server URL (default: http://localhost:11434)
- OLLAMA_MODEL: Ollama model name (default: mistral)
- OPENROUTER_API_KEY: OpenRouter API key (optional fallback)
- OPENROUTER_MODEL: OpenRouter model (default: openai/gpt-3.5-turbo)

Function:
- generate(prompt: str, context: str | None = None, model: str | None = None) -> str
"""
import os
import typing
import requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Ollama configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_ENDPOINT = f"{OLLAMA_URL}/api/generate"

# OpenRouter configuration (fallback)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
OPENROUTER_API_URL = "https://openrouter.io/api/v1/chat/completions"


def _check_ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def _generate_with_ollama(prompt: str, context: typing.Optional[str] = None, model: typing.Optional[str] = None) -> str:
    """Generate using local Ollama instance."""
    model_to_use = model or OLLAMA_MODEL
    full_prompt = prompt
    if context:
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
    
    try:
        payload = {
            "model": model_to_use,
            "prompt": full_prompt,
            "stream": False,
            "temperature": 0.2,
        }
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def _generate_with_openrouter(prompt: str, context: typing.Optional[str] = None, model: typing.Optional[str] = None, max_tokens: int = 512) -> str:
    """Generate using OpenRouter API."""
    model_to_use = model or OPENROUTER_MODEL
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/iworleyoliver/spring-2026",
            "X-Title": "Spring 2026",
        }
        
        payload = {
            "model": model_to_use,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return text
    except Exception as e:
        raise RuntimeError(f"OpenRouter request failed: {e}")


def generate(prompt: str, context: typing.Optional[str] = None, model: typing.Optional[str] = None, max_tokens: int = 512) -> str:
    """Generate a response using available LLM backend.
    
    Tries backends in order:
    1. Ollama (local - no API key needed)
    2. OpenRouter (cloud - requires OPENROUTER_API_KEY)
    3. Raises RuntimeError if neither is available
    """
    # Try Ollama first (local, no API key needed)
    if _check_ollama_available():
        try:
            return _generate_with_ollama(prompt, context, model)
        except Exception as e:
            # Fall through to OpenRouter
            pass
    
    # Try OpenRouter as fallback
    if OPENROUTER_API_KEY:
        try:
            return _generate_with_openrouter(prompt, context, model, max_tokens)
        except Exception as e:
            # Fall through to error
            pass
    
    raise RuntimeError(
        "No LLM backend available. Either:\n"
        "1. Install and run Ollama (ollama.ai), or\n"
        "2. Set OPENROUTER_API_KEY environment variable"
    )
