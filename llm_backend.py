"""
Light wrapper for LLM backends. Supports (in order of preference):
1. GPT4All (built-in, local, no external service - downloads model automatically)
2. Ollama (local open source models - requires manual setup)
3. OpenRouter (cloud-based, requires API key)

Environment variables:
- OLLAMA_URL: Ollama server URL (default: http://localhost:11434)
- OLLAMA_MODEL: Ollama model name (default: mistral)
- OPENROUTER_API_KEY: OpenRouter API key (optional fallback)
- OPENROUTER_MODEL: OpenRouter model (default: openai/gpt-3.5-turbo)
- GPT4ALL_MODEL: GPT4All model name (default: Mistral 7B)

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

try:
    from gpt4all import GPT4All
    _gpt4all_available = True
except ImportError:
    _gpt4all_available = False

# GPT4All configuration (built-in, no external service)
# Smaller, faster models: "Phi 2.5", "Mistral 7B", "Neural Chat 7B"
# For smarter responses, use Mistral 7B (takes ~20-40 seconds but much better quality)
GPT4ALL_MODEL = os.environ.get("GPT4ALL_MODEL", "Mistral 7B")

# Fine-tuned model path (if available)
FINETUNE_MODEL_PATH = os.environ.get("FINETUNE_MODEL_PATH", "./models/mistral-finetuned")
_finetune_available = os.path.exists(FINETUNE_MODEL_PATH)

# Ollama configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_ENDPOINT = f"{OLLAMA_URL}/api/generate"

# OpenRouter configuration (fallback)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
OPENROUTER_API_URL = "https://openrouter.io/api/v1/chat/completions"

# Global GPT4All instance (lazy-loaded)
_gpt4all_instance = None


def _get_gpt4all_instance():
    """Lazy-load GPT4All instance (downloads model on first use)."""
    global _gpt4all_instance
    if _gpt4all_instance is None and _gpt4all_available:
        try:
            _gpt4all_instance = GPT4All(GPT4ALL_MODEL)
        except Exception:
            pass
    return _gpt4all_instance


def _generate_with_gpt4all(prompt: str, context: typing.Optional[str] = None, model: typing.Optional[str] = None) -> str:
    """Generate using built-in GPT4All (no external service needed) with better prompting."""
    import threading
    
    model_to_use = model or GPT4ALL_MODEL
    
    # Build intelligent prompt with system instructions and context
    system_prompt = """You are a highly capable coding and development assistant with expertise in:
- Python, JavaScript, and multiple programming languages
- Software architecture and design patterns
- Debugging and problem-solving
- Technical writing and documentation
- Data analysis and algorithms

When responding:
1. Be direct and practical
2. Provide code examples when relevant
3. Explain your reasoning
4. Ask clarifying questions if needed
5. Consider edge cases and best practices"""
    
    if context:
        full_prompt = f"""{system_prompt}

## Context Information:
{context}

## User Question:
{prompt}

Please provide a thoughtful, detailed response that leverages the context above."""
    else:
        full_prompt = f"""{system_prompt}

## User Question:
{prompt}

Please provide a helpful response."""
    
    try:
        instance = _get_gpt4all_instance()
        if instance is None:
            raise RuntimeError("GPT4All not available")
        
        result = [None]
        exception = [None]
        
        def generate_with_timeout():
            try:
                result[0] = instance.generate(
                    full_prompt, 
                    max_tokens=512,  # Longer for better quality
                    temp=0.2,  # Lower temp for more focused responses
                    top_p=0.9,
                    top_k=50,
                    n_batch=128,
                )
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=generate_with_timeout, daemon=True)
        thread.start()
        thread.join(timeout=45)  # 45 second timeout for better quality
        
        if exception[0]:
            raise exception[0]
        if result[0] is None:
            raise RuntimeError("Generation timeout - model is taking too long.")
        
        return result[0].strip()
    except Exception as e:
        raise RuntimeError(f"GPT4All request failed: {e}")


def _generate_with_finetune(prompt: str, context: typing.Optional[str] = None) -> str:
    """Generate using fine-tuned Mistral model."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            FINETUNE_MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
        )
        
        # Build prompt
        if context:
            full_prompt = f"""You are a coding assistant specialized in this project.

{context}

User: {prompt}
Assistant:"""
        else:
            full_prompt = f"User: {prompt}\nAssistant:"
        
        # Tokenize and generate
        inputs = tokenizer(full_prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    except Exception as e:
        raise RuntimeError(f"Fine-tuned model failed: {e}")


def _check_ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def _generate_with_ollama(prompt: str, context: typing.Optional[str] = None, model: typing.Optional[str] = None) -> str:
    """Generate using local Ollama instance with structured prompting."""
    model_to_use = model or OLLAMA_MODEL
    
    # Build structured prompt with context
    if context:
        full_prompt = f"""You are a helpful coding and development assistant.

{context}

User Question: {prompt}

Please provide a clear, practical response based on the context above."""
    else:
        full_prompt = prompt
    
    try:
        payload = {
            "model": model_to_use,
            "prompt": full_prompt,
            "stream": False,
            "temperature": 0.3,
        }
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def _generate_with_openrouter(prompt: str, context: typing.Optional[str] = None, model: typing.Optional[str] = None, max_tokens: int = 512) -> str:
    """Generate using OpenRouter API with better system prompting."""
    model_to_use = model or OPENROUTER_MODEL
    messages = [
        {
            "role": "system",
            "content": "You are a helpful coding and development assistant. Provide clear, practical responses. Be concise but thorough."
        }
    ]
    
    if context:
        messages.append({"role": "system", "content": f"Context about recent work:\n{context}"})
    
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
            "temperature": 0.3,
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
    1. Fine-tuned Mistral (if available - best for your project)
    2. GPT4All (built-in - downloads model automatically)
    3. Ollama (local - no API key needed)
    4. OpenRouter (cloud - requires OPENROUTER_API_KEY)
    """
    # Try fine-tuned model first (if available)
    if _finetune_available:
        try:
            return _generate_with_finetune(prompt, context)
        except Exception as e:
            print(f"Fine-tuned model unavailable, falling back: {e}")
    
    # Try GPT4All (built-in, automatic model download)
    if _gpt4all_available:
        try:
            return _generate_with_gpt4all(prompt, context, model)
        except Exception as e:
            # Fall through to Ollama
            pass
    
    # Try Ollama (local, no API key needed)
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
        "No LLM backend available. To enable AI features:\n"
        "1. Fine-tune with: python3 finetune.py (best!)\n"
        "2. Or install GPT4All: pip install gpt4all\n"
        "3. Or install Ollama from ollama.ai\n"
        "4. Or set OPENROUTER_API_KEY environment variable"
    )
