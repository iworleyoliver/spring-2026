# spring-2026
coding python

here you go ms. Carter

## LLM Setup

The assistant supports multiple LLM backends (tries them in order):

### Option 1: Ollama (Recommended - Local, Open Source, No API Key)

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Start the Ollama server:
```bash
ollama serve
```
3. In another terminal, pull a model:
```bash
ollama pull mistral  # or llama2, neural-chat, etc.
```
4. The assistant will automatically use Ollama when available

Optional: Set a custom model:
```bash
export OLLAMA_MODEL="llama2"
export OLLAMA_URL="http://localhost:11434"
```

### Option 2: OpenRouter (Cloud, Requires API Key)

1. Get an API key at [openrouter.io](https://openrouter.io)
2. Create a `.env` file:
```bash
echo "OPENROUTER_API_KEY=your-openrouter-api-key-here" > .env
```
3. Optional: Set a custom model:
```bash
export OPENROUTER_MODEL="openai/gpt-4"
```

### Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Use the assistant CLI:

```bash
python3 project_tool.py assistant-analyze-add README.md
python3 project_tool.py assistant-chat "Summarize recent analyses"
```

Be careful not to commit `.env` with secrets to your repository.