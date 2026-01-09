# spring-2026
coding python

here you go ms. Carter

## LLM Setup

The assistant supports multiple LLM backends (tries them in order):

### Option 1: GPT4All (Recommended - Built-in, Auto-Download, No Setup)

The easiest way to get started. GPT4All automatically downloads a model on first use.

```bash
pip install -r requirements.txt
python3 project_tool.py assistant-chat "Hello!"  # Downloads Mistral 7B on first run
```

That's it! No API keys, no external services needed.

### Option 2: Ollama (Local, Open Source, No API Key)

For more control or using different models:

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

### Option 3: OpenRouter (Cloud, Requires API Key)

For more powerful models in the cloud:

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

### Use the Assistant

**Option 1: Web GUI (Recommended)**
```bash
streamlit run web_ui.py
```
Opens an interactive web interface in your browser at `http://localhost:8501`

**Option 2: Command Line**
```bash
python3 project_tool.py assistant-analyze-add README.md
python3 project_tool.py assistant-chat "Summarize recent analyses"
```

Be careful not to commit `.env` with secrets to your repository.