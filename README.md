# Spring 2026 - Local AI Assistant

An AI-powered assistant that runs entirely on your computer. No cloud services, no API keys (unless you choose to use them).

**👤 Not a developer?** → See [USER_GUIDE.md](USER_GUIDE.md) for step-by-step instructions

## ✨ Features

- 💬 **Chat** with an AI that remembers your conversation
- 📄 **Analyze** documents, PDFs, and code files  
- ✨ **Generate** content from text descriptions
- 🔐 **Fully Private** - Everything runs locally

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Web Interface (Recommended)

```bash
streamlit run web_ui.py
```

Opens at `http://localhost:8501`

### Run from Command Line

```bash
python3 project_tool.py assistant-chat "Hello!"
python3 project_tool.py assistant-analyze-add README.md
```

## 🤖 LLM Backend Options

### GPT4All (Default - Recommended)

Easiest option. Automatically downloads on first use, no setup required.

```bash
python3 project_tool.py assistant-chat "Hello!"  # Downloads model on first run
```

- ✅ Works offline
- ✅ No API key needed
- ✅ Auto-downloads Mistral 7B (~4GB)

### Ollama (Local, Open Source)

For more control or different models:

```bash
ollama serve          # Start Ollama
ollama pull mistral   # Pull a model
export OLLAMA_MODEL="mistral"
```

### OpenRouter (Cloud - Optional)

For more powerful models:

```bash
echo "OPENROUTER_API_KEY=your-key-here" > .env
export OPENROUTER_MODEL="openai/gpt-4"
```

Get an API key at [openrouter.io](https://openrouter.io)

⚠️ **Never commit `.env` with API keys to GitHub**