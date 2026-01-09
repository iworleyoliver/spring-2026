#!/bin/bash
# Quick setup script for spring-2026 assistant

echo "🚀 Setting up spring-2026 assistant..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Quick start:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Start web UI: streamlit run web_ui.py"
echo "  3. Or use CLI: python3 project_tool.py assistant-chat 'Hello!'"
echo ""
echo "Optional (for fine-tuning):"
echo "  pip install torch transformers datasets peft bitsandbytes"
echo "  python3 finetune.py"
echo ""
