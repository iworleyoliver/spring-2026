# User Guide - For Non-Developers

This guide is for people who just want to **use** the assistant, not code.

## What This Does

You get a **personal AI assistant** that:
- ✅ Answers your questions in real-time
- ✅ Analyzes documents (PDF, text, code)
- ✅ Generates files from descriptions
- ✅ Remembers your conversation history
- ✅ Runs entirely on your computer (private)

## Installation (One-Time Setup)

### Step 1: Download the Code
Download the project from GitHub:
```
https://github.com/iworleyoliver/spring-2026
```

Click **Code > Download ZIP**, then unzip it.

### Step 2: Install Python
1. Go to https://www.python.org/downloads/
2. Download Python 3.9 or newer
3. **Important:** Check "Add Python to PATH" during installation
4. Verify it worked: Open Command Prompt and type `python --version`

### Step 3: Open Folder in VS Code
1. Download VS Code: https://code.visualstudio.com/
2. Open VS Code
3. File > Open Folder > Select the `spring-2026` folder

### Step 4: Run Setup
1. In VS Code, open Terminal (Ctrl+`)
2. Copy and paste this:
```bash
bash setup.sh
```
3. Wait for "Setup complete!" message

## Using the Assistant

### Start the Assistant
In VS Code terminal:
```bash
streamlit run web_ui.py
```

Wait for the message: **"You can now view your Streamlit app in your browser"**

Then open: http://localhost:8501

### Using the Chat

**💬 Chat Tab:**
- Type a question
- Click "Send"
- AI responds

Examples:
- "What is machine learning?"
- "Write a poem about cats"
- "Summarize this document"

**📄 Analyze File Tab:**
- Click "Choose a file to analyze"
- Select a PDF, text file, or document
- Click "Analyze"
- The AI learns about your file

Then ask about it in Chat! Like:
- "What was in that document?"
- "Summarize the key points"

**✨ Generate File Tab:**
- Type filename: `shopping_list.txt`
- Description: "Create a weekly grocery shopping list"
- Click "Generate"
- Your file is created!

**💾 Memory Tab:**
- See all conversations
- See all analyzed files
- View your history

**❓ Help Tab:**
- Read full guide with tips

## Model Speed

In the sidebar, you can choose:

- **Smart (Mistral 7B)** - Best answers (20-45 seconds)
- **Fast (Phi 2.5)** - Instant answers (5-10 seconds)
- **⚡ Fast Mode** - Super fast (instant, simple answers)

Try Fast Mode first if you want instant responses!

## Common Questions

**Q: Is my data private?**
A: Yes! Everything runs on your computer. Nothing goes to the internet.

**Q: Can I close it and reopen later?**
A: Yes! Just run `streamlit run web_ui.py` again. Your conversations are saved.

**Q: What if I get an error?**
A: 
- Close the browser tab
- Stop the terminal (Ctrl+C)
- Run `streamlit run web_ui.py` again

**Q: Can I use this offline?**
A: Yes, after the first run when the model downloads.

**Q: How do I stop it?**
A: In VS Code terminal, press Ctrl+C

## Tips for Best Results

1. **Be specific:** Instead of "tell me about AI", try "explain neural networks in simple terms"

2. **Analyze first:** Upload a document, then ask questions about it

3. **Use context:** If the AI isn't getting it, provide more details

4. **Check Fast Mode:** If waiting is annoying, toggle ⚡ Fast Mode

## Troubleshooting

**"command not found: streamlit"**
- Run `bash setup.sh` again
- Make sure you're in the `spring-2026` folder

**"ModuleNotFoundError"**
- Run `bash setup.sh` again
- Close and reopen VS Code

**Very slow responses**
- Switch to "Fast (Phi 2.5)" model
- Or enable ⚡ Fast Mode

**Out of memory**
- Close other apps
- Switch to smaller model

## Need Help?

See the files in the repo:
- `README.md` - Technical overview
- `FINETUNE.md` - Advanced training (optional)
- `FINETUNING_ALTERNATIVES.md` - If you want better AI

---

**That's it!** You now have a personal AI assistant. Enjoy! 🎉
