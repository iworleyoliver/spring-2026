# Fine-Tuning Alternatives

Your local environment doesn't have enough disk space for fine-tuning (~30GB+ for Mistral 7B + dependencies). Here are your options:

## Option 1: Use Google Colab (FREE, Recommended)

Google Colab provides free GPU and 100GB+ storage:

1. Go to https://colab.research.google.com
2. Create a new notebook
3. Paste this code:

```python
# Clone your repo
!git clone https://github.com/iworleyoliver/spring-2026.git
%cd spring-2026

# Install dependencies
!pip install -q torch transformers datasets peft bitsandbytes tqdm

# Run fine-tuning
!python3 finetune.py
```

4. Select GPU runtime: Runtime > Change runtime type > GPU
5. Run and watch the progress bars!
6. After training, download the `models/mistral-finetuned` folder and upload to your repo

## Option 2: Use Ollama Instead (Fast, Local)

Skip fine-tuning and use Ollama with Mistral:

```bash
# Install Ollama from ollama.ai
ollama serve

# In another terminal
ollama pull mistral

# Use with your assistant
streamlit run web_ui.py
```

Then in the web UI, select Ollama as your backend. No fine-tuning needed!

## Option 3: Clean Up More Space

Delete large directories:
```bash
rm -rf /usr/local/python/3.12.1/lib/python3.12/site-packages/*  # Careful!
docker system prune -a  # Clean Docker cache
```

**Not recommended** - will break the environment.

## Option 4: Use Smaller Model

Phi 2.5 is 1.6GB and runs on CPU:
```bash
python3 -c "from gpt4all import GPT4All; GPT4All('Phi 2.5')"
```

Then it's available without fine-tuning.

---

## Recommendation

**Use Google Colab** - it's free, has unlimited storage, and GPUs for fast training (15-30 minutes). Your fine-tuned model will be ready in no time!
