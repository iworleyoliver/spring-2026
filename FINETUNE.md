# Fine-Tuning Guide

## What is Fine-Tuning?

Fine-tuning adapts a pre-trained model (Mistral 7B) to learn from **your specific project**. This makes it:
- Understand your code structure and conventions
- Know your project patterns and style
- Answer questions about YOUR code specifically
- Specialized for your domain

## Quick Start

### 1. Install Fine-Tuning Dependencies

```bash
pip install torch transformers datasets peft bitsandbytes
```

Or with the optional requirements:
```bash
pip install -q torch transformers datasets bitsandbytes peft
```

### 2. Run Fine-Tuning

```bash
python3 finetune.py
```

This will:
- ✅ Collect all your `.py` and `.md` files
- ✅ Load conversation history from memory
- ✅ Download Mistral 7B (~15GB)
- ✅ Train for 3 epochs (~30-60 minutes on GPU)
- ✅ Save to `./models/mistral-finetuned`

### 3. Use the Fine-Tuned Model

Once training completes, restart the web GUI:
```bash
streamlit run web_ui.py
```

The assistant will automatically use your fine-tuned model (it's priority #1).

## What Gets Trained On?

The fine-tuning script collects:
- **All `.py` files** - Your codebase
- **All `.md` files** - Documentation
- **Conversation history** - Past questions you've asked
- **Project structure** - How files are organized

## Performance Tips

### GPU Memory
- **8GB+**: Full training, fast (30-60 min)
- **6GB**: May work with smaller batch size
- **<6GB**: Use Phi 2.5 instead (no training needed)

### Speed Up Training
Edit `finetune.py` line 145:
```python
num_train_epochs=1,  # Reduce from 3 to 1 (faster, less quality)
per_device_train_batch_size=2,  # Reduce from 4 (if out of memory)
```

### CPU Only (Very Slow)
Not recommended, but possible. Set in `finetune.py`:
```python
device_map="cpu"
```

## What's Better After Fine-Tuning?

**Before:**
```
User: How does the analyzer work?
Assistant: [Generic response about analyzing files]
```

**After:**
```
User: How does the analyzer work?
Assistant: The analyzer in analyzer.py extracts text from PDFs and files, 
then computes word frequencies, filtering out stopwords. It returns 
statistics including character count, line count, and top N words...
```

## Estimated Times

| GPU | Time | Quality |
|-----|------|---------|
| A100 | 15-20 min | Excellent |
| RTX 4090 | 25-35 min | Excellent |
| RTX 3080 | 45-60 min | Excellent |
| GTX 1080 | 2-3 hours | Excellent |
| CPU | 24+ hours | Excellent (too slow) |

## Check Training Progress

During training, look for:
```
[10/30] loss: 3.2 ✓
[20/30] loss: 2.8 ✓
[30/30] loss: 2.4 ✓
```

Loss should decrease as it learns.

## Common Issues

**"CUDA out of memory"**
- Reduce batch size in finetune.py: `per_device_train_batch_size=2`
- Or use Phi 2.5 (no training needed)

**"No training data found"**
- Make sure you have `.py` or `.md` files in your project
- Or add some with the assistant: "Create a Python script that..."

**"Slow training on CPU"**
- Expected, but not recommended
- Use Google Colab for free GPU: https://colab.research.google.com
- Or use a simpler model (Phi 2.5)

## Advanced: Continue Training

After initial fine-tuning, you can:
1. Add more data to your project
2. Have more conversations
3. Run `python3 finetune.py` again
4. It will improve with each iteration

## Questions?

The fine-tuning process is automatic and handles most edge cases. If you get an error, the script will explain what's wrong and suggest fixes.
