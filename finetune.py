"""
Fine-tune Mistral 7B on your project codebase.
Run with: python3 finetune.py

This creates a specialized version of Mistral that understands your code.
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict

# Check if required packages are installed
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import Dataset
    import torch
except ImportError:
    print("Installing required packages for fine-tuning...")
    os.system("pip install -q torch transformers datasets bitsandbytes peft unsloth")
    print("Packages installed. Run the script again.")
    exit(1)


def collect_training_data() -> List[str]:
    """Collect code and documentation from the project."""
    training_texts = []
    
    # Collect Python files
    for py_file in glob.glob("*.py"):
        if py_file == "finetune.py":
            continue
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                training_texts.append(f"## File: {py_file}\n{content}")
                print(f"✓ Loaded {py_file}")
        except Exception as e:
            print(f"✗ Error loading {py_file}: {e}")
    
    # Collect markdown files
    for md_file in glob.glob("*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                training_texts.append(f"## Documentation: {md_file}\n{content}")
                print(f"✓ Loaded {md_file}")
        except Exception as e:
            print(f"✗ Error loading {md_file}: {e}")
    
    # Collect conversation history from memory
    try:
        with open(".assistant_memory.json", "r") as f:
            memory = json.load(f)
            # Create training examples from past conversations
            for msg in memory.get("messages", []):
                if msg.get("role") == "user":
                    training_texts.append(f"User: {msg.get('text')}")
            print(f"✓ Loaded conversation history ({len(memory.get('messages', []))} messages)")
    except Exception as e:
        print(f"Note: No conversation history yet ({e})")
    
    return training_texts


def create_dataset(texts: List[str]) -> Dataset:
    """Create a Hugging Face dataset from texts."""
    # Combine and split into chunks
    combined = "\n\n".join(texts)
    
    # Split into chunks of ~1000 characters
    chunk_size = 1000
    chunks = [combined[i:i+chunk_size] for i in range(0, len(combined), chunk_size)]
    
    # Create dataset
    data = {"text": chunks}
    dataset = Dataset.from_dict(data)
    
    print(f"✓ Created dataset with {len(chunks)} training examples")
    return dataset


def finetune_mistral():
    """Fine-tune Mistral 7B on project data."""
    
    print("🚀 Starting fine-tuning process...\n")
    
    # Step 1: Collect data
    print("📚 Step 1: Collecting training data...")
    texts = collect_training_data()
    if not texts:
        print("❌ No training data found. Make sure you have .py and .md files.")
        return
    print(f"✓ Collected {len(texts)} text sources\n")
    
    # Step 2: Create dataset
    print("🔧 Step 2: Creating training dataset...")
    dataset = create_dataset(texts)
    dataset = dataset.train_test_split(test_size=0.1)
    print(f"✓ Train: {len(dataset['train'])} samples")
    print(f"✓ Test: {len(dataset['test'])} samples\n")
    
    # Step 3: Load model and tokenizer
    print("⚙️  Step 3: Loading Mistral 7B...")
    model_name = "mistralai/Mistral-7B-v0.1"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Loaded tokenizer")
        
        # For efficient fine-tuning, use 4-bit quantization
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print(f"✓ Loaded model (quantized to 4-bit for efficiency)\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you have enough disk space (~15GB)")
        return
    
    # Step 4: Tokenize data
    print("🔤 Step 4: Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )
    print(f"✓ Tokenized {len(tokenized_datasets['train'])} training examples\n")
    
    # Step 5: Fine-tune with LoRA (efficient)
    print("🎯 Step 5: Applying LoRA for efficient fine-tuning...")
    try:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        print(f"✓ LoRA configured\n")
    except ImportError:
        print("⚠️  PEFT not available, using full fine-tuning (slower)\n")
    
    # Step 6: Training
    print("🏃 Step 6: Starting training (this may take 30-60 minutes)...")
    
    training_args = TrainingArguments(
        output_dir="./models/mistral-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        warmup_steps=100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )
    
    try:
        trainer.train()
        print("✅ Fine-tuning complete!\n")
        
        # Save the model
        print("💾 Saving fine-tuned model...")
        model.save_pretrained("./models/mistral-finetuned")
        tokenizer.save_pretrained("./models/mistral-finetuned")
        print("✓ Model saved to ./models/mistral-finetuned\n")
        
        # Update environment
        os.environ["FINETUNE_MODEL_PATH"] = "./models/mistral-finetuned"
        print("✅ Fine-tuned model ready to use!")
        print("Restart the web UI to use it: streamlit run web_ui.py")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("You may need more GPU memory or to reduce batch size")


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Mistral 7B Fine-Tuning on Your Project")
    print("=" * 60 + "\n")
    
    finetune_mistral()
