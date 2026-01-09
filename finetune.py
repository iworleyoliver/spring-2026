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
from tqdm import tqdm

# Check if required packages are installed
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import Dataset
    import torch
except ImportError:
    print("Installing required packages for fine-tuning...")
    os.system("pip install -q torch transformers datasets bitsandbytes peft unsloth tqdm")
    print("Packages installed. Run the script again.")
    exit(1)


def collect_training_data() -> List[str]:
    """Collect code and documentation from the project."""
    training_texts = []
    
    # Collect Python files
    py_files = glob.glob("*.py")
    py_files = [f for f in py_files if f != "finetune.py"]
    
    for py_file in tqdm(py_files, desc="Loading Python files", unit="file"):
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                training_texts.append(f"## File: {py_file}\n{content}")
        except Exception as e:
            print(f"✗ Error loading {py_file}: {e}")
    
    # Collect markdown files
    md_files = glob.glob("*.md")
    for md_file in tqdm(md_files, desc="Loading documentation", unit="file"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                training_texts.append(f"## Documentation: {md_file}\n{content}")
        except Exception as e:
            print(f"✗ Error loading {md_file}: {e}")
    
    # Collect conversation history from memory
    try:
        with open(".assistant_memory.json", "r") as f:
            memory = json.load(f)
            messages = memory.get("messages", [])
            for msg in tqdm(messages, desc="Loading conversation history", unit="msg"):
                if msg.get("role") == "user":
                    training_texts.append(f"User: {msg.get('text')}")
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
    
    print("\n" + "=" * 70)
    print("🚀 MISTRAL 7B FINE-TUNING PIPELINE")
    print("=" * 70 + "\n")
    
    # Step 1: Collect data
    print("📚 STEP 1: Collecting Training Data")
    print("-" * 70)
    texts = collect_training_data()
    if not texts:
        print("❌ No training data found. Make sure you have .py and .md files.")
        return
    print(f"✓ Collected {len(texts)} text sources\n")
    
    # Step 2: Create dataset
    print("🔧 STEP 2: Creating Training Dataset")
    print("-" * 70)
    dataset = create_dataset(texts)
    dataset = dataset.train_test_split(test_size=0.1)
    print(f"✓ Train: {len(dataset['train'])} samples")
    print(f"✓ Test: {len(dataset['test'])} samples\n")
    
    # Step 3: Load model and tokenizer
    print("⚙️  STEP 3: Loading Mistral 7B")
    print("-" * 70)
    model_name = "mistralai/Mistral-7B-v0.1"
    
    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Loaded tokenizer")
        
        # For efficient fine-tuning, use 4-bit quantization
        from transformers import BitsAndBytesConfig
        
        print("Configuring 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        print("Downloading model (~15GB, this may take a few minutes)...")
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
    print("🔤 STEP 4: Tokenizing Dataset")
    print("-" * 70)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
    
    print("Tokenizing training examples...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        desc="Tokenizing",
    )
    print(f"✓ Tokenized {len(tokenized_datasets['train'])} training examples\n")
    
    # Step 5: Fine-tune with LoRA (efficient)
    print("🎯 STEP 5: Applying LoRA (Low-Rank Adaptation)")
    print("-" * 70)
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
        print(f"✓ LoRA configured (efficient training, smaller file size)\n")
    except ImportError:
        print("⚠️  PEFT not available, using full fine-tuning (slower/larger)\n")
    
    # Step 6: Training with progress bar
    print("🏃 STEP 6: FINE-TUNING (3 epochs)")
    print("-" * 70)
    
    training_args = TrainingArguments(
        output_dir="./models/mistral-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_steps=10,
        eval_steps=10,
        logging_steps=5,
        learning_rate=2e-4,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        warmup_steps=100,
        report_to=[],  # Disable wandb
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )
    
    try:
        print("\nTraining progress:\n")
        trainer.train()
        print("\n✅ Fine-tuning complete!\n")
        
        # Save the model
        print("💾 STEP 7: Saving Fine-Tuned Model")
        print("-" * 70)
        model.save_pretrained("./models/mistral-finetuned")
        tokenizer.save_pretrained("./models/mistral-finetuned")
        print("✓ Model saved to ./models/mistral-finetuned\n")
        
        # Update environment
        os.environ["FINETUNE_MODEL_PATH"] = "./models/mistral-finetuned"
        
        print("=" * 70)
        print("✅ SUCCESS! Fine-tuned Model Ready")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Restart the web UI: streamlit run web_ui.py")
        print("2. Your custom model will be used automatically")
        print("3. It understands your specific codebase and patterns\n")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("You may need more GPU memory or to reduce batch size")
        print("See FINETUNE.md for troubleshooting tips")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🤖 MISTRAL 7B FINE-TUNING ON YOUR PROJECT")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Collect your code, docs, and conversation history")
    print("  2. Download Mistral 7B (~15GB)")
    print("  3. Fine-tune for 3 epochs (~30-60 min on GPU)")
    print("  4. Save custom model to ./models/mistral-finetuned")
    print("=" * 70 + "\n")
    
    finetune_mistral()
