import torch_directml
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,  # Changed from AutoModelForSequenceClassification
    AutoTokenizer,
    TrainingArguments,
    Seq2SeqTrainer,  # Changed from Trainer
    Seq2SeqTrainingArguments,  # Added for seq2seq tasks
    DataCollatorForSeq2Seq  # Added for seq2seq tasks
)
from datasets import Dataset
import torch
import sqlite3
import os
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

# Model constants
MODEL_NAME = "google/flan-t5-base"  # Changed from bert-base-uncased
MAX_INPUT_LENGTH = 512  # Increased for question + response
MAX_TARGET_LENGTH = 8   # For sentiment score output

def prepare_dataset():
    """Modified to include both question and response"""
    conn = sqlite3.connect('mental_health_data.db')
    df = pd.read_sql_query("""
        SELECT q.question_text, r.response_text, r.sentiment_score
        FROM responses r
        JOIN questions q ON r.question_id = q.question_id
        WHERE r.sentiment_score IS NOT NULL
    """, conn)
    conn.close()
    
    # Format data for T5
    df['input_text'] = 'Rate the sentiment: Question: ' + df['question_text'] + ' Response: ' + df['response_text']
    df['target_text'] = df['sentiment_score'].astype(str)  # Convert score to string
    
    dataset = Dataset.from_pandas(df[['input_text', 'target_text']])
    return dataset.train_test_split(test_size=0.2)

def compute_metrics(eval_pred):
    """Custom metrics computation for regression"""
    predictions, labels = eval_pred
    # Convert string predictions to numbers
    predictions = np.array([float(pred.strip()) for pred in predictions])
    labels = np.array([float(label.strip()) for label in labels])
    
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    
    # Binary classification metrics (for scores > 50)
    binary_preds = predictions > 50
    binary_labels = labels > 50
    accuracy = accuracy_score(binary_labels, binary_preds)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'accuracy': accuracy
    }

def train_sentiment_model(model_num):
    try:
        device = torch_directml.device()
        print(f"🔧 Initializing with device: {device}")
        
        model_path = get_model_path(model_num)
        print(f"🎯 Training model {model_num}")
        
        # Load Flan-T5 model
        print("📚 Loading Flan-T5 model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Prepare dataset
        dataset = prepare_dataset()
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100
        )
        
        # Configure training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=model_path,
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            predict_with_generate=True,
            no_cuda=True,  # For DirectML
            fp16=True,
            logging_dir=f'./logs/model_{model_num}',
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="mse"
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        # Train model
        print("📈 Starting training...")
        trainer.train()
        
        # Save model
        print("💾 Saving model...")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        print(f"✅ Model {model_num} saved successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    finally:
        if 'model' in locals():
            model.to('cpu')
        torch.cuda.empty_cache()

def predict_sentiment(model_path, question, response):
    """Modified prediction function for question-response pairs"""
    device = torch_directml.device()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    input_text = f"Rate the sentiment: Question: {question} Response: {response}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=MAX_TARGET_LENGTH)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        return float(prediction.strip())
    except ValueError:
        return None