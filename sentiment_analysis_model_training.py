import torch_directml
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import torch
import sqlite3
import os
from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_fscore_support
import numpy as np
import sys

# Configure PyTorch to use AMD GPU
device = torch_directml.device()  # Use AMD GPU
model = model.to(device)  # Move model to AMD GPU

# Create models directory if it doesn't exist
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

def get_model_path(model_num):
    """Get the path for a specific model number"""
    return os.path.join(MODELS_DIR, f"sentiment_model_{model_num}")

def prepare_dataset():
    # Load data from your SQLite database
    conn = sqlite3.connect('mental_health_data.db')
    df = pd.read_sql_query("""
        SELECT r.response_text, r.sentiment_score
        FROM responses r
        WHERE r.sentiment_score IS NOT NULL
    """, conn)
    conn.close()
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    return dataset.train_test_split(test_size=0.2)

def train_sentiment_model(model_num):
    try:
        # Initialize GPU device
        device = torch_directml.device()
        print(f"🔧 Initializing with device: {device}")
        
        model_path = get_model_path(model_num)
        print(f"🎯 Training model {model_num}")
        print(f"📂 Model will be saved to: {model_path}")

        # Load BERT model for sequence classification
        print("📚 Loading BERT model...")
        model_name = "bert-base-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=1  # Regression task
        ).to(device) # Move model to GPU

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Prepare dataset
        print("🔄 Preparing dataset...")
        dataset = prepare_dataset()
        
        def tokenize_function(examples):
            return tokenizer(
                examples["response_text"],
                padding="max_length",
                truncation=True,
                max_length=128
            )
        
        # Tokenize with larger batch size for GPU
        def tokenize_function(examples):
            return tokenizer(
                examples["response_text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",  # Return PyTorch tensors
            )
        
        print("🔤 Tokenizing datasets...")
        tokenized_datasets = dataset.map(
            tokenize_function, 
            batched=True,
            batch_size=32  # Larger batch size for GPU
        )

        # Configure training arguments for GPU
        print("⚙️ Configuring training arguments...")
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,  # Increased for GPU
            per_device_eval_batch_size=16,   # Increased for GPU
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            no_cuda=True,  # Required for DirectML
            fp16=True,     # Use mixed precision
            gradient_accumulation_steps=2,  # Accumulate gradients
        )

        # Initialize Trainer
        print("🚀 Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
        )

        # Train model
        print("📈 Starting training...")
        print_gpu_utilization()  # Before training
        trainer.train()
        print_gpu_utilization()  # After training

        # Save model
        print("💾 Saving model...")
        print_gpu_utilization()  # Before saving
        trainer.save_model(model_path)
        tokenizer.save_pretrained("./sentiment_model")
        print_gpu_utilization()  # After saving

        print(f"✅ Model {model_num} saved successfully!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    finally:
        # Cleanup
        print("🧹 Cleaning up GPU memory...")
        if hasattr(model, 'to'):
            model.to('cpu')
        torch.cuda.empty_cache()

# Add this function to sentiment_analysis_model_training.py
def print_gpu_utilization():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        print(f"\nMemory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        print(f"GPU Memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    except:
        pass

def evaluate_model(model_path="./sentiment_model"):
    try:
        print("\n=== Model Evaluation ===")
        
        # Load model and tokenizer
        device = torch_directml.device()
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Prepare test dataset
        dataset = prepare_dataset()
        test_dataset = dataset['test']
        
        # Tokenize test data
        encoded_dataset = tokenizer(
            test_dataset['response_text'],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**encoded_dataset)
            predictions = outputs.logits.cpu().numpy()
        
        # Get actual values
        actual_scores = test_dataset['sentiment_score']
        
        # Calculate metrics
        mse = mean_squared_error(actual_scores, predictions)
        rmse = np.sqrt(mse)
        
        # Convert to binary classification for precision/recall
        # Assuming sentiment scores > 50 are positive
        binary_actual = [1 if score > 50 else 0 for score in actual_scores]
        binary_pred = [1 if pred > 50 else 0 for pred in predictions]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_actual, 
            binary_pred, 
            average='binary'
        )
        accuracy = accuracy_score(binary_actual, binary_pred)
        
        # Print results
        print("\n📊 Model Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        return None
    finally:
        if 'model' in locals():
            model.to('cpu')
        torch.cuda.empty_cache()

def load_model_for_prediction(model_path="./sentiment_model"):
    device = torch_directml.device()
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def predict_sentiment(text):
        # Tokenize input
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = outputs.logits.cpu().numpy()[0][0]
        
        return prediction
    
    return predict_sentiment

def list_available_models():
    """List all trained models in the models directory"""
    models = []
    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            if item.startswith("sentiment_model_"):
                models.append(item)
    return sorted(models)

def display_menu():
    print("\n=== Sentiment Analysis Model Training ===")
    print("1. Train Sentiment Model")
    print("2. Evaluate Model")
    print("3. Use Model for Prediction")
    print("4. List Available Models")
    print("0. Exit")
    return input("\nSelect an option: ")

def main():
    while True:
        choice = display_menu()

        match choice:
            case '1':
                model_num = input("Enter model number (e.g., 1, 2, 3): ")
                if not model_num.isdigit():
                    print("❌ Please enter a valid number")
                    continue
                
                # Check if model already exists
                model_path = get_model_path(model_num)
                if os.path.exists(model_path):
                    print(f"❌ Model {model_num} already exists at {model_path}")
                    print("Please choose a different model number")
                    continue
                    
                train_sentiment_model(model_num)
            case '2':
                models = list_available_models()
                if not models:
                    print("❌ No trained models available")
                    continue
                    
                print("\nAvailable models:")
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model}")
                    
                model_num = input("\nEnter model number to evaluate: ")
                if not model_num.isdigit() or not (0 < int(model_num) <= len(models)):
                    print("❌ Invalid model number")
                    continue
                    
                selected_model = models[int(model_num)-1]
                model_path = os.path.join(MODELS_DIR, selected_model)
                
                # Check if model files exist
                if not os.path.exists(model_path):
                    print(f"❌ Model files not found at {model_path}")
                    continue
                    
                if not os.path.exists(os.path.join(model_path, "config.json")):
                    print(f"❌ Model configuration not found for {selected_model}")
                    continue
                    
                evaluate_model(model_path)
            case '3':
                models = list_available_models()
                if not models:
                    print("❌ No trained models available")
                    continue
                print("\nAvailable models:")
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model}")
                model_num = input("\nEnter model number to use: ")
                if model_num.isdigit() and 0 < int(model_num) <= len(models):
                    model_path = os.path.join(MODELS_DIR, models[int(model_num)-1])
                    predictor = load_model_for_prediction(model_path)
                    while True:
                        text = input("\nEnter text to analyze (or 'q' to quit): ")
                        if text.lower() == 'q':
                            break
                        sentiment = predictor(text)
                        print(f"Sentiment score: {sentiment:.2f}")
                else:
                    print("❌ Invalid model number")
            case '4':
                models = list_available_models()
                if models:
                    print("\nAvailable models:")
                    for model in models:
                        print(f"  - {model}")
                else:
                    print("\nNo trained models found")
            case '0':
                print("Exiting...")
                sys.exit()
            case _:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()