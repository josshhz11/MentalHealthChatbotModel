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
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

# Model path set up
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)

def get_model_path(model_num):
    """Get the path for a specific model number"""
    return os.path.join(MODELS_DIR, f"sentiment_model_{model_num}")

def list_available_models():
    """List all trained models in the models directory"""
    models = []
    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            if item.startswith("sentiment_model_"):
                models.append(item)
    return sorted(models)

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
        device = torch.device("cpu")
        print(f"🔧 Initializing with device: {device}")
        
        model_path = get_model_path(model_num)
        print(f"🎯 Training model {model_num}")
        
        # Load Flan-T5 model
        print("📚 Loading Flan-T5 model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Prepare dataset
        print("🔄 Preparing dataset...")
        dataset = prepare_dataset()
        
        # Tokenize the dataset
        print("🔤 Tokenizing dataset...")
        def preprocess_function(examples):
            inputs = examples["input_text"]
            targets = examples["target_text"]
            
            model_inputs = tokenizer(
                inputs, 
                max_length=MAX_INPUT_LENGTH,
                padding="max_length",
                truncation=True,
            )
            
            # Setup the targets
            labels = tokenizer(
                targets,
                max_length=MAX_TARGET_LENGTH,
                padding="max_length",
                truncation=True,
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Apply tokenization
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["input_text", "target_text"]  # Remove original columns
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=tokenizer.pad_token_id
        )
        
        # Configure training arguments
        """training_args = Seq2SeqTrainingArguments(
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
            metric_for_best_model="mse",
            remove_unused_columns=True
        )"""

        training_args = Seq2SeqTrainingArguments(
            per_device_train_batch_size=16,  # Can be larger on CPU
            gradient_accumulation_steps=1,   # No need with CPU
            num_train_epochs=3,
            learning_rate=1e-4,              # Slightly higher for CPU
            optim="adamw_torch",             # Optimized implementation
            dataloader_num_workers=4,        # Use multiple CPU cores
            dataloader_pin_memory=True       # Speed up data transfer
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
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
    
def evaluate_model(model_path):
    """Evaluate a trained model on the test dataset"""
    try:
        print(f"\n=== Evaluating Model: {os.path.basename(model_path)} ===")
        
        # Initialize device
        device = torch_directml.device()
        print(f"🔧 Using device: {device}")
        
        # Load model and tokenizer
        print("📚 Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Prepare dataset
        print("🔄 Preparing evaluation dataset...")
        dataset = prepare_dataset()
        test_dataset = dataset['test']
        
        # Process each sample
        print("🧪 Running evaluation...")
        predictions = []
        actual_scores = []
        
        for i, item in enumerate(test_dataset):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(test_dataset)} samples")
            
            # Get prediction
            input_text = item['input_text']
            inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_INPUT_LENGTH, 
                              truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=MAX_TARGET_LENGTH)
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                pred_score = float(prediction.strip())
                actual_score = float(item['target_text'])
                
                predictions.append(pred_score)
                actual_scores.append(actual_score)
            except ValueError:
                print(f"❌ Error parsing prediction: {prediction}")
        
        # Calculate metrics
        predictions = np.array(predictions)
        actual_scores = np.array(actual_scores)
        
        # Regression metrics
        mse = mean_squared_error(actual_scores, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_scores - predictions))
        
        # Classification metrics (using 50 as threshold)
        binary_pred = predictions > 50
        binary_actual = actual_scores > 50
        accuracy = accuracy_score(binary_actual, binary_pred)
        
        # Correlation
        correlation = np.corrcoef(predictions, actual_scores)[0, 1]
        
        # Print results
        print("\n📊 Evaluation Results:")
        print(f"Total samples: {len(predictions)}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Correlation: {correlation:.4f}")
        print(f"Binary Accuracy (>50): {accuracy:.4f}")
        
        # Score distribution
        print("\n📈 Prediction Distribution:")
        ranges = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
        for lower, upper in ranges:
            count = np.sum((predictions >= lower) & (predictions <= upper))
            percentage = (count / len(predictions)) * 100
            print(f"{lower}-{upper}: {count} predictions ({percentage:.1f}%)")
            
        # Error distribution
        errors = np.abs(predictions - actual_scores)
        print("\n⚠️ Error Distribution:")
        print(f"Errors within 5 points: {np.sum(errors <= 5) / len(errors) * 100:.1f}%")
        print(f"Errors within 10 points: {np.sum(errors <= 10) / len(errors) * 100:.1f}%")
        print(f"Errors within 20 points: {np.sum(errors <= 20) / len(errors) * 100:.1f}%")
        
        # Worst predictions
        print("\n❌ Worst Predictions:")
        worst_indices = np.argsort(errors)[-5:][::-1]
        for idx in worst_indices:
            sample = test_dataset[idx]
            print(f"Predicted: {predictions[idx]:.1f}, Actual: {actual_scores[idx]:.1f}, Error: {errors[idx]:.1f}")
            print(f"Input: {sample['input_text'][:100]}...\n")
            
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae, 
            'correlation': correlation,
            'accuracy': accuracy
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None
    finally:
        if 'model' in locals():
            model.to('cpu')
        torch.cuda.empty_cache()

def batch_predict_sentiment(model_path):
    """Predict sentiment for all unlabeled responses in the database"""
    try:
        print("🔄 Starting batch prediction...")
        
        # Connect to database
        conn = sqlite3.connect('mental_health_data.db')
        
        # Get unlabeled responses
        unlabeled_df = pd.read_sql_query("""
            SELECT r.response_id, q.question_text, r.response_text
            FROM responses r
            JOIN questions q ON r.question_id = q.question_id
            WHERE r.sentiment_score IS NULL
        """, conn)
        
        if unlabeled_df.empty:
            print("✅ No unlabeled responses found!")
            conn.close()
            return
            
        print(f"📊 Found {len(unlabeled_df)} unlabeled responses")
        choice = input(f"Do you want to process all {len(unlabeled_df)} responses? (y/n): ")
        
        if choice.lower() != 'y':
            limit = input("Enter number of responses to process: ")
            try:
                limit = int(limit)
                unlabeled_df = unlabeled_df.head(limit)
            except ValueError:
                print("❌ Invalid number, processing all responses")
        
        # Initialize tokenizer and model
        print("🔧 Loading model...")
        device = torch_directml.device()
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Process responses
        print("🚀 Starting prediction...")
        updated_count = 0
        
        for idx, row in unlabeled_df.iterrows():
            if idx % 10 == 0:
                print(f"Progress: {idx}/{len(unlabeled_df)} responses processed")
                
            try:
                # Predict sentiment
                input_text = f"Rate the sentiment: Question: {row['question_text']} Response: {row['response_text']}"
                inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=MAX_TARGET_LENGTH)
                    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                try:
                    sentiment = float(prediction.strip())
                    
                    # Update database
                    conn.execute(
                        "UPDATE responses SET sentiment_score = ? WHERE response_id = ?",
                        (int(round(sentiment)), row['response_id'])
                    )
                    updated_count += 1
                    
                    # Commit every 50 updates
                    if updated_count % 50 == 0:
                        conn.commit()
                        print(f"✅ Committed {updated_count} updates")
                        
                except ValueError:
                    print(f"❌ Could not parse prediction: {prediction}")
                    
            except Exception as e:
                print(f"❌ Error processing response {row['response_id']}: {e}")
                
        # Final commit
        conn.commit()
        
        # Update CSV files (optional - if you want to include the update_csv_files function)
        try:
            # Import update_csv_files from generate_dataset
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from generate_dataset import update_csv_files
            print("📊 Updating CSV files...")
            update_csv_files(conn)
        except ImportError:
            print("⚠️ Could not import update_csv_files function. CSV files not updated.")
            print("To update CSV files, run the 'Update DB from CSV' option in the generate_dataset.py script.")
        
        print(f"\n✅ Batch prediction complete!")
        print(f"- Total responses processed: {len(unlabeled_df)}")
        print(f"- Responses updated: {updated_count}")
        
    except Exception as e:
        print(f"❌ Error during batch prediction: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
        if 'model' in locals():
            model.to('cpu')
        torch.cuda.empty_cache()

def test_model_with_examples():
    """Test a trained model with example questions and responses"""
    models = list_available_models()
    
    if not models:
        print("❌ No trained models available")
        return
        
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
        
    model_num = input("\nSelect model to test (number): ")
    
    if not model_num.isdigit() or not (0 < int(model_num) <= len(models)):
        print("❌ Invalid model number")
        return
        
    selected_model = models[int(model_num)-1]
    model_path = os.path.join(MODELS_DIR, selected_model)
    
    if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"❌ Model files not found or incomplete: {model_path}")
        return
    
    print(f"\n🧪 Testing model: {selected_model}")
    
    # Connect to database
    conn = sqlite3.connect('mental_health_data.db')
    
    # Get some example questions
    questions_df = pd.read_sql_query("SELECT DISTINCT question_text FROM questions LIMIT 5", conn)
    
    if questions_df.empty:
        print("❌ No questions found in database")
        conn.close()
        return
    
    # Test each question with a sample response
    for _, row in questions_df.iterrows():
        question = row['question_text']
        print(f"\n📝 Question: {question}")
        
        while True:
            response = input("Enter a test response (or 'q' to select another question): ")
            if response.lower() == 'q':
                break
                
            try:
                sentiment = predict_sentiment(model_path, question, response)
                if sentiment is not None:
                    print(f"🔍 Predicted sentiment: {sentiment:.1f}/100")
                    
                    # Display sentiment category
                    if sentiment <= 20:
                        print("📊 Category: Very negative/concerning")
                    elif sentiment <= 40:
                        print("📊 Category: Moderately negative")
                    elif sentiment <= 60:
                        print("📊 Category: Neutral")
                    elif sentiment <= 80:
                        print("📊 Category: Moderately positive")
                    else:
                        print("📊 Category: Very positive/encouraging")
                else:
                    print("❌ Failed to parse sentiment prediction")
            except Exception as e:
                print(f"❌ Error during prediction: {e}")
    
    conn.close()

def display_menu():
    print("\n=== Sentiment Analysis Model Training ===")
    print("1. Train New Sentiment Model")
    print("2. Evaluate Existing Model")
    print("3. Test Model with Examples")
    print("4. List Available Models")
    print("5. Batch Predict Sentiment Scores")
    print("0. Exit")
    return input("\nSelect an option: ")

def main():
    while True:
        choice = display_menu()

        match choice:
            case '1':
                try:
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
                except Exception as e:
                    print(f"❌ Training failed: {e}")
                    
            case '2':
                models = list_available_models()
                if not models:
                    print("❌ No trained models available")
                    continue
                    
                print("\nAvailable models:")
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model}")
                    
                model_num = input("\nSelect model to evaluate (number): ")
                if not model_num.isdigit() or not (0 < int(model_num) <= len(models)):
                    print("❌ Invalid model number")
                    continue
                    
                selected_model = models[int(model_num)-1]
                model_path = os.path.join(MODELS_DIR, selected_model)
                
                # Check if model files exist
                if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
                    print(f"❌ Model files not found or incomplete: {model_path}")
                    continue
                
                # Call evaluate function
                evaluate_model(model_path)
                
            case '3':
                test_model_with_examples()
                
            case '4':
                models = list_available_models()
                if models:
                    print("\n📋 Available models:")
                    for i, model in enumerate(models, 1):
                        print(f"{i}. {model}")
                else:
                    print("\n❓ No trained models found")
                    
            case '5':
                models = list_available_models()
                if not models:
                    print("❌ No trained models available")
                    continue
                    
                print("\nAvailable models:")
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model}")
                    
                model_num = input("\nSelect model for batch prediction (number): ")
                if not model_num.isdigit() or not (0 < int(model_num) <= len(models)):
                    print("❌ Invalid model number")
                    continue
                    
                selected_model = models[int(model_num)-1]
                model_path = os.path.join(MODELS_DIR, selected_model)
                
                # Batch predict sentiment for all unlabeled responses
                batch_predict_sentiment(model_path)
                
            case '0':
                print("Exiting...")
                sys.exit()
                
            case _:
                print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()