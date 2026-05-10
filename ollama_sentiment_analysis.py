# ============================================================
# HINGLISH SENTIMENT ANALYSIS USING OLLAMA LLM MODELS
# ============================================================

import ollama
import pandas as pd
import kagglehub
import os
import logging
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ============================================================
# SETUP LOGGING
# ============================================================

log_filename = f"ollama_sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("HINGLISH SENTIMENT ANALYSIS - OLLAMA LLM MODELS")
logger.info("="*60)
logger.info(f"Log file: {log_filename}")
logger.info("")

# ============================================================
# LOAD HINGLISH SENTIMENT DATASET
# ============================================================

logger.info("Downloading Hinglish sentiment analysis dataset from Kaggle...")

path = kagglehub.dataset_download("shivajeetrai/hinglish-data-for-sentiment-analysis")
logger.info(f"Path to dataset files: {path}")

csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

if not csv_files:
    logger.error("No CSV files found in the downloaded dataset")
    raise FileNotFoundError("No CSV files found in the downloaded dataset")

dataset_file = os.path.join(path, csv_files[0])
logger.info(f"Loading dataset from: {dataset_file}")

df = pd.read_csv(dataset_file)

logger.info("")
logger.info("Dataset loaded successfully!")
logger.info(f"Dataset: Hinglish Sentiment Analysis")
logger.info(f"Number of samples: {len(df)}")
logger.info(f"Dataset columns: {df.columns.tolist()}")

# Identify text and label columns
text_column = None
label_column = None

for col in df.columns:
    col_lower = col.lower()
    if 'text' in col_lower or 'sentence' in col_lower or 'tweet' in col_lower or 'content' in col_lower:
        text_column = col
    if 'label' in col_lower or 'sentiment' in col_lower or 'class' in col_lower:
        label_column = col

if text_column is None or label_column is None:
    logger.error(f"Available columns: {df.columns.tolist()}")
    raise ValueError("Could not automatically identify text and label columns.")

logger.info("")
logger.info("Using columns:")
logger.info(f"  Text column: {text_column}")
logger.info(f"  Label column: {label_column}")

# Limit dataset size for faster testing (optional - remove or increase for full dataset)
SAMPLE_SIZE = 100  # Test with 100 samples first
df_sample = df.head(SAMPLE_SIZE)
test_data = df_sample.to_dict('records')

logger.info(f"\nUsing {len(test_data)} samples for evaluation")
logger.info("")
logger.info("Label distribution:")
logger.info(f"\n{df_sample[label_column].value_counts().to_string()}")

# ============================================================
# OLLAMA MODELS FOR SENTIMENT ANALYSIS
# ============================================================

ollama_models = {
    # Models you already have installed
    "Qwen2.5": "qwen2.5:latest",
    "DeepSeek-R1": "deepseek-r1:7b",
    
    # Recommended lightweight models for CPU (you can install these)
    # Uncomment after installing with: ollama pull <model_name>
    # "Llama3.2-3B": "llama3.2:3b",
    # "Phi3-Mini": "phi3:mini",
    # "Gemma2-2B": "gemma2:2b",
}

# ============================================================
# SENTIMENT ANALYSIS PROMPT
# ============================================================

def create_sentiment_prompt(text):
    """Create a prompt for sentiment analysis"""
    prompt = f"""Analyze the sentiment of the following text and respond with ONLY ONE WORD: either "positive", "negative", or "neutral".

Text: {text}

Sentiment:"""
    return prompt

# ============================================================
# PARSE LLM RESPONSE
# ============================================================

def parse_sentiment(response_text):
    """Parse LLM response to extract sentiment label"""
    response_lower = response_text.lower().strip()
    
    # Extract first word if response is longer
    first_word = response_lower.split()[0] if response_lower else ""
    
    # Map to binary labels (0 = negative, 1 = positive)
    # Treat neutral as positive for binary classification
    if 'positive' in first_word or 'pos' in first_word:
        return 1
    elif 'negative' in first_word or 'neg' in first_word:
        return 0
    elif 'neutral' in first_word:
        return 1  # Map neutral to positive
    else:
        # Default to neutral/positive if unclear
        logger.warning(f"Unclear sentiment response: {response_text}")
        return 1

# ============================================================
# CONVERT DATASET LABELS
# ============================================================

def convert_true_label(label):
    """Convert dataset label to binary format"""
    if isinstance(label, str):
        label_lower = label.lower()
        if 'positive' in label_lower or label_lower == '1':
            return 1
        elif 'negative' in label_lower or label_lower == '0':
            return 0
        elif 'neutral' in label_lower or label_lower == '2':
            return 1
        else:
            try:
                return int(label)
            except:
                return 0
    else:
        # If numeric, map 0=negative, 1+=positive
        return 1 if label > 0 else 0

# ============================================================
# EVALUATE EACH OLLAMA MODEL
# ============================================================

results = []

for model_name, model_id in ollama_models.items():
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"RUNNING MODEL: {model_name} ({model_id})")
    logger.info("=" * 60)
    
    try:
        # Check if model is available
        try:
            ollama.show(model_id)
        except Exception as e:
            logger.error(f"Model {model_id} not found. Please install with: ollama pull {model_id}")
            logger.error(str(e))
            continue
        
        true_labels = []
        predicted_labels = []
        
        # Process each sample
        for idx, sample in enumerate(test_data):
            
            # Get text
            text = sample.get(text_column, "")
            if not text or pd.isna(text):
                continue
            
            # Get true label
            true_label = sample.get(label_column, 0)
            true_label = convert_true_label(true_label)
            
            # Create prompt
            prompt = create_sentiment_prompt(text)
            
            # Get prediction from Ollama
            try:
                response = ollama.chat(
                    model=model_id,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'temperature': 0.1,  # Low temperature for consistent results
                        'num_predict': 10,   # Limit response length
                    }
                )
                
                response_text = response['message']['content']
                pred_label = parse_sentiment(response_text)
                
                true_labels.append(true_label)
                predicted_labels.append(pred_label)
                
                # Log progress every 10 samples
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(test_data)} samples")
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue
        
        # Calculate metrics
        if len(true_labels) > 0:
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division="warn")
            recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division="warn")
            f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division="warn")
            
            logger.info("")
            logger.info(f"Accuracy : {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall   : {recall:.4f}")
            logger.info(f"F1 Score : {f1:.4f}")
            
            results.append({
                "Model": model_name,
                "Model_ID": model_id,
                "Samples_Processed": len(true_labels),
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            })
        else:
            logger.error("No samples were successfully processed")
    
    except Exception as e:
        logger.error("")
        logger.error("ERROR OCCURRED")
        logger.error(str(e))
        logger.exception("Full traceback:")

# ============================================================
# FINAL RESULTS
# ============================================================

logger.info("")
logger.info("=" * 60)
logger.info("FINAL RESULTS")
logger.info("=" * 60)

if results:
    results_df = pd.DataFrame(results)
    logger.info("")
    logger.info(f"\n{results_df.to_string()}")
    
    # Save results
    results_df.to_csv("ollama_sentiment_results.csv", index=False)
    
    logger.info("")
    logger.info("Results saved to:")
    logger.info("  - ollama_sentiment_results.csv")
    logger.info(f"  - {log_filename}")
else:
    logger.error("No results to display. Please check if models are installed.")
    logger.info("\nTo install models, run:")
    logger.info("  ollama pull qwen2.5:latest")
    logger.info("  ollama pull deepseek-r1:7b")
    logger.info("  ollama pull llama3.2:3b")
    logger.info("  ollama pull phi3:mini")
    logger.info("  ollama pull gemma2:2b")

logger.info("")
logger.info("="*60)
logger.info("EVALUATION COMPLETED")
logger.info("="*60)

# Made with Bob
