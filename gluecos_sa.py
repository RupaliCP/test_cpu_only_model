# ============================================================
# HINDI-ENGLISH CODE-MIXED SENTIMENT ANALYSIS
# USING HINGLISH SENTIMENT ANALYSIS DATASET FROM KAGGLE
#
# Metrics:
# - Accuracy
# - Precision
# - Recall
# - F1 Score
#
# NO TRAINING REQUIRED
# ============================================================

# INSTALL REQUIRED LIBRARIES:
# pip install transformers torch scikit-learn pandas kagglehub

import torch
import numpy as np
import pandas as pd
import kagglehub
import os
import logging
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

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

# Create log filename with timestamp
log_filename = f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("HINGLISH SENTIMENT ANALYSIS - BERT MODELS EVALUATION")
logger.info("="*60)
logger.info(f"Log file: {log_filename}")
logger.info("")

# ============================================================
# 1. LOAD HINGLISH SENTIMENT ANALYSIS DATASET
# ============================================================

logger.info("Downloading Hinglish sentiment analysis dataset from Kaggle...")

# Download latest version
path = kagglehub.dataset_download("shivajeetrai/hinglish-data-for-sentiment-analysis")

logger.info(f"Path to dataset files: {path}")

# Load the dataset
# The dataset typically contains CSV files with text and sentiment labels
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

if not csv_files:
    logger.error("No CSV files found in the downloaded dataset")
    raise FileNotFoundError("No CSV files found in the downloaded dataset")

# Load the first CSV file (or you can specify the exact filename)
dataset_file = os.path.join(path, csv_files[0])
logger.info(f"Loading dataset from: {dataset_file}")

df = pd.read_csv(dataset_file)

logger.info("")
logger.info("Dataset loaded successfully!")
logger.info(f"Dataset: Hinglish Sentiment Analysis")
logger.info(f"Number of samples: {len(df)}")
logger.info(f"Dataset columns: {df.columns.tolist()}")
logger.info("")
logger.info("First few rows:")
logger.info(f"\n{df.head().to_string()}")

# Identify text and label columns
# Common column names: 'text', 'sentence', 'tweet', 'content'
# Common label names: 'label', 'sentiment', 'class'
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
    raise ValueError("Could not automatically identify text and label columns. Please check the dataset structure.")

logger.info("")
logger.info("Using columns:")
logger.info(f"  Text column: {text_column}")
logger.info(f"  Label column: {label_column}")

# Convert to list format for processing
test_data = df.to_dict('records')

logger.info("")
logger.info("Label distribution:")
logger.info(f"\n{df[label_column].value_counts().to_string()}")

# ============================================================
# 2. MODEL LIST
# ============================================================

models = {

    # --------------------------------------------------------
    # STANDARD BERT MODELS (Fine-tuned for Sentiment Analysis)
    # --------------------------------------------------------

    "BERT_Multilingual_Sentiment":
        "nlptown/bert-base-multilingual-uncased-sentiment",

    "DistilBERT_Multilingual":
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student",

    "MuRIL_Base":
        "google/muril-base-cased",

    "IndicBERT":
        "ai4bharat/indic-bert",

    # --------------------------------------------------------
    # CODE-MIXED BERT MODELS
    # --------------------------------------------------------

    "HingBERT":
        "l3cube-pune/hing-bert",

    "HingRoBERTa":
        "l3cube-pune/hing-roberta",

    "mBERT_Cased":
        "bert-base-multilingual-cased"
}

# ============================================================
# 3. FUNCTION TO CONVERT LABELS
# ============================================================

# Different models output labels differently.
# We convert everything into:
#
# 0 = Negative
# 1 = Neutral
# 2 = Positive
#
# For binary classification, we map:
# 0 = Negative
# 1/2 = Positive


def convert_prediction(predicted_label, num_labels):
    """
    Convert model predictions to standardized format
    """
    predicted_label = int(predicted_label)
    
    # For 3-class models (negative, neutral, positive)
    if num_labels >= 3:
        # Map neutral and positive to positive (1)
        if predicted_label >= 1:
            return 1
        else:
            return 0
    
    # For 2-class models
    elif num_labels == 2:
        return predicted_label
    
    # For 5-star rating models
    elif num_labels == 5:
        # 1-2 stars = negative (0)
        # 3-5 stars = positive (1)
        if predicted_label >= 2:
            return 1
        else:
            return 0
    
    return predicted_label


# ============================================================
# 4. EVALUATE EACH MODEL
# ============================================================

results = []

for model_name, model_path in models.items():

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"RUNNING MODEL: {model_name}")
    logger.info("=" * 60)

    try:

        # ----------------------------------------------------
        # LOAD TOKENIZER
        # ----------------------------------------------------

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        model.eval()

        # ----------------------------------------------------
        # STORE PREDICTIONS
        # ----------------------------------------------------

        true_labels = []
        predicted_labels = []

        # ----------------------------------------------------
        # LOOP THROUGH TEST DATA
        # ----------------------------------------------------

        for sample in test_data:

            # Get text from the identified text column
            text = sample.get(text_column, "")
            
            # Skip empty texts
            if not text or pd.isna(text):
                continue
            
            # Get true label from the identified label column
            true_label = sample.get(label_column, 0)
            
            # Convert label to integer if it's a string
            if isinstance(true_label, str):
                true_label_lower = true_label.lower()
                if 'positive' in true_label_lower or true_label_lower == '1':
                    true_label = 1
                elif 'negative' in true_label_lower or true_label_lower == '0':
                    true_label = 0
                elif 'neutral' in true_label_lower or true_label_lower == '2':
                    true_label = 2
                else:
                    try:
                        true_label = int(true_label)
                    except:
                        true_label = 0

            # ----------------------------------------------
            # TOKENIZE
            # ----------------------------------------------

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            # ----------------------------------------------
            # MODEL PREDICTION
            # ----------------------------------------------

            with torch.no_grad():

                outputs = model(**inputs)

                logits = outputs.logits

                predicted_class = torch.argmax(
                    logits,
                    dim=1
                ).item()

            # ----------------------------------------------
            # CONVERT LABELS
            # ----------------------------------------------

            num_labels = model.config.num_labels
            pred_label = convert_prediction(predicted_class, num_labels)
            
            # Ensure true_label is in binary format
            if true_label > 1:
                true_label = 1

            true_labels.append(true_label)
            predicted_labels.append(pred_label)

        # ----------------------------------------------------
        # CALCULATE METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            true_labels,
            predicted_labels
        )

        precision = precision_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division="warn"
        )

        recall = recall_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division="warn"
        )

        f1 = f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division="warn"
        )

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        logger.info(f"Accuracy : {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall   : {recall:.4f}")
        logger.info(f"F1 Score : {f1:.4f}")

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

    except Exception as e:

        logger.error("")
        logger.error("ERROR OCCURRED")
        logger.error(str(e))
        logger.exception("Full traceback:")

# ============================================================
# 5. FINAL RESULTS TABLE
# ============================================================

logger.info("")
logger.info("=" * 60)
logger.info("FINAL RESULTS")
logger.info("=" * 60)

results_df = pd.DataFrame(results)

logger.info("")
logger.info(f"\n{results_df.to_string()}")

# ============================================================
# 6. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "hinglish_sentiment_results.csv",
    index=False
)

logger.info("")
logger.info("Results saved to:")
logger.info("  - hinglish_sentiment_results.csv")
logger.info(f"  - {log_filename}")
logger.info("")
logger.info("="*60)
logger.info("EVALUATION COMPLETED SUCCESSFULLY")
logger.info("="*60)