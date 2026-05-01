import json
import os
import time
import requests
from datetime import datetime
from tqdm import tqdm

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
DATA_FILE = "halueval_test.jsonl"

RAW_ARCHIVE_FILE = "halueval_raw_archive.jsonl"
PROCESSED_RESULTS_FILE = "halueval_results.jsonl"

MODEL_NAME = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"

MAX_RETRIES = 3
RETRY_DELAY = 2

# CPU SAFE SETTINGS
NUM_SAMPLES = 3  # reduce if system is slow

# -------------------------------------------------------
# RESUME SUPPORT
# -------------------------------------------------------
def get_processed_count():
    if not os.path.exists(PROCESSED_RESULTS_FILE):
        return 0
    with open(PROCESSED_RESULTS_FILE, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# -------------------------------------------------------
# OLLAMA QUERY
# -------------------------------------------------------
def query_model(prompt, temperature=0.3, max_tokens=300):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "stream": False
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            return data["message"]["content"], data

        except Exception as e:
            print(f"[Retry {attempt+1}] Error: {e}")
            time.sleep(RETRY_DELAY)

    return "ERROR", {"error": "Failed after retries"}

# -------------------------------------------------------
# MULTI-SAMPLE GENERATION (CPU SAFE - SEQUENTIAL)
# -------------------------------------------------------
def generate_samples(prompt):
    texts = []
    raws = []

    for _ in range(NUM_SAMPLES):
        text, raw = query_model(prompt, temperature=0.7)
        texts.append(text)
        raws.append(raw)

        time.sleep(0.3)  # prevents CPU overload

    return texts, raws

# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------
dataset = []

with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

print(f"Loaded {len(dataset)} records")

processed_count = get_processed_count()
print(f"Resuming from: {processed_count}")

# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
with open(RAW_ARCHIVE_FILE, "a", encoding="utf-8") as archive, \
     open(PROCESSED_RESULTS_FILE, "a", encoding="utf-8") as processed:

    for idx in tqdm(range(processed_count, len(dataset)), desc="Processing"):

        entry = dataset[idx]

        passage = entry["passage"]
        question = entry["question"]
        reference_answer = entry["answer"]

        prompt = f"""
Answer the question ONLY using the information in the passage.

If the answer is not present in the passage, say "Not in passage".

Passage:
{passage}

Question:
{question}

Answer:
"""

        start_time = time.time()

        # -------------------------------------------------------
        # GENERATE RESPONSES
        # -------------------------------------------------------
        texts, raws = generate_samples(prompt)

        # -------------------------------------------------------
        # ARCHIVE RAW RESPONSES
        # -------------------------------------------------------
        archive.write(json.dumps({
            "question": question,
            "passage": passage,
            "reference_answer": reference_answer,
            "raw_responses": raws,
            "timestamp": datetime.now().isoformat()
        }) + "\n")

        # -------------------------------------------------------
        # SAVE PROCESSED OUTPUT
        # -------------------------------------------------------
        processed.write(json.dumps({
            "question": question,
            "passage": passage,
            "reference_answer": reference_answer,
            "generated_answers": texts
        }) + "\n")

        # flush periodically (important for crash recovery)
        if idx % 5 == 0:
            archive.flush()
            processed.flush()

        elapsed = time.time() - start_time
        tqdm.write(f"Processed {idx+1} in {elapsed:.2f}s")

# -------------------------------------------------------
# DONE
# -------------------------------------------------------
print("\nPipeline completed successfully.")