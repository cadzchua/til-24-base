from transformers import Trainer, TrainingArguments
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from dotenv import load_dotenv
import os
from pathlib import Path
import base64
import json
from typing import Dict, List
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from transformers.pipelines.audio_utils import ffmpeg_read
import random
import jsonlines

load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME", "7up")
TEAM_TRACK = os.getenv("TEAM_TRACK", "advanced")


input_dir = Path(f"/home/jupyter/{TEAM_TRACK}")
# input_dir = Path(f"../../data/{TEAM_TRACK}/train")
results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
# results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
instances = []
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

sampling_rate = processor.feature_extractor.sampling_rate
data = {'key': [], 'audio': [], 'transcript': []}
with jsonlines.open(input_dir / "asr.jsonl") as reader:
    for obj in reader:
        if len(data['key']) < 10:  # Only keep the first 10 entries
            for key, value in obj.items():
                data[key].append(value)

# Convert to a Hugging Face dataset
dataset = Dataset.from_dict(data)

# Shuffle the dataset
dataset = dataset.shuffle(seed=42)
print(dataset)
# Split the dataset into training, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, train_size + val_size))
test_dataset = dataset.select(range(train_size + val_size, train_size + val_size + test_size))

# Define the sizes for train, test, and validation sets
train_size = int(0.7 * len(instances))
test_size = int(0.15 * len(instances))
val_size = len(instances) - train_size - test_size

# Split the data
train_data = instances[:train_size]
test_data = instances[train_size:train_size + test_size]
val_data = instances[train_size + test_size:]

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=1e-4,
    per_device_train_batch_size=1,  # Reduce to one for simplicity
    num_train_epochs=10,
    weight_decay=0.005,
    save_steps=500,
    eval_steps=500,
    logging_steps=10,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,  # Use the validation dataset for evaluation
    tokenizer=processor.feature_extractor
)

# Train the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate(eval_dataset=valt_data)
print(results)


# Save the trained model
trainer.save_model("./trained_whisper_model")