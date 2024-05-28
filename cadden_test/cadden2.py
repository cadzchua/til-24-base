import jsonlines
import torchaudio
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments, AutoModelForSequenceClassification, EarlyStoppingCallback, TrainerCallback
from pathlib import Path
import torch, random
import librosa, os
import IPython.display as ipd
from dotenv import load_dotenv
import evaluate
from torch_audiomentations import Compose, Gain, PitchShift, AddColoredNoise, PolarityInversion, PeakNormalization, HighPassFilter, LowPassFilter, Shift
from spellchecker import SpellChecker
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_metric
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'facebook/wav2vec2-base-960h'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name, pad_token_id=processor.tokenizer.pad_token_id).to(device)

load_dotenv()
torch.cuda.empty_cache()
TEAM_NAME = os.getenv("TEAM_NAME", "7up")
TEAM_TRACK = os.getenv("TEAM_TRACK", "advanced")


input_dir = Path(f"/home/jupyter/{TEAM_TRACK}")
# input_dir = Path(f"../../data/{TEAM_TRACK}/train")
results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
# results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
instances = []

data = {'key': [], 'audio': [], 'transcript': []}
with jsonlines.open(input_dir / "asr.jsonl") as reader:
    for obj in reader:
        if len(data['key']) < 100:
            for key, value in obj.items():
                data[key].append(value)

# Convert to a Hugging Face dataset
dataset = Dataset.from_dict(data)

# Shuffle the dataset
dataset = dataset.shuffle(seed=42)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, train_size + val_size))
test_dataset = dataset.select(range(train_size + val_size, train_size + val_size + test_size))

def preprocess_data(examples):
    input_values = []
    attention_masks = []
    labels = []

    for audio_path, transcript in zip(examples['audio'], examples['transcript']):
        speech_array, sampling_rate = torchaudio.load(input_dir / "audio" / audio_path)
        processed = processor(speech_array.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt", padding=True)

        # Process labels with the same processor settings
        with processor.as_target_processor():
            label = processor(transcript, return_tensors="pt", padding=True)

        input_values.append(processed.input_values.squeeze(0))
        # Create attention masks based on the input values
        attention_mask = torch.ones_like(processed.input_values)
        attention_mask[processed.input_values == processor.tokenizer.pad_token_id] = 0  # Set padding tokens to 0
        attention_masks.append(attention_mask.squeeze(0))

        # Ensure labels are padded to the same length as inputs if needed
        padded_label = torch.full(processed.input_values.shape[1:], -100, dtype=torch.long)
        actual_length = label.input_ids.shape[1]
        padded_label[:actual_length] = label.input_ids.squeeze(0)
        labels.append(padded_label)

    # Concatenate all batches
    examples['input_values'] = torch.stack(input_values)
    examples['attention_mask'] = torch.stack(attention_masks)
    examples['labels'] = torch.stack(labels)

    return examples

train_dataset = train_dataset.map(preprocess_data, batched=True, batch_size=16, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess_data, batched=True, batch_size=16, remove_columns=val_dataset.column_names)
test_dataset = test_dataset.map(preprocess_data, batched=True, batch_size=16, remove_columns=test_dataset.column_names)

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
      
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        
# data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

def compute_metrics(pred):

    pred_logits = pred.predictions
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    if type(processor).__name__ == "Wav2Vec2ProcessorWithLM":
        pred_str    = processor.batch_decode(pred_logits).text
    else:   
       pred_ids = np.argmax(pred_logits, axis=-1)
       pred_str = processor.batch_decode(pred_ids)
  
    # this will always work (even if the processor has a decoder LM)
    label_str = processor.tokenizer.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

training_args = TrainingArguments(
  output_dir=f"./test-wav2vec2/Checkpoint",
  group_by_length=False,
  per_device_train_batch_size=16, 
  evaluation_strategy="epoch",
  save_strategy="epoch",
  num_train_epochs=30,
  fp16=True,#change to True on CUDA
  gradient_checkpointing=True, 
  save_steps=1000,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=100,
  save_total_limit=1,
  load_best_model_at_end = True,
  optim = "adamw_torch", 
  dataloader_pin_memory = False
)

trainer = Trainer(
    model=model,
    # data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset, 
    eval_dataset=val_dataset,
    tokenizer=processor.feature_extractor,
    callbacks = [MyCallback]    
)

trainer.train()