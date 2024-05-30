import inflect
import re
import torch
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration
from io import BytesIO

class ASRManager:
    def __init__(self):
        # initialize the model here
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small.en")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small.en", language="en", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained("cadzchua/whisper-small-en-7up") # -v2-with-audiomentations2")
        self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = inflect.engine()

    def transcribe(self, audio_bytes: bytes) -> str:
        # Load and preprocess audio
        audio_input, sampling_rate = self.load_audio(audio_bytes)

        # Ensure the audio is at the expected sample rate
        if sampling_rate != self.processor.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(sampling_rate, self.processor.feature_extractor.sampling_rate)
            audio_input = resampler(torch.tensor(audio_input))

        # Process the audio input
        inputs = self.processor(audio_input, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt")
        inputs = {key: value.to('cuda' if torch.cuda.is_available() else 'cpu') for key, value in inputs.items()}
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(inputs["input_features"])

        # Decode the transcription
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        normalized_text = self.normalize_text(transcription)

        return normalized_text
    
    def load_audio(self, audio_path):
        audio_file = BytesIO(audio_path)
        speech_array, sampling_rate = torchaudio.load(audio_file)
        return speech_array.squeeze().numpy(), sampling_rate

    def normalize_text(self, text):
        
        # Replace numbers with words
        def replace_with_words(match):
            number = match.group(0)
            return ' '.join(self.p.number_to_words(digit) for digit in number)

        # Use regex to find numbers and replace them with words
        text = re.sub(r'\b\d+\b', replace_with_words, text)
        return text