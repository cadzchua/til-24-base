from typing import List
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Owlv2ForObjectDetection
import io
import numpy as np
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

class VLMManager:
    def __init__(self):
        # initialize the model here
        self.model_id = "google/owlv2-base-patch16"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_id)
        self.model.to(self.device)

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model
        text = [["a photo of a " + caption.lower()]]
        image = self.process_image(image)
        result = self.run_model(image, text)
        bbox = self.convert_bbox_format(result)
        return bbox

    def process_image(self, image_data: bytes) -> Image:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    
    def get_preprocessed_image(self, pixel_values):
        pixel_values = pixel_values.squeeze().cpu().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image

    def run_model(self, image: Image, text: str):
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        unnormalized_image = self.get_preprocessed_image(inputs.pixel_values)
        
        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=0.2, target_sizes=target_sizes
        )
        
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = text[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        
        # Find the index of the highest score
        max_score_index = scores.argmax().item()

        # Retrieve the box, score, and label with the highest score
        best_box = boxes[max_score_index]
        best_score = scores[max_score_index]
        best_label = labels[max_score_index]

        # Convert the box to a list of rounded values
        best_box = [round(coord, 2) for coord in best_box.tolist()]
        
        return best_box
    
    def convert_bbox_format(self, bbox: List[float]) -> List[int]:
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        return [int(xmin), int(ymin), int(width), int(height)]