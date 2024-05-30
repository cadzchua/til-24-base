from typing import List
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import io

class VLMManager:
    def __init__(self):
        # initialize the model here
        self.model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self.model.to(self.device)

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model
        text = "a " + caption.lower() + "."
        image = self.process_image(image)
        result = self.run_model(image, text)
        scores_list = result[0]['scores'].tolist()
        # Get the index of the maximum score
        max_score_index = scores_list.index(max(scores_list))
        # Retrieve the box with the highest score
        box_with_highest_score = result[0]['boxes'][max_score_index].tolist()
        bbox = self.convert_bbox_format(box_with_highest_score)
        return bbox

    def process_image(self, image_data: bytes) -> Image:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image

    def run_model(self, image: Image, text: str):
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.2,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        
        return results
    
    def convert_bbox_format(self, bbox: List[float]) -> List[int]:
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        return [int(xmin), int(ymin), int(width), int(height)]
