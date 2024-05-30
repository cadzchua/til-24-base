import requests
from typing import List
from transformers import CLIPProcessor, CLIPModel
import torch
import io
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class VLMManager:
    def __init__(self):
        # define processor and model
        # self.model_id = "/home/jupyter/CLIP-large-finetuned"
        # self.model_id = "openai/clip-vit-large-patch14"
        self.model_id = "/home/jupyter/clip-finetune"
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)
        # move model to device if possible
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model
        text = "a " + caption.lower()
        image = self.process_image(image)
        result = self.run_model(image, text, 0.80)
        return result

    def process_image(self, image_data: bytes) -> Image:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image

    def run_model(self, image: Image, text: str, threshold: float):
        # transform the image into tensor
        transform = transforms.ToTensor()

        img = transform(image)

        # add extra dimension for later calculations
        patches = img.data.unfold(0,3,3)

        # break the image into patches (in height dimension)
        patch = 20

        patches = patches.unfold(1, patch, patch)

        # break the image into patches (in width dimension)
        patches = patches.unfold(2, patch, patch)
        
        window = 2
        stride = 1

        scores = torch.zeros(patches.shape[1], patches.shape[2])
        runs = torch.ones(patches.shape[1], patches.shape[2])

        for Y in range(0, patches.shape[1]-window+1, stride):
            for X in range(0, patches.shape[2]-window+1, stride):
                big_patch = torch.zeros(patch*window, patch*window, 3)
                patch_batch = patches[0, Y:Y+window, X:X+window]
                for y in range(window):
                    for x in range(window):
                        big_patch[
                            y*patch:(y+1)*patch, x*patch:(x+1)*patch, :
                        ] = patch_batch[y, x].permute(1, 2, 0)
                # we preprocess the image and class label with the CLIP processor
                inputs = self.processor(
                    images=big_patch,  # big patch image sent to CLIP
                    return_tensors="pt",  # tell CLIP to return pytorch tensor
                    text=text,  # class label sent to CLIP
                    padding=True,
                    do_rescale=False
                ).to(self.device) # move to device if possible

                # calculate and retrieve similarity score
                score = self.model(**inputs).logits_per_image.item()
                # sum up similarity scores from current and previous big patches
                # that were calculated for patches within the current window
                scores[Y:Y+window, X:X+window] += score
                # calculate the number of runs on each patch within the current window
                runs[Y:Y+window, X:X+window] += 1

        # average score for each patch
        scores /= runs

        # transform the patches tensor 
        adj_patches = patches.squeeze(0).permute(3, 4, 2, 0, 1)
        # normalize scores
        scores = (
            scores - scores.min()) / (scores.max() - scores.min()
        )
        # multiply patches by scores
        adj_patches = adj_patches * scores
        # rotate patches to visualize
        adj_patches = adj_patches.permute(3, 4, 2, 0, 1)

        Y = adj_patches.shape[0]
        X = adj_patches.shape[1]

        # clip the scores' interval edges
        for _ in range(1):
            scores = np.clip(scores-scores.mean(), 0, np.inf)

        # normalize scores
        scores = (
            scores - scores.min()) / (scores.max() - scores.min()
        )
        
        # transform the patches tensor 
        adj_patches = patches.squeeze(0).permute(3, 4, 2, 0, 1)
        
        # multiply patches by scores
        adj_patches = adj_patches * scores

        # rotate patches to visualize
        adj_patches = adj_patches.permute(3, 4, 2, 0, 1)
        
        Y = adj_patches.shape[0]
        X = adj_patches.shape[1]
        
        # scores higher than threshold
        detection = scores > threshold

        # non-zero positions
        np.nonzero(detection)

        y_min, y_max = (
            np.nonzero(detection)[:,0].min().item(),
            np.nonzero(detection)[:,0].max().item()+1
        )

        x_min, x_max = (
            np.nonzero(detection)[:,1].min().item(),
            np.nonzero(detection)[:,1].max().item()+1
        )

        y_min *= patch
        y_max *= patch
        x_min *= patch
        x_max *= patch

        height = y_max - y_min
        width = x_max - x_min

        return [x_min, y_min, width, height]