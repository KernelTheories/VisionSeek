from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

print("Model Loaded")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Processor Loaded")

url = "https://images.unsplash.com/photo-1517841905240-472988babdf9"

img = Image.open(requests.get(url=url, stream=True).raw)

print("Image Downloaded")

texts = ['cat', 'dog', 'human']

input = processor(images=img, text=texts, return_tensors = "pt", padding = True)

with torch.inference_mode():
    output = model(**input)
logits_per_img = output.logits_per_image
probs = logits_per_img.softmax(dim = 1)
print()

for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item():.4f}")