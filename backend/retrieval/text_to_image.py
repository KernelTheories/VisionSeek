import faiss
import torch
import open_clip
import numpy as np

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

INDEX_PATH = "datasets/faiss_index.bin"

PATHS_PATH = "datasets/embeddings/image_paths.npy"

index = faiss.read_index(str(INDEX_PATH))

image_paths = np.load(PATHS_PATH)

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='laion2b_s34b_b79k'
)

tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

query = "green tee"
text = tokenizer([query])

with torch.inference_mode():

    text_features = model.encode_text(text)

    text_features /= text_features.norm(
        dim=-1,
        keepdim=True
    )

query_embedding = text_features.numpy()

TOP_K = 5

distances, indices = index.search(
    query_embedding,
    TOP_K
)

print("\nTop Matches:\n")

fig, axes = plt.subplots(1, TOP_K, figsize=(15, 5))

for i, idx in enumerate(indices[0]):

    img_path = image_paths[idx]

    print(f"{i+1}: {img_path}")

    image = Image.open(img_path)

    axes[i].imshow(image)
    axes[i].axis("off")

plt.suptitle(query)

plt.show()