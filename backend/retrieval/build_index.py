import faiss
import numpy as np
from pathlib import Path

EMB_PATH = "datasets/embeddings/image_embeddings.npy"

embeddings = np.load(EMB_PATH)
print("Embeddings Loaded", embeddings.shape)


dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)
print("FAISS index size:", index.ntotal)

INDEX_PATH = "datasets/faiss_index.bin"

faiss.write_index(index, str(INDEX_PATH))

print("Index saved.")
