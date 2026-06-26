from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import faiss
import numpy as np

# -------------------------------
# Load DINOv2
# -------------------------------

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

# -------------------------------
# Function to get embedding
# -------------------------------


def get_embedding(image_path):

    image = Image.open(image_path)

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0]

    embedding = embedding.cpu().numpy().astype(
        "float32"
    )

    # IMPORTANT
    faiss.normalize_L2(
        embedding
    )

    return embedding


# -------------------------------
# Create gallery embeddings
# -------------------------------

emb1 = get_embedding(
    "resources/person1.jpg"
)

emb2 = get_embedding(
    "resources/person2.jpg"
)

# -------------------------------
# Create Faiss Index
# -------------------------------

dimension = 768

index = faiss.IndexFlatIP(
    dimension
)

index.add(emb1)
index.add(emb2)

# -------------------------------
# Query Image
# -------------------------------

query = get_embedding(
    "resources/person1.jpg"
)

# -------------------------------
# Search
# -------------------------------

distances, indices = index.search(
    query,
    k=2
)

print(
    "Similarities:"
)

print(
    distances
)

print(
    "Indices:"
)

print(
    indices
)
