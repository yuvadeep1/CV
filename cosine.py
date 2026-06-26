from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
from torch.nn.functional import cosine_similarity

# Load DINOv2
processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

# ---------- IMAGE 1 ----------
image1 = Image.open("resources/rc1.png")

inputs1 = processor(
    images=image1,
    return_tensors="pt"
)

with torch.no_grad():
    outputs1 = model(**inputs1)

embedding1 = outputs1.last_hidden_state[:, 0]

# ---------- IMAGE 2 ----------
image2 = Image.open("resources/rc1.png")

inputs2 = processor(
    images=image2,
    return_tensors="pt"
)

with torch.no_grad():
    outputs2 = model(**inputs2)

embedding2 = outputs2.last_hidden_state[:, 0]

# ---------- COSINE SIMILARITY ----------
similarity = cosine_similarity(
    embedding1,
    embedding2
)

print("Similarity =", similarity.item())
