from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

image = Image.open("resources/person.jpg")
print(image.size)

inputs = processor(
    images=image,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

embedding = outputs.last_hidden_state[:, 0]

print(embedding[0][:10])
