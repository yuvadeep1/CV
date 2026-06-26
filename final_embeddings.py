import cv2
from ultralytics import YOLO

from transformers import (
    AutoImageProcessor,
    AutoModel
)

from PIL import Image

import torch
import numpy as np

# -------------------------
# YOLO
# -------------------------

yolo_model = YOLO(
    "yolov8n.pt"
)

# -------------------------
# DINOv2
# -------------------------

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

dino_model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

# -------------------------
# Video
# -------------------------

cap = cv2.VideoCapture(
    "resources/frnds.mp4"
)

while True:

    success, frame = cap.read()

    if not success:
        break

    results = yolo_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    annotated_frame = results[0].plot()

    boxes = results[0].boxes

    if boxes.id is not None:

        for box in boxes:
            cls = box.cls[0]

            if cls != 0:
                continue
            # -----------------
            # Track ID
            # -----------------

            track_id = int(
                box.id[0]
            )

            # -----------------
            # Bounding Box
            # -----------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # -----------------
            # Crop Person
            # -----------------

            crop = frame[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            # -----------------
            # OpenCV -> PIL
            # -----------------

            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )

            pil_image = Image.fromarray(
                crop_rgb
            )

            # -----------------
            # DINOv2
            # -----------------

            inputs = processor(
                images=pil_image,
                return_tensors="pt"
            )

            with torch.no_grad():

                outputs = dino_model(
                    **inputs
                )

            embedding = (
                outputs
                .last_hidden_state[:, 0]
            )

            print(
                f"ID={track_id}"
            )

            print(
                embedding[0][:5]
            )

            # -----------------
            # Show Crop
            # -----------------

            cv2.imshow(
                f"Person_{track_id}",
                crop
            )

    cv2.imshow(
        "Tracking",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
