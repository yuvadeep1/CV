import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
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

            # Track ID
            track_id = int(box.id[0])

            # Confidence
            confidence = float(box.conf[0])

            # Bounding Box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            print(
                f"ID={track_id} | "
                f"CONF={confidence:.2f} | "
                f"BOX=({x1},{y1},{x2},{y2})"
            )

            # Crop Person
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:

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
