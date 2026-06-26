import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)  # webcam

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    if len(results) > 0:

        boxes = results[0].boxes

        for box in boxes:

            # Class
            cls = int(box.cls[0])

            # Confidence
            conf = float(box.conf[0])

            # Bounding Box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Track ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            print("-------------------")
            print("Class:", cls)
            print("Confidence:", conf)
            print("Track ID:", track_id)
            print(
                "Box:",
                x1,
                y1,
                x2,
                y2
            )

    annotated = results[0].plot()

    cv2.imshow(
        "Tracking",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
