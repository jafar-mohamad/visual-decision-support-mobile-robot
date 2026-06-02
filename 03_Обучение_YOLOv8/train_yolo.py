from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
 data="dataset.yaml",
 imgsz=640,
 epochs=50,
 batch=8,
 device="cpu",
 patience=100,
 name="road_yolov8n_4cls_clean_v2"
)