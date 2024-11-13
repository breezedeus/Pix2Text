# coding: utf-8
from doclayout_yolo import YOLOv10

# Load the pre-trained model
model = YOLOv10("doclayout_yolo_docstructbench_imgsz1024.pt")
img = 'docs/examples/page-authors-1.png'
# img = 'docs/examples/page.png'
# Perform prediction
det_res = model.predict(
    img,   # Image to predict
    imgsz=1024,        # Prediction image size
    conf=0.2,          # Confidence threshold
    device="mps"    # Device to use (e.g., 'cuda:0' or 'cpu')
)[0]
print(det_res.boxes)
breakpoint()