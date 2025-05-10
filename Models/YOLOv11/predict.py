from ultralytics import YOLO
model = YOLO("yolov11_custom.pt")
model.predict(source = "10.Jpeg", show = True, save = True)