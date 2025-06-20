from ultralytics import YOLO
import time

# Load model
model = YOLO("yolov11_custom.pt")

# Start timer
start = time.time()

# Run prediction
results = model.predict(source="2.PNG", show=True, save=True)

# End timer
end = time.time()

# Print inference duration
print(f"Inference time: {end - start:.4f} seconds")
