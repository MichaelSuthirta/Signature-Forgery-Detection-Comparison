from ultralytics import YOLO

def main():
    model = YOLO("yolov11_custom.pt")
    metrics = model.val(data="dataset_custom.yaml")
    print(metrics)

if __name__ == '__main__':
    main()
