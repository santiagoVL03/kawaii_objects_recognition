from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    print(model.model)  # model.model es un módulo de PyTorch (nn.Module)

if __name__ == "__main__":
    main()