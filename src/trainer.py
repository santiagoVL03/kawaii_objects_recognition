# train_yolo.py

from ultralytics import YOLO

def main():
    # Cargar modelo base preentrenado
    model = YOLO("yolo11n.pt")  # cambia si usas yolov8n.pt o cualquier otro

    # Entrenamiento
    model.train(
        data="clother.yaml",
        epochs=10,
        imgsz=640,
        batch=8,
        name="clothes_recognition_own_dataset",
    )

if __name__ == "__main__":
    main()
