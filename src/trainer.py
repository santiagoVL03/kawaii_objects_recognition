# train_yolo.py

from ultralytics import YOLO

def main():
    # Cargar modelo base preentrenado
    model = YOLO("yolov8n.pt")  # cambia si usas yolov8n.pt o cualquier otro
    model.info()  # Muestra información del modelo
    
    # Entrenamiento
    model.train(
        data="clother.yaml",
        epochs=6,
        imgsz=640,
        batch=16,
        name="clothes_recognition_own_dataset",
        patience=2,
    )

if __name__ == "__main__":
    main()
