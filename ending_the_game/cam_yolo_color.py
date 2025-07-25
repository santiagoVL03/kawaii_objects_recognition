import cv2
from ultralytics import YOLO
from color_utils import get_dominant_color_name  # importar la función

# Cargar modelos
model_person = YOLO('pesos/best.pt')      # Ajusta según tus rutas reales
model_fashion = YOLO('pesos/best_fashion.pt')

# Abrir webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame.")
        break

    output_frame = frame.copy()

    # Paso 1: detectar personas
    results_person = model_person.predict(source=frame, conf=0.3, verbose=False)

    for r in results_person:
        for box in r.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            person_crop = frame[y1:y2, x1:x2]

            # Paso 2: detectar prendas dentro del recorte
            results_fashion = model_fashion.predict(source=person_crop, conf=0.25, verbose=False)

            for rf in results_fashion:
                for fbox, cls in zip(rf.boxes.xyxy.cpu().numpy().astype(int), rf.boxes.cls.cpu().numpy().astype(int)):
                    fx1, fy1, fx2, fy2 = fbox
                    label = rf.names[cls]

                    # Recorte de la prenda
                    prenda_crop = person_crop[fy1:fy2, fx1:fx2]
                    color_name = get_dominant_color_name(prenda_crop)

                    # Dibujar en la imagen
                    cv2.rectangle(output_frame, (x1 + fx1, y1 + fy1), (x1 + fx2, y1 + fy2), (255, 0, 255), 2)
                    cv2.putText(output_frame, f"{label} ({color_name})", (x1 + fx1, y1 + fy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Mostrar resultado
    cv2.imshow("YOLOv8 - Personas y prendas", output_frame)

    # Salir con ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
