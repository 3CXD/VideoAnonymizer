import cv2
import mediapipe as mp

# Inicializar MediaPipe FaceMesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Definir los índices de los ojos en FaceMesh
RIGHT_EYE_LANDMARKS = [33, 133, 160, 158, 153, 144, 362, 385, 387, 263, 373, 380]
LEFT_EYE_LANDMARKS = [263, 362, 385, 387, 373, 380, 33, 133, 160, 158, 153, 144]

# Iniciar la captura de video
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Convertir imagen a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # Volver a BGR para dibujar
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Obtener coordenadas de los ojos
                h, w, _ = image.shape  # Altura y ancho de la imagen

                def get_eye_bbox(eye_landmarks):
                    """Obtiene el rectángulo delimitador de un ojo."""
                    x_min = min([face_landmarks.landmark[i].x for i in eye_landmarks]) * w
                    x_max = max([face_landmarks.landmark[i].x for i in eye_landmarks]) * w
                    y_min = min([face_landmarks.landmark[i].y for i in eye_landmarks]) * h
                    y_max = max([face_landmarks.landmark[i].y for i in eye_landmarks]) * h
                    return int(x_min), int(y_min), int(x_max), int(y_max)

                # Obtener cajas delimitadoras de los ojos
                x1, y1, x2, y2 = get_eye_bbox(RIGHT_EYE_LANDMARKS)
                x3, y3, x4, y4 = get_eye_bbox(LEFT_EYE_LANDMARKS)

                # Dibujar rectángulos negros sobre los ojos
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)  # Ojo derecho
                cv2.rectangle(image, (x3, y3), (x4, y4), (0, 0, 0), -1)  # Ojo izquierdo

        # Mostrar la imagen con los ojos bloqueados
        cv2.imshow('Face Mesh with Black Eyes', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:  # Presiona ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
