import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


RIGHT_EYE_LANDMARKS = [33, 133, 160, 158, 153, 144, 362, 385, 387, 263, 373, 380]
LEFT_EYE_LANDMARKS = [263, 362, 385, 387, 373, 380, 33, 133, 160, 158, 153, 144]


cap = cv2.VideoCapture(0)

last_face_time = None
mask_visible = False
mask_duration = 3  

with mp_face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Comprobar si hay caras detectadas
        if results.multi_face_landmarks:
            last_face_time = time.time()
            mask_visible = True 

            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape  

                def get_eye_bbox(eye_landmarks, scale=4.0):
                    """Obtiene el rectángulo delimitador de un ojo con un factor de escala."""
                    x_min = min([face_landmarks.landmark[i].x for i in eye_landmarks]) * w
                    x_max = max([face_landmarks.landmark[i].x for i in eye_landmarks]) * w
                    y_min = min([face_landmarks.landmark[i].y for i in eye_landmarks]) * h
                    y_max = max([face_landmarks.landmark[i].y for i in eye_landmarks]) * h

                    
                    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                    width, height = (x_max - x_min) * 2.0, (y_max - y_min) * scale

                    
                    x_min, x_max = int(cx - width / 2), int(cx + width / 2)
                    y_min, y_max = int(cy - height / 2), int(cy + height / 2)

                    return x_min, y_min, x_max, y_max

                
                face_points = np.array([
                    (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                    for i in range(468)  
                ])
                
               
                mask = np.zeros((h, w), dtype=np.uint8)
                convex_hull = cv2.convexHull(face_points)
                cv2.fillConvexPoly(mask, convex_hull, 255)

               
                blurred_image = cv2.GaussianBlur(image, (35, 35), 30)

                #Aqui se dibuja el desenfoque
                image = np.where(mask[:, :, np.newaxis] == 255, blurred_image, image)

                x1, y1, x2, y2 = get_eye_bbox(RIGHT_EYE_LANDMARKS)
                x3, y3, x4, y4 = get_eye_bbox(LEFT_EYE_LANDMARKS)

                # Aqui se dibuja el rectangulo
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)  
                cv2.rectangle(image, (x3, y3), (x4, y4), (0, 0, 0), -1)  

        else:
            
            if last_face_time is not None and time.time() - last_face_time > mask_duration:
                
                mask_visible = False

            
            if mask_visible:
                
                mask = np.zeros((h, w), dtype=np.uint8)
                convex_hull = cv2.convexHull(face_points)
                cv2.fillConvexPoly(mask, convex_hull, 255)

                blurred_image = cv2.GaussianBlur(image, (35, 35), 30)
                image = np.where(mask[:, :, np.newaxis] == 255, blurred_image, image)

                x1, y1, x2, y2 = get_eye_bbox(RIGHT_EYE_LANDMARKS)
                x3, y3, x4, y4 = get_eye_bbox(LEFT_EYE_LANDMARKS)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
                cv2.rectangle(image, (x3, y3), (x4, y4), (0, 0, 0), -1)

       
        cv2.imshow('Anonymizer', cv2.flip(image, 1))
        
        # Salir si se presiona 'ESC'
        if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty('Anonymizer', cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
