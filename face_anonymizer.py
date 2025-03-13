import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE_LANDMARKS = [33, 133, 160, 158, 153, 144, 362, 385, 387, 263, 373, 380]
LEFT_EYE_LANDMARKS = [263, 362, 385, 387, 373, 380, 33, 133, 160, 158, 153, 144]

class FaceAnonymizer:
    def __init__(self, max_faces=10):
        self.cap = cv2.VideoCapture(0)
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.last_face_time = None
        self.mask_visible = False
        self.mask_duration = 3  # Segundos sin detectar rostro antes de quitar la máscara

    def get_frame(self, apply_censorship=True):
        """Captura un frame de la cámara y aplica la censura si es necesario."""
        success, image = self.cap.read()
        if not success:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape  

        if results.multi_face_landmarks and apply_censorship:
            self.last_face_time = time.time()
            self.mask_visible = True  

            for face_landmarks in results.multi_face_landmarks:
                face_points = np.array([
                    (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                    for i in range(468)
                ])
                
                mask = np.zeros((h, w), dtype=np.uint8)
                convex_hull = cv2.convexHull(face_points)
                cv2.fillConvexPoly(mask, convex_hull, 255)

                blurred_image = cv2.GaussianBlur(image, (35, 35), 30)
                image = np.where(mask[:, :, np.newaxis] == 255, blurred_image, image)

                # Censura de ojos
                x1, y1, x2, y2 = self.get_eye_bbox(face_landmarks, RIGHT_EYE_LANDMARKS, w, h)
                x3, y3, x4, y4 = self.get_eye_bbox(face_landmarks, LEFT_EYE_LANDMARKS, w, h)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)  
                cv2.rectangle(image, (x3, y3), (x4, y4), (0, 0, 0), -1)  

        elif not results.multi_face_landmarks and apply_censorship:
            if self.last_face_time is not None and time.time() - self.last_face_time > self.mask_duration:
                self.mask_visible = False

            if self.mask_visible:
                blurred_image = cv2.GaussianBlur(image, (35, 35), 30)
                image = blurred_image  

        return image

    def get_eye_bbox(self, face_landmarks, eye_landmarks, w, h, scale=4.0):
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

    def release(self):
        """Libera la cámara y los recursos de MediaPipe."""
        self.cap.release()
        self.face_mesh.close()
