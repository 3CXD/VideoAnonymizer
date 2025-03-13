import sys
import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QCheckBox
from face_anonymizer import FaceAnonymizer

class FaceAnonymizerApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Anonymizer")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.checkbox = QCheckBox("Censurar rostros", self)
        self.checkbox.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.checkbox)
        self.setLayout(layout)

        self.face_anonymizer = FaceAnonymizer()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        """Captura el frame procesado y lo muestra en la interfaz."""
        frame = self.face_anonymizer.get_frame(apply_censorship=self.checkbox.isChecked())
        if frame is not None:
            qt_image = self.convert_cv_qt(frame)
            self.image_label.setPixmap(qt_image)

    def convert_cv_qt(self, cv_img):
        """Convierte una imagen de OpenCV a QPixmap para PyQt6."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    def closeEvent(self, event):
        """Libera recursos al cerrar la ventana."""
        self.face_anonymizer.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceAnonymizerApp()
    window.show()
    sys.exit(app.exec())
