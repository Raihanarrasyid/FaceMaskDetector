from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtGui import QFont
import subprocess
import detect_mask_image

def main():
    app = QApplication([])
    window = QMainWindow()
    window.setWindowTitle('Face Mask Detection App')
    window.setGeometry(100, 100, 800, 600)

    def run_detection_on_image():
        detect_mask_image.detect_mask()

    def run_detection_on_video():
        subprocess.run(['python3', 'detect_mask_video.py'])

    def train_detector():
        subprocess.run(['python3', 'train_mask_detector.py', '--dataset', 'dataset'])

    button1 = QPushButton('Deteksi Masker pada Gambar', window)
    button1.setGeometry(100, 100, 600, 100)
    button1.setFont(QFont('Arial', 24))
    button1.clicked.connect(run_detection_on_image)


    button2 = QPushButton('Deteksi Masker pada Video', window)
    button2.setGeometry(100, 250, 600, 100)
    button2.setFont(QFont('Arial', 24))
    button2.clicked.connect(run_detection_on_video)

    button3 = QPushButton('Latih Detektor Masker', window)
    button3.setGeometry(100, 400, 600, 100)
    button3.setFont(QFont('Arial', 24))
    button3.clicked.connect(train_detector)


    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
