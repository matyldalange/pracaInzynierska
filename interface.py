import sys
import os
import cv2
from datetime import datetime
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap
from game import main
import numpy as np
from tensorflow.keras.models import load_model
from PyQt5.QtGui import QImage, QPixmap
import pandas as pd

class Aplikacja(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Aplikacja do nakładania maseczki")
        self.setGeometry(300, 300, 1500, 1000)

        self.pix2pix_model = load_model('/Users/matyldalange/PycharmProjects/pix2pix-trening,test,uczeniemetryk/final_model.h5')
        self.img_shape = (256, 256)

        self.label = QLabel("Wybierz swój obraz!", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.label_powitalny = QLabel("APLIKACJA DO NAKŁADANIA MASECZKI", self)
        self.label_powitalny.setAlignment(Qt.AlignCenter)

        self.output_label = QLabel(self)
        self.output_label.setGeometry(10, 300, 256, 256)

        self.thermal_output_label = QLabel(self)
        self.thermal_output_label.setGeometry(10, 300, 370, 300)
        self.thermal_output_label.hide()


        self.button_play_game = QPushButton("Zagraj w grę", self)
        self.button_play_game.clicked.connect(self.start_game)

        self.button_read_fun_fact = QPushButton("Ciekawostka", self)
        self.button_read_fun_fact.clicked.connect(self.show_fun_fact)

        self.button_folder = QPushButton("Wybierz zdjęcie z folderu", self)
        self.button_folder.clicked.connect(self.choose_image)

        self.button_mask = QPushButton("Nałóż maseczkę", self)
        self.button_mask.clicked.connect(self.apply_mask)

        self.button_exit = QPushButton("Wyjdź z aplikacji", self)
        self.button_exit.clicked.connect(self.close)


        layout = QVBoxLayout()
        layout.addWidget(self.label_powitalny)
        layout.addItem(QSpacerItem(50, 50, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(self.label)
        layout.addWidget(self.button_play_game)
        layout.addWidget(self.button_read_fun_fact)
        layout.addWidget(self.button_folder)
        layout.addWidget(self.button_mask)
        layout.addWidget(self.button_exit)
        self.setLayout(layout)

        self.doctor_label = QLabel(self)
        self.doctor_label.setPixmap(QPixmap("/Users/matyldalange/Desktop/doktorZmaska.png").scaled(400, 400))
        self.doctor_label.setGeometry(600, 150, 400, 400)
        self.doctor_label.show()
        self.doctor_direction = 1
        self.doctor_speed = 10


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.move_doctor)
        self.timer.start(50)

        self.fun_fact_label = QLabel(self)
        self.fun_fact_label.setPixmap(QPixmap("/Users/matyldalange/Desktop/ciekawostla.png").scaled(500, 500))
        self.fun_fact_label.setGeometry(1100, 150, 500, 500)
        self.fun_fact_label.hide()


        self.setStyleSheet("""
            QWidget {
                background-color: #E8F5E9;  
                color: #004D40;              
                font-family: 'Trebuchet MS', sans-serif;  
                font-size: 16px;             
                padding: 10px;               
            }
            QLabel {
                font-family: 'Trebuchet MS', sans-serif;
                font-size: 28px;             
                font-weight: bold;           
                color: #00796B;
                text-align: center;
            }
            QPushButton {
                background-color: #00796B;  
                color: white;               
                padding: 10px;                
                border: none;                 
                border-radius: 5px;          
                font-size: 16px; 
                font-family: 'Trebuchet MS', sans-serif;            
            }
            QPushButton:hover {
                background-color: #004D40;    
            }
            QPushButton:pressed {
                background-color: #00695C;     
            }
        """)


    def start_game(self):
        self.label.setText("Gra się uruchomi!")
        main()

    def show_fun_fact(self):
        if self.fun_fact_label.isVisible():
            self.fun_fact_label.hide()
            if hasattr(self, 'fun_fact_timer'):
                self.fun_fact_timer.stop()
        else:
            self.fun_fact_label.show()

            if not hasattr(self, 'fun_fact_timer'):
                self.fun_fact_timer = QTimer(self)
                self.fun_fact_timer.setSingleShot(True)
                self.fun_fact_timer.timeout.connect(self.fun_fact_label.hide)

            self.fun_fact_timer.start(10000)


    def check_image_type(self):
        if not hasattr(self, 'selected_image_path'):
            self.label.setText("Najpierw wybierz obraz!")
            return

        image = cv2.imread(self.selected_image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            self.label.setText("Nie udało się wczytać obrazu.")
            return

        print(image.shape)

        if len(image.shape) == 2:
            self.image_type = 'thermal'
        elif len(image.shape) == 3:
            self.image_type = 'rgb'
        else:
            self.image_type = None

    def choose_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz obraz", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
        if file_path:
            self.label.setText(f"Super, zapisaliśmy Twój obraz!")
            self.selected_image_path = file_path
            self.check_image_type()
        else:
            self.label.setText("Nie wybrano obrazu.")

    def apply_mask(self):
        if not hasattr(self, 'selected_image_path'):
            self.label.setText("Najpierw wybierz obraz!")
            return

        if not hasattr(self, 'image_type') or self.image_type is None:
            self.label.setText("Nie rozpoznano typu obrazu!")
            return

        image_path = self.selected_image_path
        output_folder = "/Users/matyldalange/Desktop/obrazy_z_maseczkami/"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            self.label.setText("Nie udało się wczytać obrazu.")
            return

        if self.image_type == 'thermal':
            self.apply_mask_thermal(image, output_folder)
        elif self.image_type == 'rgb':
            self.apply_mask_rgb()
        else:
            self.label.setText("Nieobsługiwany format obrazu.")
            return

    def apply_mask_thermal(self, image, output_folder):
        csv_path = "/Users/matyldalange/Desktop/Charlotte-ThermalFace/S1.csv"
        image_path = self.selected_image_path
        mask_front_path = '/Users/matyldalange/Desktop/MaskTheFace-master/masks/templates/surgical.png'
        mask_left_path = '/Users/matyldalange/Desktop/MaskTheFace-master/masks/templates/surgical_left.png'
        mask_right_path = '/Users/matyldalange/Desktop/MaskTheFace-master/masks/templates/surgical_right.png'

        mask_front = cv2.imread(mask_front_path, cv2.IMREAD_UNCHANGED)
        mask_left = cv2.imread(mask_left_path, cv2.IMREAD_UNCHANGED)
        mask_right = cv2.imread(mask_right_path, cv2.IMREAD_UNCHANGED)

        if any(mask is None for mask in [mask_front, mask_left, mask_right]):
            raise ValueError("Nie udało się wczytać jednej lub więcej masek. Sprawdź ścieżki.")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Nie udało się wczytać obrazu twarzy. Sprawdź ścieżkę.")

        image_id = os.path.splitext(os.path.basename(image_path))[0]
        csv_id = 'R' + image_id[1:]

        landmarks_df = pd.read_csv(csv_path, sep=",")

        landmarks = landmarks_df[landmarks_df['ID'] == csv_id]
        if landmarks.empty:
            raise ValueError(f"Nie znaleziono danych dla ID {csv_id}")

        landmarks = landmarks.iloc[0]
        nose = (int(landmarks['x27']), int(landmarks['y27']))
        chin = (int(landmarks['x8']), int(landmarks['y8']))
        left_cheek = (int(landmarks['x0']), int(landmarks['y0']))
        right_cheek = (int(landmarks['x16']), int(landmarks['y16']))

        cheek_difference = right_cheek[0] - left_cheek[0]
        if cheek_difference < -70:
            selected_mask = mask_left
            x_offset, y_offset, height_scale, width_scale = 50, 8, 1.4, 1.7
        elif cheek_difference > 70:
            selected_mask = mask_right
            x_offset, y_offset, height_scale, width_scale = -20, 50, 1.1, 1.3
        else:
            selected_mask = mask_front
            x_offset, y_offset, height_scale, width_scale = 0, 30, 1.1, 1.1

        face_width = int(np.linalg.norm(np.array(right_cheek) - np.array(left_cheek)) * width_scale)
        face_height = int(np.linalg.norm(np.array(chin) - np.array(nose)) * height_scale)
        scaled_mask = cv2.resize(selected_mask, (face_width, face_height), interpolation=cv2.INTER_AREA)
        mask_x = int(nose[0] - face_width / 2) + x_offset
        mask_y = int(nose[1] - face_height / 2) + y_offset
        alpha_mask = scaled_mask[:, :, 3] / 255.0
        for c in range(3):
            image[mask_y:mask_y + scaled_mask.shape[0], mask_x:mask_x + scaled_mask.shape[1], c] = (
                    alpha_mask * scaled_mask[:, :, c] +
                    (1 - alpha_mask) * image[mask_y:mask_y + scaled_mask.shape[0], mask_x:mask_x + scaled_mask.shape[1],
                                       c]
            )

        output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_masked.jpg"
        output_path = os.path.join(output_folder, output_filename)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(output_path, image)
        result_image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        height, width = result_image.shape
        qt_image = QImage(result_image.data, width, height, width, QImage.Format_Grayscale8)
        self.thermal_output_label.setPixmap(QPixmap.fromImage(qt_image))
        self.thermal_output_label.show()

    def apply_mask_rgb(self):
        if not hasattr(self, 'selected_image_path'):
            self.label.setText("Najpierw wybierz obraz!")
            return

        image_path = self.selected_image_path
        output_folder = "/Users/matyldalange/Desktop/obrazy_z_maseczkami/"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"maseczka_{base_name}")

        image = cv2.imread(image_path)
        if image is None:
            self.label.setText("Nie udało się wczytać obrazu.")
            return

        image_resized = cv2.resize(image, self.img_shape[:2])
        image_normalized = image_resized / 127.5 - 1.0
        input_tensor = np.expand_dims(image_normalized, axis=0)

        generated_image = self.pix2pix_model.predict(input_tensor)[0]
        generated_image = (generated_image + 1) * 127.5
        generated_image = generated_image.astype(np.uint8)

        blurred_image = cv2.GaussianBlur(generated_image, (7, 7), 0)

        cv2.imwrite(output_path, blurred_image)

        blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
        height, width, channel = blurred_image_rgb.shape
        bytes_per_line = 3 * width
        qt_image = QImage(blurred_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        self.output_label.setPixmap(QPixmap.fromImage(qt_image))
        self.label.setText(f"To Twój obraz z maseczką!")

    def move_doctor(self):
        window_width = self.width()
        doctor_width = self.doctor_label.width()
        center_start = window_width // 4
        center_end = 3 * window_width // 4

        current_pos = self.doctor_label.x()
        new_pos = current_pos + self.doctor_direction * self.doctor_speed

        if new_pos + doctor_width > center_end or new_pos < center_start:
            self.doctor_direction *= -1

        self.doctor_label.move(min(max(new_pos, center_start), center_end - doctor_width), self.doctor_label.y())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Aplikacja()
    window.show()
    sys.exit(app.exec_())


