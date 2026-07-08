import cv2
import dlib
import numpy as np
import os
from glob import glob
from retinaface import RetinaFace
import math

# Ścieżki do różnych wersji masek
mask_front_path = '/Users/matyldalange/Desktop/MaskTheFace-master/masks/templates/surgical.png'
mask_left_path = '/Users/matyldalange/Desktop/MaskTheFace-master/masks/templates/surgical_left.png'
mask_right_path = '/Users/matyldalange/Desktop/MaskTheFace-master/masks/templates/surgical_right.png'

mask_front = cv2.imread(mask_front_path, cv2.IMREAD_UNCHANGED)
mask_left = cv2.imread(mask_left_path, cv2.IMREAD_UNCHANGED)
mask_right = cv2.imread(mask_right_path, cv2.IMREAD_UNCHANGED)

shape_predictor_path = '/Users/matyldalange/Desktop/ZBIOR-koniec/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

#Funkcja obliczająca skierowanie twarzy(na wprost, lewy profil, prawy profil)
def calculate_face_angle(face_landmarks):
    left_eye = face_landmarks[36]
    right_eye = face_landmarks[45]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = math.atan2(dy, dx) * 180.0 / math.pi
    return angle


def select_mask(face_landmarks):
    left_eye = face_landmarks[36]
    right_eye = face_landmarks[45]
    nose = face_landmarks[30]

    face_angle = calculate_face_angle(face_landmarks)

    nose_offset_x = nose[0] - (left_eye[0] + right_eye[0]) / 2

    if abs(face_angle) > 20:
        if face_angle > 0:
            return mask_right
        else:
            return mask_left
    else:
        if nose_offset_x > 5:
            return mask_right
        elif nose_offset_x < -5:
            return mask_left
        else:
            return mask_front


def apply_mask_to_face(image, shape):
    face_landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

    mask = select_mask(face_landmarks)

    # Punkty na twarzy do dopasowania maski
    left_cheek = face_landmarks[2]
    right_cheek = face_landmarks[14]
    chin = face_landmarks[8]
    nose = face_landmarks[30]
    left_eye = face_landmarks[36]
    right_eye = face_landmarks[45]

    face_width = np.linalg.norm(right_cheek - left_cheek)
    face_height = np.linalg.norm(chin - nose) * 1.4

    if mask is mask_left:
        face_width *= 1.2
    elif mask is mask_right:
        face_width *= 1.25

    face_width = int(face_width * 1.1)
    face_height = int(face_height)

    scaled_mask = cv2.resize(mask, (face_width, face_height), interpolation=cv2.INTER_AREA)

    mask_x = left_cheek[0] - 3
    mask_y = nose[1] - int(face_height * 0.23)
    if mask is mask_right:
        mask_x = left_cheek[0] - 22

    eye_line_y = min(left_eye[1], right_eye[1])

    if mask_y < eye_line_y:
        mask_y = min(mask_y + int(face_height * 0.1), eye_line_y)

    if mask_y + face_height < nose[1] + 2:
        mask_x -= int(face_width * 0.3)

    img_h, img_w = image.shape[:2]
    mask_h, mask_w = scaled_mask.shape[:2]

    if mask_x < 0:
        scaled_mask = scaled_mask[:, -mask_x:]
        mask_x = 0
    if mask_y < 0:
        scaled_mask = scaled_mask[-mask_y:, :]
        mask_y = 0
    if mask_x + mask_w > img_w:
        scaled_mask = scaled_mask[:, :img_w - mask_x]
    if mask_y + mask_h > img_h:
        scaled_mask = scaled_mask[:img_h - mask_y, :]

    alpha_mask = scaled_mask[:, :, 3] / 255.0
    mask_rgb = scaled_mask[:, :, :3]

    for c in range(3):
        image[mask_y:mask_y + scaled_mask.shape[0], mask_x:mask_x + scaled_mask.shape[1], c] = (
                (1.0 - alpha_mask) * image[mask_y:mask_y + scaled_mask.shape[0], mask_x:mask_x + scaled_mask.shape[1],
                                     c] +
                alpha_mask * mask_rgb[:, :, c]
        )

    return image



def detect_faces_with_retinaface(image):
    faces = RetinaFace.detect_faces(image)
    face_rects = []

    if faces is not None:
        for key, face_info in faces.items():
            facial_area = face_info['facial_area']
            x, y, w, h = facial_area[0], facial_area[1], facial_area[2] - facial_area[0], facial_area[3] - facial_area[1]
            face_rects.append((x, y, w, h))

    return face_rects

def process_image_with_retinaface(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Błąd: nie można załadować obrazu {image_path}")
        return None

    faces = detect_faces_with_retinaface(image)

    if len(faces) == 0:
        print(f"Brak wykrytej twarzy w obrazie {image_path}")
        return image

    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    face = dlib.rectangle(x, y, x + w, y + h)
    shape = predictor(image, face)
    image = apply_mask_to_face(image, shape)

    return image


def mask_images_in_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob(os.path.join(input_dir, '*.jpeg'))

    for image_path in image_paths:
        output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpeg', '_masked.jpg'))

        if os.path.exists(output_path):
            print(f"Pominięto: {output_path} (już istnieje)")
            continue

        masked_image = process_image_with_retinaface(image_path)
        if masked_image is not None:
            cv2.imwrite(output_path, masked_image)
            print(f"Zapisano: {output_path}")
        else:
            print(f"Nie udało się przetworzyć obrazu: {image_path}")


input_dir = '/Users/matyldalange/Desktop/obrazkiAplikacja/'
output_dir = '/Users/matyldalange/Desktop/obrazkiAplikacjaMaski/'
mask_images_in_directory(input_dir, output_dir)

