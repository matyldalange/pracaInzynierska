import cv2
import pandas as pd
import numpy as np
import math

csv_path = "/Users/matyldalange/Desktop/Charlotte-ThermalFace/S1.csv"
image_path = "/Users/matyldalange/Desktop/Charlotte-ThermalFace/S1/N11185.jpg"

mask_front_path = '/Users/matyldalange/Desktop/MaskTheFace-master/masks/templates/surgical.png'
mask_left_path = '/Users/matyldalange/Desktop/MaskTheFace-master/masks/templates/surgical_left.png'
mask_right_path = '/Users/matyldalange/Desktop/MaskTheFace-master/masks/templates/surgical_right.png'

mask_front = cv2.imread(mask_front_path, cv2.IMREAD_UNCHANGED)
mask_left = cv2.imread(mask_left_path, cv2.IMREAD_UNCHANGED)
mask_right = cv2.imread(mask_right_path, cv2.IMREAD_UNCHANGED)

if any(mask is None for mask in [mask_front, mask_left, mask_right]):
    raise ValueError("Nie udało się wczytać jednej lub więcej masek.")

image = cv2.imread(image_path)
if image is None:
    raise ValueError("Nie udało się wczytać obrazu termograficznego.")

landmarks_df = pd.read_csv(csv_path, sep=",")
image_id = "R11185"

landmarks = landmarks_df[landmarks_df['ID'] == image_id]
if landmarks.empty:
    raise ValueError(f"Nie znaleziono danych dla ID {image_id}")

landmarks = landmarks.iloc[0]
nose = (int(landmarks['x27']), int(landmarks['y27']))
chin = (int(landmarks['x8']), int(landmarks['y8']))
left_cheek = (int(landmarks['x0']), int(landmarks['y0']))
right_cheek = (int(landmarks['x16']), int(landmarks['y16']))

cheek_difference = right_cheek[0] - left_cheek[0]

#Wybór rodzaju maski
if cheek_difference < -70:
    selected_mask = mask_left
    x_offset = 31
    y_offset = 15
    height_scale = 1.5
    width_scale = 1.6
elif cheek_difference > 70:
    selected_mask = mask_right
    x_offset = -20
    y_offset = 40
    height_scale = 1.1
    width_scale = 1.4
else:
    selected_mask = mask_front
    x_offset = 0
    y_offset = 35
    height_scale = 1.2
    width_scale = 1.2

face_width = int(np.linalg.norm(np.array(right_cheek) - np.array(left_cheek)) * width_scale)
face_height = int(np.linalg.norm(np.array(chin) - np.array(nose)) * height_scale)

scaled_mask = cv2.resize(selected_mask, (face_width, face_height), interpolation=cv2.INTER_AREA)

mask_x = int(nose[0] - face_width / 2) + x_offset
mask_y = int(nose[1] - face_height / 2) + y_offset

y1, y2 = max(0, mask_y), min(image.shape[0], mask_y + scaled_mask.shape[0])
x1, x2 = max(0, mask_x), min(image.shape[1], mask_x + scaled_mask.shape[1])
mask_y1, mask_y2 = max(0, -mask_y), scaled_mask.shape[0] - max(0, mask_y + scaled_mask.shape[0] - image.shape[0])
mask_x1, mask_x2 = max(0, -mask_x), scaled_mask.shape[1] - max(0, mask_x + scaled_mask.shape[1] - image.shape[1])


if scaled_mask.shape[2] == 4:
    alpha_mask = scaled_mask[mask_y1:mask_y2, mask_x1:mask_x2, 3] / 255.0
    for c in range(3):
        image[y1:y2, x1:x2, c] = (
            alpha_mask * scaled_mask[mask_y1:mask_y2, mask_x1:mask_x2, c] +
            (1 - alpha_mask) * image[y1:y2, x1:x2, c]
        )
else:
    raise ValueError("Maska nie zawiera kanału alfa.")

output_path = "/Users/matyldalange/Desktop/Charlotte-ThermalFace/S1/N11185_masked_corrected_adjusted0.jpg"
cv2.imwrite(output_path, image)
