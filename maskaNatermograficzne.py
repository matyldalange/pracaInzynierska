import cv2
import numpy as np
import pandas as pd

# Ścieżki do maseczek
mask_front_path = '/Users/matyldalange/Desktop/MaskTheFace-master/masks/templates/surgical.png'
mask_left_path = '/Users/matyldalange/Desktop/MaskTheFace-master/masks/templates/surgical_left.png'
mask_right_path = '/Users/matyldalange/Desktop/MaskTheFace-master/masks/templates/surgical_right.png'

# Wczytanie maseczek
mask_front = cv2.imread(mask_front_path, cv2.IMREAD_UNCHANGED)
mask_left = cv2.imread(mask_left_path, cv2.IMREAD_UNCHANGED)
mask_right = cv2.imread(mask_right_path, cv2.IMREAD_UNCHANGED)

# Wczytanie współrzędnych punktów charakterystycznych z pliku CSV
csv_path = '/Users/matyldalange/Desktop/Charlotte-ThermalFace/S1.csv'
landmarks_df = pd.read_csv(csv_path)


def apply_mask(image, landmarks, mask):
    """
    Nakłada maseczkę na obraz na podstawie punktów charakterystycznych.
    """
    # Punkty kluczowe
    left_cheek = np.array([landmarks['x2'], landmarks['y2']])
    right_cheek = np.array([landmarks['x14'], landmarks['y14']])
    chin = np.array([landmarks['x8'], landmarks['y8']])
    nose = np.array([landmarks['x30'], landmarks['y30']])

    # Dopasowanie szerokości i wysokości maseczki
    face_width = int(np.linalg.norm(right_cheek - left_cheek))
    face_height = int(np.linalg.norm(chin - nose) * 1.4)
    resized_mask = cv2.resize(mask, (face_width, face_height), interpolation=cv2.INTER_AREA)

    # Pozycjonowanie maseczki
    mask_x = left_cheek[0]
    mask_y = nose[1] - int(face_height * 0.5)
    mask_h, mask_w = resized_mask.shape[:2]

    # Przycinanie maseczki
    img_h, img_w = image.shape[:2]
    if mask_x + mask_w > img_w:
        resized_mask = resized_mask[:, :img_w - mask_x]
    if mask_y + mask_h > img_h:
        resized_mask = resized_mask[:img_h - mask_y, :]

    # Nakładanie maseczki
    alpha_mask = resized_mask[:, :, 3] / 255.0
    mask_rgb = resized_mask[:, :, :3]

    for c in range(3):
        image[mask_y:mask_y + resized_mask.shape[0], mask_x:mask_x + resized_mask.shape[1], c] = (
                (1 - alpha_mask) * image[mask_y:mask_y + resized_mask.shape[0], mask_x:mask_x + resized_mask.shape[1],
                                   c] +
                alpha_mask * mask_rgb[:, :, c]
        )
    return image


def process_image(image_path, output_path, landmarks):
    """
    Przetwarza pojedynczy obraz i zapisuje wynik z maseczką.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Błąd wczytywania obrazu: {image_path}")
        return

    # Wybór maseczki (domyślnie "na wprost")
    mask = mask_front

    # Nakładanie maseczki
    masked_image = apply_mask(image, landmarks, mask)

    # Zapis wyniku
    cv2.imwrite(output_path, masked_image)
    print(f"Zapisano obraz z maseczką: {output_path}")


# Przetwarzanie obrazów na podstawie CSV
input_image_path = '/Users/matyldalange/Desktop/Charlotte-ThermalFace/S1/N11102.jpg'
output_image_path = '/Users/matyldalange/Desktop/Charlotte-ThermalFace/S1/N11102_masked.jpg'

# Odfiltrowanie odpowiednich punktów z CSV
image_id = 'N11102'
# Przekształć image_id na format odpowiadający plikowi CSV
csv_id = image_id.replace('N', 'R')

# Filtruj dane z CSV
filtered_landmarks = landmarks_df[landmarks_df['ID'] == csv_id]

if filtered_landmarks.empty:
    print(f"Błąd: brak danych punktów charakterystycznych dla obrazu {image_id} (ID w CSV: {csv_id})")
else:
    landmarks = filtered_landmarks.iloc[0].to_dict()

process_image(input_image_path, output_image_path, landmarks)

