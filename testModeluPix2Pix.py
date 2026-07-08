import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS
import matplotlib.pyplot as plt
import torch

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 256, 256, 3

# Funkcja ładująca zbiór danych testowych (obrazy z maseczkami oraz bez)
def load_test_images(test_folder_A, test_folder_B):
    images, masks = [], []
    for filename in os.listdir(test_folder_A):
        if '_masked' not in filename:
            img_path = os.path.join(test_folder_A, filename)
            mask_filename = filename.split('.')[0] + '_masked.' + filename.split('.')[1]
            mask_path = os.path.join(test_folder_B, mask_filename)

            if os.path.exists(mask_path):
                img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                mask = load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

                img = img_to_array(img) / 127.5 - 1
                mask = img_to_array(mask) / 127.5 - 1

                images.append(img)
                masks.append(mask)

    return np.array(images), np.array(masks)

test_folder_A = '/Users/matyldalange/Desktop/ZBIOR-koniec/bezMasekTest/'
test_folder_B = '/Users/matyldalange/Desktop/ZBIOR-koniec/zMaskamiTest/'

X_test, y_test = load_test_images(test_folder_A, test_folder_B)

generator = load_model('/Users/matyldalange/PycharmProjects/pix2pix-trening,test,uczeniemetryk/saved_models/final_model.h5')

ssim_values = []
lpips_values = []
lpips_model = LPIPS(net='vgg')

#Testowanie wszystich obrazów ze zbioru testowego
for i in range(len(X_test)):
    img = np.expand_dims(X_test[i], axis=0)
    generated_img = generator.predict(img)[0]

    y_test_tensor = torch.tensor(y_test[i]).float()
    generated_img_tensor = torch.tensor(generated_img).float()

    y_test_tensor = y_test_tensor.permute(2, 0, 1).unsqueeze(0)
    generated_img_tensor = generated_img_tensor.permute(2, 0, 1).unsqueeze(0)

    ssim_val = ssim(y_test[i], generated_img, multichannel=True, win_size=3, channel_axis=2, data_range=1.0)
    ssim_values.append(ssim_val)

    lpips_val = lpips_model(y_test_tensor, generated_img_tensor)
    lpips_val = lpips_val.item()
    lpips_values.append(lpips_val)

    if i < 3:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow((X_test[i] * 0.5 + 0.5))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Real Masked Image")
        plt.imshow((y_test[i] * 0.5 + 0.5))
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Generated Masked Image")
        plt.imshow((generated_img * 0.5 + 0.5))
        plt.axis('off')
        plt.show()

mean_ssim = np.mean(ssim_values)
mean_lpips = np.mean(lpips_values)

print(f"Mean SSIM on test set: {mean_ssim}")
print(f"Mean LPIPS on test set: {mean_lpips}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(ssim_values, label="SSIM")
plt.title("SSIM")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lpips_values, label="LPIPS")
plt.title("LPIPS")
plt.legend()

plt.show()


