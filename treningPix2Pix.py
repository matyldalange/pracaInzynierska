import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.utils import plot_model


IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 256, 256, 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
BATCH_SIZE = 1
EPOCHS = 40

#Funkcja ładująca obrazy
def load_images(train_folder_A, train_folder_B, val_folder_A, val_folder_B):
    def load_folder_images(folder_A, folder_B):
        images, masks = [], []
        for filename in os.listdir(folder_A):
            if '_masked' not in filename:
                img_path = os.path.join(folder_A, filename)
                mask_filename = filename.split('.')[0] + '_masked.' + filename.split('.')[1]
                mask_path = os.path.join(folder_B, mask_filename)

                if os.path.exists(mask_path):
                    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    mask = load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

                    img = img_to_array(img) / 127.5 - 1  # Skala [-1, 1]
                    mask = img_to_array(mask) / 127.5 - 1  # Skala [-1, 1]

                    images.append(img)
                    masks.append(mask)

        return np.array(images), np.array(masks)

    X_train, y_train = load_folder_images(train_folder_A, train_folder_B)
    X_val, y_val = load_folder_images(val_folder_A, val_folder_B)

    return X_train, y_train, X_val, y_val


train_folder_A = '/Users/matyldalange/Desktop/ZBIOR-koniec/ParyTrening/'
train_folder_B = '/Users/matyldalange/Desktop/ZBIOR-koniec/ParyMaskedTrening/'
val_folder_A = '/Users/matyldalange/Desktop/ZBIOR-koniec/ParyWalidacja/'
val_folder_B = '/Users/matyldalange/Desktop/ZBIOR-koniec/ParyMaskedWalidacja/'

X_train, y_train, X_val, y_val = load_images(train_folder_A, train_folder_B, val_folder_A, val_folder_B)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


def ssim_metric(y_true, y_pred):
    y_true = (y_true + 1) / 2  # SSIM zostaje przeskalowany do zakresu [0, 1]
    y_pred = (y_pred + 1) / 2
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

#Funkcja konfigurująca generator
def build_generator():
    inputs = Input(shape=IMG_SHAPE)
    def downsample(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = Dropout(0.2)(d)
        return d
    def upsample(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        u = Conv2DTranspose(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        u = LeakyReLU(alpha=0.2)(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = Concatenate()([u, skip_input])
        return u

    d1 = downsample(inputs, 64, bn=False)
    d2 = downsample(d1, 128)
    d3 = downsample(d2, 256)
    d4 = downsample(d3, 512)
    d5 = downsample(d4, 512)
    d6 = downsample(d5, 512)
    d7 = downsample(d6, 512)

    u1 = upsample(d7, d6, 512)
    u2 = upsample(u1, d5, 512)
    u3 = upsample(u2, d4, 512)
    u4 = upsample(u3, d3, 256)
    u5 = upsample(u4, d2, 128)
    u6 = upsample(u5, d1, 64)

    u7 = Conv2DTranspose(IMG_CHANNELS, kernel_size=4, strides=2, padding='same', activation='tanh')(u6)
    return Model(inputs, u7)

#Funkcja konfigurująca dyskryminator
def build_discriminator():
    img_A = Input(shape=IMG_SHAPE)
    img_B = Input(shape=IMG_SHAPE)
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    def discriminator_block(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = Dropout(0.2)(d)
        return d

    d1 = discriminator_block(combined_imgs, 64, bn=False)
    d2 = discriminator_block(d1, 128)
    d3 = discriminator_block(d2, 256)
    d4 = discriminator_block(d3, 512)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([img_A, img_B], validity)

#Funkcja konfigurująca model Pix2Pix
def build_pix2pix(generator, discriminator):
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=Adam(0.0001, 0.5),
        metrics=[ssim_metric]
    )
    discriminator.trainable = False

    img_A = Input(shape=IMG_SHAPE)
    fake_B = generator(img_A)
    valid = discriminator([img_A, fake_B])

    pix2pix = Model(img_A, [valid, fake_B])
    pix2pix.compile(
        loss=['binary_crossentropy', 'mae'],
        loss_weights=[1, 100],
        optimizer=Adam(0.0001, 0.5),
        metrics=[ssim_metric]
    )

    return pix2pix


def train(generator, discriminator, pix2pix, X_train, y_train, X_val, y_val):
    valid = np.ones((BATCH_SIZE,) + (IMG_HEIGHT // 2 ** 4, IMG_WIDTH // 2 ** 4, 1))
    fake = np.zeros((BATCH_SIZE,) + (IMG_HEIGHT // 2 ** 4, IMG_WIDTH // 2 ** 4, 1))

    d_losses, g_losses, ssim_scores = [], [], []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        for batch_i in range(len(X_train) // BATCH_SIZE):
            idx = np.random.randint(0, len(X_train), BATCH_SIZE)
            imgs_A = X_train[idx]
            imgs_B = y_train[idx]

            fake_B = generator.predict(imgs_A)

            d_loss_real = discriminator.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = discriminator.train_on_batch([imgs_A, fake_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = pix2pix.train_on_batch(imgs_A, [valid, imgs_B])

            ssim_score = ssim_metric(imgs_B, fake_B)

            d_losses.append(d_loss[0])
            g_losses.append(g_loss[0])
            ssim_scores.append(ssim_score.numpy())

            print(f"[Epoch {epoch + 1}/{EPOCHS}] [D loss: {d_loss[0]}] [G loss: {g_loss[0]}] [SSIM: {ssim_score.numpy()}]")

        # Zapisywanie wag co 10 epok
        if (epoch + 1) % 10 == 0:
            weight_filename = f'wagi_ponownyTrening_epoch_{epoch + 1}.hdf5'
            generator.save_weights(weight_filename)
            print(f"Zapisano wagi generatora do pliku {weight_filename}")

    # Rysowanie wykresów
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(d_losses, label="D Loss")
    plt.title("Dyskryminator Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(g_losses, label="G Loss")
    plt.title("Generator Loss")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(ssim_scores, label="SSIM")
    plt.title("SSIM")
    plt.legend()

    plt.show()

    plot_model(generator, show_shapes=True, dpi=64)
    plot_model(discriminator, show_shapes=True, dpi=64)

generator = build_generator()
discriminator = build_discriminator()
pix2pix = build_pix2pix(generator, discriminator)

train(generator, discriminator, pix2pix, X_train, y_train, X_val, y_val)

generator.save('final_model.h5')



