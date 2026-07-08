import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, Concatenate, Conv2DTranspose

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 256, 256, 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


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


generator = build_generator()

generator.load_weights('/Users/matyldalange/PycharmProjects/pix2pix-trening,test,uczeniemetryk/wagi_zmienione_epoch_40.hdf5')

generator.save('/Users/matyldalange/PycharmProjects/pix2pix-trening,test,uczeniemetryk/final_model_test40.h5')

print("Model z wagami zapisany w pliku final_model.h5")
