from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

model_path = '/Users/matyldalange/PycharmProjects/pix2pix-trening,test,uczeniemetryk/final_model.h5'

model = load_model(model_path)

# Zapisywanie architektury modelu Pix2Pix do pliku png
plot_model(model, to_file='pix2pix_architecture.png', show_shapes=True, show_layer_names=True)


