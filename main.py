#import dependencies
import numpy as np
import tensorflow as tf
import cv2
import keras
import pandas as pd
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import layers
from random import seed
from random import randint

gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# plot the curve if 1
plot_loss_curve = 1

# test the model if 1
test_the_model = 1

# import dataset
wafers = np.load('wafer/data.npy')
noised_wafers = wafers + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=wafers.shape)

# model architecture
input_wafer = keras.Input(shape=(26,26,3))
x = layers.Conv2D(16, (2, 2), activation='relu', padding='same')(input_wafer)
# x = layers.MaxPooling2D((2, 2), padding='same')(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = layers.MaxPooling2D((2, 2), padding='same')(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(3, (2, 2), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (2, 2), activation='sigmoid', padding='same')(x)
autoencoder = keras.Model(input_wafer, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='mse')

loss =autoencoder.fit(wafers,
                noised_wafers,
                epochs=20,
                batch_size=32)

autoencoder.save('pretrained/ae_weight.h5')

if plot_loss_curve:
    history_df = pd.DataFrame(loss.history)
    fig = plt.figure(figsize=(10, 4))
    plt.plot(history_df.index.to_list(), history_df["loss"], linewidth=2) #history_df["accuracy"]
    plt.ylabel("Training Loss", fontsize=12)
    plt.xlabel("Epochs", fontsize=12)
    plt.title("Loss Function over Time", fontsize=14, fontstyle='italic')
    plt.show()

if test_the_model:
    seed()
    rand_num = randint(0, len(wafers))
    wafer = wafers[rand_num]
    rebuilt_wafer = autoencoder.predict(np.expand_dims(wafers[rand_num], axis=0))
    rebuilt_wafer = rebuilt_wafer.reshape(wafers.shape[1:])

    plt.subplot(1,2,1)
    plt.imshow(wafer)
    plt.subplot(1,2,2)
    plt.imshow(rebuilt_wafer)
    plt.show()