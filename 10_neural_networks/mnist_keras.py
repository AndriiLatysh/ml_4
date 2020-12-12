import pandas as pd
import numpy as np
import sklearn.preprocessing as sk_preprocessing
import keras.models as keras_models
import keras.layers as keras_layers
import keras.datasets as keras_datasets
import keras.utils as keras_utils
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt


np.set_printoptions(linewidth=200)

(X_train, y_train), (X_test, y_test) = keras_datasets.mnist.load_data()

image_index = 42
#
# print(y_train[image_index])
# print(X_train[image_index])
# plt.imshow(X_train[image_index], cmap="Greys")
# plt.show()

X_train_count = X_train.shape[0]
X_test_count = X_test.shape[0]
X_height = X_train.shape[1]
X_width = X_train.shape[2]

input_shape = (X_height, X_width, 1)

print(f"Train size: {X_train_count}")
print(f"Test size: {X_test_count}")

X_train = X_train.reshape(X_train_count, X_height, X_width, 1).astype("float32") / 255
X_test = X_test.reshape(X_test_count, X_height, X_width, 1).astype("float32") / 255

y_train = keras_utils.to_categorical(y_train)
y_test = keras_utils.to_categorical(y_test)

# print(y_train[image_index])

CNN_model = keras_models.Sequential()

CNN_model.add(keras_layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
CNN_model.add(keras_layers.MaxPooling2D(pool_size=(2, 2)))

CNN_model.add(keras_layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
CNN_model.add(keras_layers.MaxPooling2D(pool_size=(2, 2)))

CNN_model.add(keras_layers.Flatten())

CNN_model.add(keras_layers.Dense(64, activation="relu"))
CNN_model.add(keras_layers.Dropout(0.2))

CNN_model.add(keras_layers.Dense(10, activation="softmax"))

CNN_model.compile(loss="mean_squared_error",
                  optimizer="adam",
                  metrics=["accuracy"])

print(CNN_model.summary())

CNN_model.fit(x=X_train, y=y_train, epochs=10, batch_size=100, shuffle=True)

CNN_model_evaluation = CNN_model.evaluate(x=X_test, y=y_test, verbose=0)

print("CV accuracy: {}".format(CNN_model_evaluation[1]))

y_predicted = CNN_model.predict(X_test)

confusion_matrix = sk_metrics.confusion_matrix(y_test.argmax(axis=1), y_predicted.argmax(axis=1))
print(confusion_matrix)

CNN_model.save("models/MNIST_CNN.h5")
