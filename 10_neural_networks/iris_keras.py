import pandas as pd
import numpy as np
import sklearn.preprocessing as sk_preprocessing
import keras.models as keras_models
import keras.layers as keras_layers
import matplotlib.pyplot as plt


iris_dataset = pd.read_csv("data/iris.csv").sample(frac=1)
column_names = iris_dataset.columns.tolist()

X = iris_dataset[column_names[:-1]]
y = iris_dataset[column_names[-1]]

min_max_scaler = sk_preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

one_hot_encoder = sk_preprocessing.LabelBinarizer()
y = one_hot_encoder.fit_transform(y)

# print(y)

# hidden_neurons = 8

for hidden_neurons in [1, 2, 4, 8, 16]:

    ANN_model = keras_models.Sequential()

    ANN_model.add(keras_layers.Dense(hidden_neurons, activation="relu", input_shape=(4, )))

    ANN_model.add(keras_layers.Dense(3, activation="softmax"))

    ANN_model.compile(loss="mean_squared_error",
                      optimizer="adam",
                      metrics=["accuracy"])

    print(ANN_model.summary())

    training_history = ANN_model.fit(X, y, epochs=2000, batch_size=len(X), validation_split=0.25)

    plt.plot(training_history.history["val_accuracy"], label=str(hidden_neurons))

plt.legend()
plt.show()
