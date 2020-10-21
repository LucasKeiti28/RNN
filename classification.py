# Classification with Tensorflow

# Importing Libs
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import data
fashion_mnist = keras.datasets.fashion_mnist
(x_treino, y_treino), (x_teste, y_teste) = fashion_mnist.load_data()

# Analysing the data set
print("\n")
print("Shape trainning data:", x_treino.shape)
print("Shape testing data", x_teste.shape)
print("Shape labels data", np.unique(y_treino))
print("\n")

# Defining labels (number to label title)
class_names = {
    i:cn for i, cn in enumerate([
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ])
}

# Function to plotting images
def plot(images, labels, predictions = None):

    # Creating grid with 5 collumns
    n_cols = min(5, len(images))
    n_rows = math.ceil(len(images) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (n_cols + 3, n_rows + 4))

    # Checking if has predictions to plot
    if predictions is None:
        predictions = [None] * len(labels)

    # Loop through data to plot in figure
    for i, (x,y_true, y_pred) in enumerate(zip(images, labels, predictions)):
        ax = axes.flat[i]
        ax.imshow(x, cmap = plt.cm.binary)

        # Print the real value of y
        ax.set_title(f"L:{class_names[y_true]}")

        # Print the predicted value of y
        if y_pred is not None:
            ax.set_xlabel(f"Prev: {class_names[y_pred]}")
        
        ax.set_xticks([])
        ax.set_yticks([])

# Plotting some images
plot(x_treino[:15], y_treino[:15])
plt.show()

# Standartization data
x_treino = x_treino/255.0
x_teste = x_teste/255.0

# Creating Model
modelo = keras.Sequential(layers=[
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(228, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compila o modelo, adding: optimizer and cost function
modelo.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# Trainning model
modelo.fit(x_treino, y_treino, batch_size = 60, epochs = 20, validation_split = 0.2)

# Evaluated model
loss, accuracy = modelo.evaluate(x_teste, y_teste)
print(f"Acuracia do modelo = {accuracy*100:.2f} %")

# Predictions with model
print(modelo.predict_classes(x_teste))

# Saving predictions
preds = modelo.predict_classes(x_teste)

# Getting some randomly images
rand_idxs = np.random.permutation(len(x_teste))[:20]

# Plot das previaões
plot(x_teste[rand_idxs], y_teste[rand_idxs], preds[rand_idxs])
plt.show()