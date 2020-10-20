import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Model
class RegModel:
    def __init__(self):
        self.W = tf.Variable(16.0)
        self.b = tf.Variable(10.0)

    def __call__(self, x):
        return self.W * x + self.b

# Defining the true values to Weights and Bias
TRUE_W = 3.0
TRUE_b = 0.5

# Number of examples
NUM_EXAMPLES = 1000

# Generate x values
X = tf.random.normal(shape = (NUM_EXAMPLES,))

# Generate noise
noise = tf.random.normal(shape=(NUM_EXAMPLES,))

# Generate the true value of y
y = X * TRUE_W + TRUE_b + noise

# Calculating the error using cost function
def loss(y, y_pred):
    return tf.reduce_mean(tf.square(y-y_pred))

# Function to execute trainning
def train(modelo, X, y, lr=0.01):
    with tf.GradientTape() as t:
        current_loss = loss(y, modelo(X))

        derivada_W, derivada_b = t.gradient(current_loss, [modelo.W, modelo.b])
        modelo.W.assign_sub(lr * derivada_W)
        modelo.b.assign_sub(lr * derivada_b)

# Creating Model Instance
modelo = RegModel()

# Definimos listas vazias para W e b
Ws, bs = [], []

# Número de épocas (quantas vezes o modelo vai passar pelos dados)
epochs = 20

# Treinamento
print("\n")
for epoch in range(epochs):
    Ws.append(modelo.W.numpy()) 
    bs.append(modelo.b.numpy())

    current_loss = loss(y, modelo(X))

    train(modelo, X, y, lr = 0.1)
    
    print(f"Epoch {epoch}: Loss (Erro): {current_loss.numpy()}")

# Plot
plt.plot(range(epochs), Ws, 'r', range(epochs), bs, 'b')
plt.plot([TRUE_W] * epochs, 'r--', [TRUE_b] * epochs, 'b--')
plt.legend(['W Previsto', 'b Previsto', 'W Real', 'b Real'])
plt.show()

# Plot do valor "real"
plt.scatter(X, y, label = "Valor Real")

# Plot do valor "previsto" pelo nosso modelo
plt.scatter(X, modelo(X), label = "Valor Previsto")

plt.legend()
plt.show()
