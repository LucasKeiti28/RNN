# Linear Regression with Tensorflow

# Importing Libs
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

# Defining real values to weights and bias
TRUE_W = 3.0
TRUE_b = 0.5

# Number of examples
NUM_EXAMPLES = 1000

# Generation x values, array with 1-dimension
X = tf.random.normal(shape = (NUM_EXAMPLES, ))

# Generation Noise
noise = tf.random.normal(shape = (NUM_EXAMPLES,))

# Generation the true value of y (tensor with 1-dimension)
y = X * TRUE_W + TRUE_b + noise

# Calculate the error using a cost function
def loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

# Function to execute trainning
def train(modelo, X, y, lr=0.01):
    with tf.GradientTape() as t:
        current_loss = loss(y, modelo(X))

    # Calculate the deviation using gradient, using cost function
    derivada_W, derivada_b = t.gradient(current_loss, [modelo.W, modelo.b])
    modelo.W.assign_sub(lr * derivada_W)
    modelo.b.assign_sub(lr * derivada_b)

# Creating model
modelo = RegModel()

# Defining empty lists to receive W and b
Ws, bs = [], []

# Number of epochst, how many times the model will pass through data
epochs = 20

# Trainning the model
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
plt.legend(['W Predicted', 'b Predicted', 'W Real', 'b Real'])
plt.show()

# Ploting real values
plt.scatter(X, y, label = "Valor Real")

# Plotting predicted values by model trained
plt.scatter(X, modelo(X), label = "Valor Previsto")

plt.legend()
plt.show()
