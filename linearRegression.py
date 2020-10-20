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

modelo = RegModel()
modelo(20)

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

# Plotting Real Values
plt.scatter(X,y,label="Valor Real")

# Plotting Predicted Values
plt.scatter(X, modelo(X), label="Valor Previsto")

plt.legend()
plt.show()