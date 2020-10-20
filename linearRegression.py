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

print(modelo(20))