# Redes Neurais Artificiais - Multilayer Perceptron (MLP)
# Algoritimo Tipico: BackPropagation

# Importing Libs
import numpy as np 
import tensorflow as tf 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Generating synthetical data

# Hyperparameters
size = 200000
num_epochs = 10
learning_rate = 0.001

# Generating X Values
x1 = np.random.randint(0,100, size)
x2 = np.random.randint(0,100, size)
x_treino = np.dstack((x1, x2))[0]

# Generating Y Values
y_treino = 3*(x1**(1/2)) + 2*(x2**2)

# Print
print("\nValores e shape de x:")
print(x_treino)
print(x_treino.shape)
print("\nValores e shape de y:")
print(y_treino)
print(y_treino.shape)

# Building MLP Model using Keras

# Defining the sequential layers
modelo_v1 = tf.keras.Sequential()

# Layer 1: Input Layer
modelo_v1.add(tf.keras.layers.Dense(64, input_shape=(2,), activation='sigmoid'))

# Layer 2: Hidden Layer
modelo_v1.add(tf.keras.layers.Dense(128, activation='relu'))

# Layer 3: Exit Layer (Output)
modelo_v1.add(tf.keras.layers.Dense(1))

# Optimizer model with:
modelo_v1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.MSE)

# Trainning Model
print('\n Treinamento do modelo:')
modelo_v1.fit(x_treino, y_treino, epochs=num_epochs)

# Sumário do modelo
print("\nSumário do modelo:")
modelo_v1.summary()

# Evaluating performance model
scores_treino = modelo_v1.evaluate(x_treino, y_treino, verbose=0)
print("\nErro final em treino: {:.0f}".format(scores_treino))

# Testing Model
#Generating new data to x (input values)
x1 = np.array([100, 9, 62, 79, 94, 91, 71, 41])
x2 = np.array([65, 39, 40, 44, 77, 42, 36, 74])
x_teste = np.dstack((x1, x2))[0]

#Real data to y
y_teste = 3*(x1**(1/2)) + 2*(x2**2)

# Making Predictions
print("\nTeste do Modelo:")
y_pred = modelo_v1.predict(x_teste)

# Evaluating performance model
scores_teste = modelo_v1.evaluate(x_teste, y_teste, verbose=0)
print("\nErro final em teste: {:.0f}".format(scores_teste))

print("\n")
for i in range(5):
	print ('''Entrada(x): ({}, {}), Saida(y): ({:.0f}), Previsão do Modelo(y_pred): ({:.0f})'''.format(x1[i], x2[i], y_teste[i], y_pred[i][0]))

print("\n")