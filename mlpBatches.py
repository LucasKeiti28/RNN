# Multilayer perceptron with batches
# Used when has a huge amount of data and there is no enough space in memory to process data

# Import Libs
import numpy as np
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Hyperparameters
size = 200000
num_epochs = 20
learning_rate = 0.0001
n_batches = 10000
batch_size = 100

# Generating X Values
x1 = np.random.randint(0,100, size)
x2 = np.random.randint(0,100, size)
x_treino = np.dstack((x1,x2))[0]

# Generating Y Values
y_treino = 3*(x1**(1/2)) + 2*(x2**2)

# Print
print("\nValores e shape de x:")
print(x_treino)
print(x_treino.shape)
print("\nValores e shape de y:")
print(y_treino)
print(y_treino.shape)

modelo_v3 = tf.keras.Sequential()
modelo_v3.add(tf.keras.layers.Dense(64, input_shape=(2,), activation='sigmoid'))
modelo_v3.add(tf.keras.layers.Dense(128, activation = 'relu'))
modelo_v3.add(tf.keras.layers.Dense(128, activation = 'relu'))
modelo_v3.add(tf.keras.layers.Dense(2))

# Creating Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Creting cost function
mse = tf.keras.losses.MSE

# Compile Model
modelo_v3.compile(optimizer=optimizer, loss=mse)

# Preparing batches data trainning
dataset_treino = tf.data.Dataset.from_tensor_slices((x_treino, y_treino))
dataset_treino = dataset_treino.repeat().shuffle(x_treino.shape[0]).batch(batch_size)

print ("\nIniciando o Treinamento do Modelo. Por favor aguarde!")

# Loop through batches
for i, (batch_xs_treino, batch_ys_treino) in enumerate(dataset_treino.take(n_batches), 1):
    with tf.GradientTape() as tape:
        y_pred = modelo_v3(batch_xs_treino)
        loss = mse(batch_ys_treino[0], y_pred[0])
        grads = tape.gradient(loss, modelo_v3.trainable_variables)
        processed_grads = [g for g in grads]
        grads_and_vars = zip(processed_grads, modelo_v3.trainable_variables)
    # Optimization weights and bias
    optimizer.apply_gradients(grads_and_vars)

print('\nTaxa de Erro final em Treino:', np.mean(loss.numpy()))

# Sumário do modelo
print("\nSumário do modelo:")
modelo_v3.summary()

# Testando o Modelo

# Gerando novos dados para x
x1 = np.array([100, 9, 62, 79, 94, 91, 71, 41])
x2 = np.array([65, 39, 40, 44, 77, 42, 36, 74])
x_teste = np.dstack((x1, x2))[0]

# Gerando novos dados para y
y_teste = 3*(x1**(1/2)) + 2*(x2**2)

# Preparting batches with testing data
dataset_teste = tf.data.Dataset.from_tensor_slices((x_teste, y_teste))
dataset_teste = dataset_teste.repeat().shuffle(x_teste.shape[0]).batch(batch_size)

print ("\nIniciando a Avaliação com Dados de Teste. Por favor aguarde!")

for i, (batch_xs_teste, batch_ys_teste) in enumerate(dataset_teste.take(n_batches), 1):
    with tf.GradientTape() as tape:
        y_pred = modelo_v3(batch_xs_teste)
        loss = mse(batch_ys_teste[0], y_pred)

print ('\nTaxa de Erro Final em Teste: ', np.mean(loss.numpy()))

# Making predictions with model created
print("\nFazendo Previsões com o Modelo...")
y_pred = modelo_v3.predict(x_teste)

print("\n")

for i in range(5):
    print ('''Entrada(x): ({}, {}), Saida(y): ({:.0f}), Previsão do Modelo(y_pred): ({:.0f})'''.format(x1[i], x2[i], y_teste[i], y_pred[i][0]))

print("\n")


# Salvando o modelo
modelo_v3.save('modelo_v3.h5')

# Carregando o modelo salvo
novo_modelo_v3 = tf.keras.models.load_model('modelo_v3.h5')
