# The Logistic Regression Algorithm to classifie images 

# Importing Packages
import math
import tensorflow as tf 
from tensorflow.keras.datasets import fashion_mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Importing Datasets
(x_treino, y_treino), (x_teste, y_teste) = fashion_mnist.load_data()

# Normalizing Data
x_treino, x_teste = x_treino/255., x_teste/255.

# Adjusting shape
# Flatting matrix to 1-dimensional Tensor
# Matrix (28 x 28) => Tensor (,784)
x_treino = tf.reshape(x_treino, shape=(-1, 784))
x_teste = tf.reshape(x_teste, shape=(-1, 784))

# Building Model

# Defining Initial Weights and Bias
pesos = tf.Variable(tf.random.normal(shape=(784, 10), dtype=tf.float64))
vieses = tf.Variable(tf.random.normal(shape=(10,), dtype=tf.float64))

# Logistic Regression Function
def logistic_regression(x):
    lr = tf.add(tf.matmul(x, pesos), vieses)
    return lr

# Cross Entropy Function
def cross_entropy(y_true, y_pred):
    y_true = tf.one_hot(y_true, 10)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred)
    return tf.reduce_mean(loss)

# Otipmizing the cost function
def grad(x, y):
    with tf.GradientTape() as tape:
        y_pred = logistic_regression(x)
        loss_val = cross_entropy(y, y_pred)
    return tape.gradient(loss_val, [pesos, vieses])

# Defining Hyperparameters
n_batches = 10000
batch_size = 128
learning_rate = 0.01

# Create optimizer using SGD (Stochastic Gradient Descent)
optimizer = tf.optimizers.SGD(learning_rate)

# Function to calculate accuracy
def accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype = tf.int32)
    preds = tf.cast(tf.argmax(y_pred, axis=1), dtype = tf.int32)
    preds = tf.equal(y_true, preds)
    return tf.reduce_mean(tf.cast(preds, dtype = tf.float32))

# Preparing batches to trainning data
dataset_treino = tf.data.Dataset.from_tensor_slices((x_treino, y_treino))
dataset_treino = dataset_treino.repeat().shuffle(x_treino.shape[0]).batch(batch_size)

print ("\nIniciando o Treinamento")

# Trainning Cycle
for batch_numb, (batch_xs_treino, batch_ys_treino) in enumerate(dataset_treino.take(n_batches), 1):
    # Calculate Gradient
    gradientes = grad(batch_xs_treino, batch_ys_treino)

    # Optimizing Weights with gradient
    optimizer.apply_gradients(zip(gradientes, [pesos, vieses]))

    # Make Prediction
    y_pred = logistic_regression(batch_xs_treino)

    # Calculate Error
    loss = cross_entropy(batch_ys_treino, y_pred)

    # Calculate Accuracy
    acc = accuracy(batch_ys_treino, y_pred)

    # Print
    print("Número do Batch: %i, Erro do Modelo: %f, Acurácia em Treino: %f" % (batch_numb, loss, acc))

print ("\nTreinamento concluído!")

# Testando o Modelo

# Preparando os dados de teste
dataset_teste = tf.data.Dataset.from_tensor_slices((x_teste, y_teste))
dataset_teste = dataset_teste.repeat().shuffle(x_teste.shape[0]).batch(batch_size)

print ("\nIniciando a Avaliação com Dados de Teste. Por favor aguarde!")

# Loop pelos dados de teste, previsões e cálculo da acurácia
for batch_numb, (batch_xs_teste, batch_ys_teste) in enumerate(dataset_teste.take(n_batches), 1):
    y_pred = logistic_regression(batch_xs_teste)
    acc = accuracy(batch_ys_teste, y_pred)
    acuracia = tf.reduce_mean(tf.cast(acc, tf.float64))

print("\nAcurácia em Teste: %f" % acuracia)

print("\nFazendo Previsão de Uma Imagem:")

# Obtendo os dados de algumas imagens
dataset_teste = tf.data.Dataset.from_tensor_slices((x_teste, y_teste))
dataset_teste = dataset_teste.repeat().shuffle(x_teste.shape[0]).batch(10)

# Fazendo previsões
for batch_numb, (batch_xs, batch_ys) in enumerate(dataset_teste.take(1), 1):
    # print("\nImagem:", batch_xs)
    print("\nClasse Real:", batch_ys)
    y_pred = tf.math.argmax(logistic_regression(batch_xs), axis = 1)
    # y_pred = logistic_regression(batch_xs)
    print("Classe Prevista:", y_pred)

print("\nExemplo de Peso e Viés Aprendidos:")
print(pesos[2,9])
print(vieses[2])
print("\n")