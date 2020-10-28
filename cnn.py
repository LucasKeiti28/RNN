# CNNs : Convolutional Neural Network

# Importing Packages
import input_data 
import tensorflow as tf 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Handle script to run with TF2
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False 
tf.compat.v1.disable_eager_execution()

# Dataset 
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# Parameteres
learning_rate = 0.001 
training_iters = 100000
batch_size = 128 
display_step = 10

n_input = 784 # shape 28 x 28
n_classes = 10  # total classes 10

# Dropout (Generalization method, reducing overfitting and increase the accuracy in testing data)
dropout = 0.75

# Creating graphs
x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.compat.v1.placeholder(tf.float32) # dropout 

# Converting input (x) to tensor 
_X = tf.reshape(x, shape = [-1, 28, 28, 1])

# Function to create model

# Convolutional Layer
def conv2d(img, w, b): 
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
        input = img, filters=w, strides=[1,1,1,1], padding='VALID'
    ), b))

# Pooling after convolutional process
def max_pool(img, k):
    return tf.nn.max_pool2d(input=img, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

# Intiate values to weights and bias
# Weights
wc1 = tf.Variable(tf.random.normal([5,5,1,32]))
wc2 = tf.Variable(tf.random.normal([5,5,32,64]))
wd1 = tf.Variable(tf.random.normal([4*4*64, 1024]))
wout = tf.Variable(tf.random.normal([1024, n_classes]))

# Bias
bc1 = tf.Variable(tf.random.normal([32]))
bc2 = tf.Variable(tf.random.normal([64]))
bd1 = tf.Variable(tf.random.normal([1024]))
bout = tf.Variable(tf.random.normal([n_classes]))

# First Layer (Convolutional)
conv1 = conv2d(_X, wc1, bc1)

# First Layer (Pooling)
conv1 = max_pool(conv1, k=2)

# Applying Dropout
conv1 = tf.nn.dropout(conv1, 1 - (keep_prob))

# Second Layer (Convolutional)
conv2 = conv2d(conv1, wc2, bc2)

# Second Layer (Pooling)
conv2 = max_pool(conv2, k=2)

# Applying Dropout
conv2 = tf.nn.dropout(conv2, 1 - (keep_prob))

# Full Connected Layer
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1), bd1)) 
dense1 = tf.nn.dropout(dense1, 1 - (keep_prob))

# Prediction
pred = tf.add(tf.matmul(dense1, wout), bout)

# Cost Function e Otimização
cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = tf.stop_gradient( y)))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Avaliando o Modelo
correct_pred = tf.equal(tf.argmax(input=pred,axis=1), tf.argmax(input=y,axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32))

# Initialize variables
init = tf.compat.v1.global_variables_initializer()

# Session
with tf.compat.v1.Session() as sess:
    sess.run(init)
    step = 1
    # Keep trainning until reache the maximum number of iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate accuracy
            acc = sess.run(accuracy, feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 1})
            # Loss Calculation
            loss = sess.run(cost, feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 1})
            print ("Iteração " + str(step*batch_size) + ", Perda = " + "{:.6f}".format(loss) + ", Acurácia em Treino = " + "{:.5f}".format(acc))
        step += 1
    print ("Otimização Concluída!")
    # Calculando acurácia para 256 mnist test images
    print ("Acurácia em Teste:", sess.run(accuracy, feed_dict = {x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))

