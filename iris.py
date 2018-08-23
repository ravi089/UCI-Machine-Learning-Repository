# UCI Machine Learning Repository <Iris Dataset>.
# <Classification>
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Prepare the data.
iris = pd.read_csv('../data/iris_dataset.csv', header=None)
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
n_samples = iris.shape[0]

x_vals = iris.iloc[:, :4].values
y_data = iris.iloc[:, 4].values
y_vals = np.zeros([n_samples, 3])
for i in range(n_samples):
    y_vals[i][mapping[y_data[i]]] = 1
    
# Train/Test split.
train_x, test_x, train_y, test_y = train_test_split(x_vals,
                                                    y_vals,
                                                    test_size = 0.2)
# Hyper parameters.
training_epochs = 200
learning_rate = 0.01
n_input = 4
n_hidden_1 = 50
n_hidden_2 = 25
n_hidden_3 = 10
n_classes = 3
display_step = 10

# Graph input.
X = tf.placeholder(shape=[None, n_input], dtype=tf.float32)
Y = tf.placeholder(shape=[None, n_classes], dtype=tf.float32)

# Weights and Biases for the network.
net_param = {
    'layer_1': {
        'weight': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'bias': tf.Variable(tf.random_normal([n_hidden_1]))
    },
    'layer_2': {
        'weight': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'bias': tf.Variable(tf.random_normal([n_hidden_2]))
    },
    'layer_3': {
        'weight': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'bias': tf.Variable(tf.random_normal([n_hidden_3]))
    },
    'layer_out': {
        'weight': tf.Variable(tf.random_normal([n_hidden_3, n_classes])),
        'bias': tf.Variable(tf.random_normal([n_classes]))
    }
}

# Network.
def neural_network(input_layer):
    layer_1 = tf.add(tf.matmul(input_layer, net_param['layer_1']['weight']), net_param['layer_1']['bias'])
    layer_2 = tf.add(tf.matmul(layer_1, net_param['layer_2']['weight']), net_param['layer_2']['bias'])
    layer_3 = tf.add(tf.matmul(layer_2, net_param['layer_3']['weight']), net_param['layer_3']['bias'])
    out_layer = tf.add(tf.matmul(layer_3, net_param['layer_out']['weight']), net_param['layer_out']['bias'])
    return out_layer

logits = neural_network(X)
prediction = tf.nn.softmax(logits)

# Loss and optimizer for the network.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# Evaluation.
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Training/Testing.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1, training_epochs + 1):
        sess.run(optimizer, feed_dict = {X: train_x, Y: train_y})
        if step % display_step == 0 or step == 1:
            cost, acc = sess.run([loss, accuracy], feed_dict = {X: train_x,
                                                                Y: train_y})
            print ('Step ' + str(step) + ', Minibatch Loss = ' + \
                   '{:.4f}'.format(cost) + ', Training Accuracy = ' + \
                   '{:.4f}'.format(acc))
    print ('Training Done!!!')
    print ('Testing Accuracy:', \
           sess.run(accuracy, feed_dict = {X: test_x,
                                           Y: test_y}))
