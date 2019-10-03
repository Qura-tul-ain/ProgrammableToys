import numpy as np
import tensorflow as tf
from dataset import *
import os
import argparse
import cv2
from glob import glob
from keras.preprocessing import image
import random

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train CNN to classify images into N number of Classes.')
parser.add_argument("command", metavar="<command>", help="'train' or 'test'")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
datasetTrain = load_cached(cache_path="../cache files/Train_cache.pkl",in_dir="../Dataset/Train")

num_classes_Train = datasetTrain.num_classes
datasetValidation = load_cached(cache_path="../cache files/Validation_cache.pkl",in_dir="../Dataset/Test")

num_classes_Validation = datasetValidation.num_classes
image_paths_train, cls_train, labels_train = datasetTrain.get_training_set()
image_paths_test, cls_test, labels_test = datasetValidation.get_training_set()

training_iters = 10
learning_rate = 0.0001
batch_size = 1
n_input_W = 80
n_input_H = 80
n_classes = 12

#both placeholders are of type float
x = tf.placeholder("float", [None, n_input_H,n_input_W,3])

y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.layers.max_pooling2d(x, pool_size=[k, k], strides=[k, k, ], padding='SAME')

weights = {
    'wc1': tf.get_variable('W0', shape=(5,5,3,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.get_variable('W2', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wcAli': tf.get_variable('ali', shape=(5,5,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'crowd1': tf.get_variable('crowd1', shape=(5,5,128,128), initializer=tf.contrib.layers.xavier_initializer()),
    'crowd2': tf.get_variable('crowd2', shape=(5,5,128,128), initializer=tf.contrib.layers.xavier_initializer()),
    'crowd3': tf.get_variable('crowd3', shape=(5,5,128,512), initializer=tf.contrib.layers.xavier_initializer()),
    'crowd4': tf.get_variable('crowd4', shape=(5, 5, 512, 512), initializer=tf.contrib.layers.xavier_initializer()),
    'crowd5': tf.get_variable('crowd5', shape=(5, 5, 512, 256), initializer=tf.contrib.layers.xavier_initializer()),
    'crowd6': tf.get_variable('crowd6', shape=(5,5,256,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W3', shape=(64,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wd2': tf.get_variable('W4', shape=(64,32), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(32,n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'BAli': tf.get_variable('BAli1', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'Bcrowd1': tf.get_variable('Bcrowd1', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'Bcrowd2': tf.get_variable('Bcrowd2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'Bcrowd3': tf.get_variable('Bcrowd3', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'Bcrowd4': tf.get_variable('Bcrowd4', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'Bcrowd5': tf.get_variable('Bcrowd5', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'Bcrowd6': tf.get_variable('Bcrowd6', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bd2': tf.get_variable('B4', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B5', shape=(12), initializer=tf.contrib.layers.xavier_initializer()),
}




def conv_net(x, weights, biases):

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    #conv1 = conv2d(conv1, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3A = conv2d(conv3, weights['wcAli'], biases['BAli'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3A, k=2)

    conv4 = conv2d(conv3, weights['crowd1'], biases['Bcrowd1'])
    conv4A = conv2d(conv4, weights['crowd2'], biases['Bcrowd2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv4 = maxpool2d(conv4A, k=2)

    conv5 = conv2d(conv4, weights['crowd2'], biases['Bcrowd2'])
    conv5A = conv2d(conv5, weights['crowd3'], biases['Bcrowd3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv5 = maxpool2d(conv5A, k=2)


    conv6 = conv2d(conv5, weights['crowd4'], biases['Bcrowd4'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv6 = maxpool2d(conv6, k=2)

    conv7 = conv2d(conv6, weights['crowd5'], biases['Bcrowd5'])
    conv8 = conv2d(conv7, weights['crowd6'], biases['Bcrowd6'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv8 = maxpool2d(conv8, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    dropout1=tf.layers.dropout(conv8, rate=0.5)
    fc1 = tf.reshape(dropout1, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc3 = tf.nn.relu(fc2)
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    return out


pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image.
# and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Initializing the variables
init = tf.global_variables_initializer()
test_X = load_images(image_paths_test)

test_Y=labels_test
saver = tf.train.Saver()

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
total_iterations=0
