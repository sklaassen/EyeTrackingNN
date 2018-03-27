import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
from PIL import Image
import glob
import numpy as np
import os

ImageClump = glob.glob(".\Images\*.jpg")
Images = np.zeros((len(ImageClump),28*28),dtype=np.float32)
labels = np.zeros((len(ImageClump),4),dtype=np.float32)
count = 0
for inFile in ImageClump:
    image = np.asarray(Image.open(inFile)).flatten()
    Images[count,:] = image[:]
    print(np.asarray(os.path.splitext(inFile)[0].split("\\")[2].split("_"))[:4])
    #labels[count,:] = np.asarray(os.path.splitext(inFile)[0].split("\\")[2].split("_"))[:3])
    count+=1

print(Images)

batch_size = 128
n_classes = 4 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, 28*28])
x_shaped = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    return out_layer

layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
fc = tf.reshape(layer2, [-1, 7*7*64])

wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
wd2 = tf.Variable(tf.truncated_normal([1000, n_classes], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([n_classes], stddev=0.01), name='bd2')

fc = tf.nn.relu(tf.matmul(fc, wd1) + bd1)
output = tf.matmul(fc, wd2) + bd2


cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output) )
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

hm_epochs = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(hm_epochs):
        _, epoch_loss = sess.run([optimizer,accuracy], feed_dict={x: Images[200:,:], y: labels[200:,:]})

        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)


    print('Accuracy:',sess.run(accuracy,feed_dict={x:Images[:200,:], y:labels[:200,:]}))
