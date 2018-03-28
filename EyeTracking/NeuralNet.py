import tensorflow as tf
from PIL import Image
import glob
import numpy as np
import os

ImageClump = glob.glob(".\Images\*.jpg")
Images = np.zeros((len(ImageClump),28*28),dtype=np.float32)
labels = np.zeros((len(ImageClump),9),dtype=np.float32)

shuffle = np.random.permutation(len(ImageClump))
trainToTestRatio = int(8/10*len(ImageClump))

count = 0
for inFile in ImageClump:
    image = np.asarray(Image.open(inFile)).flatten()
    Images[shuffle[count],:] = image[:]
    #print(np.asarray(os.path.splitext(inFile)[0].split("\\")[2].split("_"))[:9])
    labels[shuffle[count],:] = np.asarray(os.path.splitext(inFile)[0].split("\\")[2].split("_"))[:9]
    count+=1

#print(Images)
trainImages = Images[:trainToTestRatio,:]
testImages = Images[trainToTestRatio:,:]
trainLabels = labels[:trainToTestRatio,:]
testLabels = labels[trainToTestRatio:,:]
print("Total data Count:",len(ImageClump))
print("Training data Count: " ,len(trainImages))
print("Testing data Count: ",len(testImages))
batch_size = 128
n_classes = 9

# tf Graph input
x = tf.placeholder(tf.float32, [None, 28*28])
x_shaped = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    out_layer += bias
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,padding='SAME')
    return out_layer

layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
fc = tf.reshape(layer2, [-1, 7*7*64])

wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.3), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')

wd2 = tf.Variable(tf.truncated_normal([1000, n_classes], stddev=0.3), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([n_classes], stddev=0.01), name='bd2')

fc = tf.nn.relu(tf.matmul(fc, wd1) + bd1)
output = tf.matmul(fc, wd2) + bd2


cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output) )
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

hm_epochs = 100
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess,'/tmp/eyeTracker.ckpt')
    for epoch in range(hm_epochs):
        totcost = 0
        batch_count = 0
        for batch in range(0,len(trainImages),batch_size):
            _, epoch_loss = sess.run([optimizer,cost], feed_dict={x: trainImages[batch:batch+batch_size,:], y: trainLabels[batch:batch+batch_size,:]})
            totcost += epoch_loss
            batch_count +=1

        saver.save(sess,'/tmp/eyeTracker.ckpt')
        train_acc = sess.run(accuracy, feed_dict={x: trainImages, y: trainLabels})
        test_acc = sess.run(accuracy, feed_dict={x: testImages, y: testLabels})

        print('Epoch', epoch, '/',hm_epochs,' acc1:',train_acc,'cost: ',totcost/batch_count,' acc2: ',test_acc)


    print('Accuracy:',sess.run(accuracy,feed_dict={x: testImages, y: testLabels}))
