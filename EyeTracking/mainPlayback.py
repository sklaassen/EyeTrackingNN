import pygame
import tensorflow as tf
import cv2
import numpy as np
import os
import time
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)
BLACK = (0,0,0)
WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
TEXTCOLOR = (  0,   0,  0)
(width, height) = (1366, 728)
directions = {'l':0,'lu':0,'u':0,'ru':0,'r':0,'rd':0,'d':0,'ld':0,'c':0}

running = True
#up down left right
ImageSize = 100
ofs = 20
def main():
	global running, screen
	pos = [(ofs,height//2),(ofs,ofs),(width//2,ofs),(width-ofs,ofs),(width-ofs,height//2),(width-ofs,height-ofs),(width//2,height-ofs),(ofs,height-ofs),(width//2,height//2)]
	cv2.namedWindow("preview")
	vc = cv2.VideoCapture(0)
	rval, frame = vc.read()
	faceCascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
#------------------------------------
	n_classes = 9
	x = tf.placeholder(tf.float32, [None, ImageSize*ImageSize])
	x_shaped = tf.reshape(x, [-1, ImageSize, ImageSize, 1])

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
	fc = tf.reshape(layer2, [-1, (ImageSize//4)*(ImageSize//4)*64])

	wd1 = tf.Variable(tf.truncated_normal([(ImageSize//4)*(ImageSize//4) * 64, 1000], stddev=0.3), name='wd1')
	bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')

	wd2 = tf.Variable(tf.truncated_normal([1000, n_classes], stddev=0.3), name='wd2')
	bd2 = tf.Variable(tf.truncated_normal([n_classes], stddev=0.01), name='bd2')

	fc = tf.nn.relu(tf.matmul(fc, wd1) + bd1)
	output = tf.matmul(fc, wd2) + bd2
	predict = tf.argmax(output, 1)
	saver = tf.train.Saver()
	sess = tf.Session()
	saver.restore(sess,'/tmp/eyeTracker.ckpt')
#-------------------
	for index in directions:
		print(index)
	pygame.init()
	screen = pygame.display.set_mode((width, height))
	pygame.display.set_caption("eye tracking")
	screen.fill(BLACK)
	pygame.display.update()

	while running:
		ev = pygame.event.get()

		for event in ev:
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.MOUSEBUTTONDOWN:
				running = False

		screen.fill(BLACK)


		cv2.imshow("preview", frame)
		rval, frame = vc.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
		if len(faces)==1:
			for (xx, y, w, h) in faces:
				cv2.rectangle(frame, (xx, y), (xx+w, y+h), (0, 255, 0), 2)
				gray = gray[y+10:y+h-10, xx+10:xx+w-10]
				gray = np.array(cv2.resize(gray,(ImageSize,ImageSize))).flatten()
				img = np.zeros((1,ImageSize*ImageSize),dtype=np.float32)
				img[0,:] = gray[:]
				prediction = sess.run(predict, {x: img})
				#print(prediction)
				pygame.draw.circle(screen, BLUE, pos[int(prediction[0])], 20)

		pygame.display.update()
	vc.release()
	cv2.destroyAllWindows()

	#release is not actually stoping
	vc.stop()

if __name__ == '__main__':
	main()
