# EyeTrackingNN
Testing Convolution neural network to do eye tracking

main.py is used to generate the Images that are fed into the neural network. it does this by moving a circle around the screen and tracking the face throught a webcam. the inmages that are saved are only of the face and of a size of 28 by 28 pixles. the labels of the data is stored in the name of the image. they corespond to if the person is looking in 8 diffrent sections of the screen (left, top left, top, top right, right, bottom right, bottom, bottom left, center)

NeuralNetwork.py is a convelutional neural network program that learns from scratch the data accumulated from the main meathod. It is able to do much better than random but it can still be improved greatly
