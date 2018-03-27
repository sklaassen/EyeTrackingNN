import pygame
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

running = True
#up down left right
directions = {'l':0,'lu':0,'u':0,'ru':0,'r':0,'rd':0,'d':0,'ld':0,'c':0}
lewayPercent = 33/2

def main():
	delayTime = 0.01
	nextTime = time.time()
	global running, screen
	cv2.namedWindow("preview")
	vc = cv2.VideoCapture(0)
	rval, frame = vc.read()
	faceCascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


	pygame.init()
	screen = pygame.display.set_mode((width, height))
	pygame.display.set_caption("eye tracking")
	screen.fill(BLACK)
	pygame.display.update()


	offset = 20
	curr=0
	dest = 1
	step = 0
	maxStep = 50
	pos = [0,0]
	locks=np.array([[0+offset,0+offset],[width-offset,0+offset],[width-offset,height-offset],[0+offset,height-offset]],np.float32)
	path, dirs, files = os.walk("./Images/").__next__()
	counter = len(files)

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
		if len(faces)==1 and time.time() >= nextTime:
			nextTime = time.time() + delayTime
			step= step+1
			if step == maxStep:
				step = 0
				curr = dest
				dest = (dest+1)%len(locks)
			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
				gray = gray[y+10:y+h-10, x+10:x+w-10]
				gray = cv2.resize(gray,(28,28))
				for index in directions:
					directions[index] = 0

				if pos[0] <width/2-width*(1/lewayPercent):
					#left
					if pos[1] <height/2-height*(1/lewayPercent):
						#up
						directions['lu'] = 1
					elif pos[0] >height/2+height*(1/lewayPercent):
						#down
						directions['ld'] = 1
					else:
						#none
						directions['l'] = 1
				elif pos[0] >width/2+width*(1/lewayPercent):
					#right
					if pos[1] <height/2-height*(1/lewayPercent):
						#up
						directions['ru'] = 1
					elif pos[0] >height/2+height*(1/lewayPercent):
						#down
						directions['rd'] = 1
					else:
						#none
						directions['r'] = 1
				else:
					#none
					if pos[1] <height/2-height*(1/lewayPercent):
						#up
						directions['u'] = 1
					elif pos[0] >height/2+height*(1/lewayPercent):
						#down
						directions['d'] = 1
					else:
						#none
						directions['c'] = 1
				filename = "./Images/"
				for index in directions:
					filename = filename + str(directions[index]) + "_"

				filename += str(counter)
				cv2.imwrite(filename + ".jpg",gray)
				counter = counter +1

		pos = (locks[dest]-locks[curr])/maxStep*step+locks[curr]
		pygame.draw.circle(screen, BLUE, pos, 20)



		pygame.display.update()



	vc.release()
	cv2.destroyAllWindows()

	#release is not actually stoping
	vc.stop()

if __name__ == '__main__':
	main()
