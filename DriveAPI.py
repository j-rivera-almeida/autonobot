import pyautogui
import time
import threading
import matplotlib.pyplot as plt
import PIL
from PIL import ImageGrab

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy

import cv2
from predictor import predictor



class Square:
	def __init__(self):
		self.xMin = -1
		self.xMax = -1
		self.yMin = -1
		self.xMax = -1
		self.name = ""
		
	def __str__(self):
		return self.name + " - " + "xMin: " + str(self.xMin) + ", " + "xMax: " + str(self.xMax) + ", " + "yMin: " + str(self.yMin) + ", " + "yMax: " + str(self.yMax)
	
class Cone(Square):
	pass
	
class ArucoMarker(Square):
	def __init__(self):
		Square.__init__(self)
		self.marker = -1
		
	def __str__(self):
		return self.name + " - " + "marker: " + str(self.marker) + ", " + "xMin: " + str(self.xMin) + ", " + "xMax: " + str(self.xMax) + ", " + "yMin: " + str(self.yMin) + ", " + "yMax: " + str(self.yMax)
	

		
class Rover:
	drive = 'drive'
	reverse = 'reverse'
	park = 'park'
	press = 'press'
	release = 'release'
	upArrow = 'up'
	downArrow = 'down'
	leftArrow = 'left'
	rightArrow = 'right'
	keyUp = 'up'
	keyDown = 'down'
	left = 'left'
	right = 'right'
	straight = 'straight'
	
	alive = 'alive'
	forceStopped = 'forceStopped'

	def __init__(self):
		self.gear = Rover.park
		self.pedal = Rover.release
		self.upArrowKeyState = Rover.keyUp
		self.downArrowKeyState = Rover.keyUp
		self.leftArrowKeyState = Rover.keyUp
		self.rightArrowKeyState = Rover.keyUp
		self.direction = Rover.straight
		self.state = Rover.alive
		self.screen = pyautogui.screenshot() #.resize((480,270))
		self.preds = []
		self.number = 0
		self.startUpFinished = False
		# self.monitor = {"top": 0, "left": 0, "width": 800, "height": 600}
		
			
	def PressGas(self):
		self.pedal = Rover.press
		if self.gear == Rover.drive:
			pyautogui.keyDown(Rover.upArrow)
			self.upArrowKeyState = Rover.keyDown
		elif self.gear == Rover.reverse:
			pyautogui.keyDown(Rover.downArrow)	
			self.downArrowKeyState = Rover.keyDown
		
	def ReleaseGas(self):
		pyautogui.keyUp(Rover.upArrow)
		pyautogui.keyUp(Rover.downArrow)
		self.pedal = Rover.release
		self.upArrowKeyState = Rover.keyUp
		self.downArrowKeyState = Rover.keyUp
	
	def PutInDrive(self):
		pyautogui.keyUp(Rover.downArrow)
		self.downArrowKeyState = Rover.keyUp
		self.gear = Rover.drive
		
	def PutInReverse(self):
		pyautogui.keyUp(Rover.upArrow)
		self.upArrowKeyState = Rover.keyUp
		self.gear = Rover.reverse
		
	def TurnLeft(self):
		pyautogui.keyUp(Rover.rightArrow)
		pyautogui.keyDown(Rover.leftArrow)
		self.rightArrowKeyState = Rover.keyUp
		self.leftArrowKeyState = Rover.keyDown
		self.direction = Rover.left
		return self
	
	def TurnRight(self):
		pyautogui.keyUp(Rover.leftArrow)
		pyautogui.keyDown(Rover.rightArrow)
		self.leftArrowKeyState = Rover.keyUp
		self.rightArrowKeyState = Rover.keyDown
		self.direction = Rover.right
		return self
	
	def GoStraight(self):
		pyautogui.keyUp(Rover.leftArrow)
		pyautogui.keyUp(Rover.rightArrow)
		self.leftArrowKeyState = Rover.keyUp
		self.rightArrowKeyState = Rover.keyUp
		self.direction = Rover.straight
		return self
		
	def ForceStop(self):
		self.state = Rover.forceStopped	
		pyautogui.keyUp(Rover.upArrow)
		pyautogui.keyUp(Rover.downArrow)
		pyautogui.keyUp(Rover.leftArrow)
		pyautogui.keyUp(Rover.rightArrow)
		
	def Finish(self):
		self.ForceStop()
		
	def DriveFor(self, timeLength):
		time.sleep(timeLength)
		
	def For(self, timeLength):
		time.sleep(timeLength)
		
	def LoadCurveStraightModel(self, json):
		self.curveStraightJson = json
		
	def LoadCurveStraightWeights(self, weights):
		self.curveStraightWeights = weights
	
	def StartCurveStraightModel(self):
		# load json and create model
		json_file = open(self.curveStraightJson, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights(self.curveStraightWeights)
		print("Loaded model from disk")
		self.model = loaded_model
		
	def LoadYoloModel(self, config):
		self.yoloConfig = config
		
	def LoadYoloWeights(self, weights):
		self.yoloWeights = weights
		
	def StartYoloModel(self):
		args = {
			"classes": ["Cone","Aruco 1", "Aruco 2"],
			"weights": self.yoloWeights,
			"config": self.yoloConfig
		}
		self.yolo = predictor(args["classes"], args["weights"], args["config"])
		
	def AnalyzeStartUp(self):
		print("AnalyzeStartUp Not Implemented")
		
	def AnalyzeScript(self):
		self.AnalyzeStartUp()
		self.startUpFinished = True			

		while self.state != Rover.forceStopped:
			self.Analyze()
		
	def Analyze(self):
		print("Analyze Not Implemented")	
		
	def PredictYolo(self, image):
		output = self.yolo.predict_coord(image)
		
		arucoMarkers = self.get_aruco_task(output)
		cones = self.get_cones(output)
					
		arucoMarkerSquares = []
		coneSquares = []
		
		if arucoMarkers is not -1:
			am_left,am_right = self.get_lr_cone(arucoMarkers)
			
			arucoMarkerLeftSquare = ArucoMarker()
			arucoMarkerLeftSquare.name = "Aruco Marker Left"
			arucoMarkerLeftSquare.marker = am_left[0]
			arucoMarkerLeftSquare.xMin = am_left[2]
			arucoMarkerLeftSquare.xMax = am_left[4]
			arucoMarkerLeftSquare.yMin = am_left[3]
			arucoMarkerLeftSquare.yMax = am_left[5]
			
			arucoMarkerRightSquare = ArucoMarker()
			arucoMarkerRightSquare.name = "Aruco Marker Right"
			arucoMarkerRightSquare.marker = am_right[0]
			arucoMarkerRightSquare.xMin = am_right[2]
			arucoMarkerRightSquare.xMax = am_right[4]
			arucoMarkerRightSquare.yMin = am_right[3]
			arucoMarkerRightSquare.yMax = am_right[5]
			
			arucoMarkerSquares.append(arucoMarkerLeftSquare)
			arucoMarkerSquares.append(arucoMarkerRightSquare)
			

		if cones is not None and output is not None:
			cone_left,cone_right = self.get_lr_cone(cones)
			
			coneLeftSquare = Square()
			coneLeftSquare.name = "Cone Left"
			coneLeftSquare.xMin = cone_left[2]
			coneLeftSquare.xMax = cone_left[4]
			coneLeftSquare.yMin = cone_left[3]
			coneLeftSquare.yMax = cone_left[5]
			
			coneRightSquare = Square()
			coneRightSquare.name = "Cone Right"
			coneRightSquare.xMin = cone_right[2]
			coneRightSquare.xMax = cone_right[4]
			coneRightSquare.yMin = cone_right[3]
			coneRightSquare.yMax = cone_right[5]
						
			coneSquares.append(coneLeftSquare)
			coneSquares.append(coneRightSquare)
			
		return arucoMarkerSquares, coneSquares
	
	
	def PredictCurveStraight(self, image):
		prediction = self.model.predict(image)
		if prediction[0][0] > prediction[0][1]:
			return "straight"
		return "curve"
		
	def CaptureScreen(self):
		self.screen = pyautogui.screenshot()
		
	def InterpretImageAsRGBResize(self, width, height):
		array = numpy.array(self.screen.resize((width, height)))
		return array
		
	def InterpretImageAsRGB(self):
		array = numpy.array(self.screen)
		return array
		
	def InterpretImageAsRGBResizeInArray(self, width, height):
		array = numpy.array(self.screen.resize((width, height)))
		array = array[numpy.newaxis,...]
		return array
		
	def InterpretImageAsRGBInArray(self):
		array = numpy.array(self.screen)
		array = array[numpy.newaxis,...]
		return array
		
	def InterpretImageAsBGRResize(self, width, height):
		array = (numpy.array(self.screen.resize((width, height))))[:,:,::-1]
		return array
		
	def InterpretImageAsBGR(self):
		array = (numpy.array(self.screen))[:,:,::-1]
		return array
		
	def InterpretImageAsBGRResizeInArray(self, width, height):
		array = (numpy.array(self.screen.resize((width, height))))[:,:,::-1]
		array = array[numpy.newaxis,...]
		return array
		
	def InterpretImageAsBGRInArray(self):
		array = (numpy.array(self.screen))[:,:,::-1]
		array = array[numpy.newaxis,...]
		return array
		
	def Input(self):
		char = input()
		self.ForceStop()
		
	def DriveStartUp(self):
		print("DriveStartUp Not Implemented")	
		
	def DriveScript(self):
		while self.startUpFinished == False:
			time.sleep(.25)
		
		self.DriveStartUp()
		
		while self.state != Rover.forceStopped:
			self.Drive()
		
	def Drive(self):
		print("Drive Not Implemented")
		
		
				
		
		
		
			
	def Run(self):
		print(3)
		time.sleep(0.5)
		print(2)
		time.sleep(0.5)
		print(1)
		time.sleep(0.5)
		
		self.analyzeThread = threading.Thread(target=self.AnalyzeScript)
		self.analyzeThread.start()
		
		self.drivingThread = threading.Thread(target=self.DriveScript)
		self.drivingThread.start()
		
		self.inputThread = threading.Thread(target=self.Input)
		self.inputThread.start()
		
		self.analyzeThread.join()
		self.drivingThread.join()
		self.inputThread.join()
		
		
		
		
		
	#YOLO Support Functions
	def get_area(self, item):
		coord1 = item[2]
		coord2 = item[3]
		coord3 = item[4]
		coord4 = item[5]
		width = coord4-coord2
		height = coord3-coord1
		area = width*height
		return area

	def get_aruco_task(self, output):
		arucos = []
		if output is None:
			#print("no arucos seen")
			objective = 4
			return -1
		for t in output:
			if t[0] == "Aruco 2" or t[0] == "Aruco 1":
				arucos.append(t)
		#print(arucos)
		if len(output) == 0:
			objective = 4
			#print("no arucos seen")
			return -1
		if len(output) == 1:
			objective = 1
			#print("only one arucos seen")
			return -1
		temp = [None,None]
		max1 = 0
		max2 = 0
		#print("finding max arucos")
		for x in range(0,len(arucos)):
			if arucos[x][1] >= 0.5:
				area = self.get_area(arucos[x])
				if area > max2:
					if area > max1:
						max2 = max1
						max1 = area
						temp[1] = temp[0]
						temp[0] = arucos[x]
						
						
					else:
						max2 = area
						temp[1] = arucos[x]
		if temp[0] is None or temp[1] is None:
			objective = 4
			return -1
		#print(max1,max2)
		if max1*0.5 > max2:
			#print("aruco 1 is too big compared to aruco 2")
			objectve = 1
			return -1
		if temp[0][0] != temp[1][0]:
			#print("two different aruco markers")
			objective = 1
			return -1
		else:
			if '2' in temp[0][0]:
				objective = 2
				#print("aruco 2 is seen")
				return temp
			else:
				#print("aruco 1 is seen")
				objective = 3
				return temp
				
	def get_cones(self, output):
		cones = []
		if output is None or len(output) is 0:
			#print("RETURNING NONE")
			return None
		if output is not None:
			for t in output:
				if t[0] == "Cone" and t[1] > 0.75:
					cones.append(t)
		temp = [None,None]
		max1 = 0
		max2 = 0
		#print("finding max cones")
		for x in range(0,len(cones)):
			if cones[x][1] >= 0.5:
				area = self.get_area(cones[x])
				if area > max2:
					if area > max1:
						max2 = max1
						max1 = area
						temp[1] = cones[0]
						temp[0] = cones[x]    
					else:
						max2 = area
						temp[1] = cones[x]
		if temp[0] is None:
			return None
		return temp
		
	def color_cones(self, img,cones):
		#print(cones)
		for c in cones:
			img = self.draw_rect(img,[c[2],c[3],c[4],c[5]])
		return img
		
	def get_slope(self, ams):
		am_left,am_right = self.get_lr_cone(ams)
		return float((am_right[5]-am_left[5])/(am_right[2]-am_left[4]))
		
	def get_lr_cone(self, ams):
		am_left = []
		am_right = []
		if ams[0][2] < ams[1][2]:
			am_left = ams[0]
			am_right = ams[1]
		else:
			am_left = ams[1]
			am_right = ams[0]
		return [am_left,am_right]
		
	def display_image(self, img,name):
		scale_percent = 60 # percent of original size
		width = int(img.shape[1] * scale_percent / 100)
		height = int(img.shape[0] * scale_percent / 100)
		dim = (width, height)
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		cv2.imshow(name,resized)
		cv2.waitKey(1)
		
	def draw_rect(self, img,rect,color=(0,0,255)):
		img = cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),color,3)
		return img
		
	def draw_rect_aruco(self, img,rect1,rect2):
		im2 = cv2.rectangle(img,(rect1[0],rect1[1]),(rect1[2],rect1[3]),(255,0,0),3)
		im2 = cv2.rectangle(im2,(rect2[0],rect2[1]),(rect2[2],rect2[3]),(0,255,255),3)
		return im2
