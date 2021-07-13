import DriveAPI
import threading
import time

import pyautogui
import time
import threading
import matplotlib.pyplot as plt
import PIL

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy

from win32api import GetSystemMetrics

Width = GetSystemMetrics(0)
Height = GetSystemMetrics(1)
			
class MyRover(DriveAPI.Rover):
    
	
    def AnalyzeStartUp(rover):
        rover.StartCurveStraightModel()
        rover.StartYoloModel()
        rover.curveStraightPrediction = "straight"
        rover.prevPrediction = "straight"
        # #Runs once when the rover is started, then Analyze is called in a loop
        rover.arucoMarkers = []
        rover.prevPredictionMarker = "straight"
        rover.starttime = 0
        rover.finishtime = 0
        
        
		# #Here is where to do any setup items, such as putting the rover in drive and pressing the gas
        rover.PutInDrive()
			
    def Analyze(rover):		
		# capture the screen
        rover.CaptureScreen()
		
		# then translate the screenshot into the forms the CurveStraight and Yolo predictors need
        imageBGR = rover.InterpretImageAsBGR()
        imageRGB = rover.InterpretImageAsRGBResizeInArray(480, 270)
		
		# send the images through the deep learning models
        
        rover.curveStraightPrediction = rover.PredictCurveStraight(imageRGB)	
        rover.arucoMarkers, rover.cones = rover.PredictYolo(imageBGR)	
		
# =============================================================================
# 		# rover.curveStraightPrediction is string that is either "straight" or "curve"
# 		
# 		# rover.cones is a list of objects of type Cone and has the follow members
# 		# name
# 		# xMin
# 		# xMax
# 		# yMin
# 		# yMax
# 		
# 		# rover.arucoMarkers is a list of objects of type ArucoMarker and has the follow members
# 		# name
# 		# xMin
# 		# xMax
# 		# yMin
# 		# yMax
# 		# marker
# 		
# =============================================================================
        for arucoMarker in rover.arucoMarkers:
            print(arucoMarker)
            
        #for cone in rover.cones:
        #    print(cone)
        for path in rover.curveStraightPrediction:
            print(path)
            
				
    def DriveStartUp(rover):
        #pass
        rover.PressGas()
        rover.DriveFor(3)
        rover.ReleaseGas()
        
		
    def Drive(rover):
		# access rover.curveStraightPrediction, rover.curveStraightPrediction, and rover.curveStraightPrediction
		# here to make driving decisions
	
        if rover.curveStraightPrediction == "straight":
            print("straight")
            rover.finishtime = time.time()
            rover.GoStraight()
            rover.PressGas()
            rover.DriveFor(0.1)
            rover.ReleaseGas()
            rover.prevPrediction = "straight"
            rover.starttime = time.time()
            
        elif rover.curveStraightPrediction == "curve" and rover.prevPrediction != "curve":
            print("curve")
            rover.finishtime = time.time()    			
            rover.TurnLeft()
            rover.PressGas()
            rover.DriveFor(0.05)
            rover.GoStraight()
            rover.DriveFor(0.1)
            rover.ReleaseGas()
            rover.prevPrediction = "curve"
            rover.starttime = time.time()
            
        elif (rover.finishtime-rover.starttime) > 3:
            print("reset triggered")
            rover.prevPrediction = "reset"
            
        if rover.arucoMarkers != []:
            for arucoMarker in rover.arucoMarkers:
                if arucoMarker.name == "Aruco Marker Right" and rover.prevPredictionMarker != "strong left" and arucoMarker.xMax < (Width*0.25):
                    print("strong left")
                    rover.TurnLeft()
                    rover.PressGas()
                    rover.DriveFor(0.25)
                    rover.GoStraight()
                    rover.DriveFor(0.1)
                    rover.ReleaseGas()
                    rover.prevPredictionMarker = "strong left"
                
                elif arucoMarker.name == "Aruco Marker Right" and rover.prevPredictionMarker != "weak left" and arucoMarker.xMax < (Width*0.50):
                    print("weak left")
                    rover.TurnLeft()
                    rover.PressGas()
                    rover.DriveFor(0.1)
                    rover.GoStraight()
                    rover.DriveFor(0.05)
                    rover.ReleaseGas()
                    rover.prevPredictionMarker = "weak left"
                
                
                elif arucoMarker.name == "Aruco Marker Left" and rover.prevPredictionMarker != "strong left" and arucoMarker.xMin > (Width*0.75):
                    print("strong right")
                    rover.TurnRight()
                    rover.PressGas()
                    rover.DriveFor(0.25)
                    rover.GoStraight()
                    rover.DriveFor(0.1)
                    rover.ReleaseGas()
                    rover.prevPredictionMarker = "strong left"
                    
                elif arucoMarker.name == "Aruco Marker Left" and rover.prevPredictionMarker != "weak right" and arucoMarker.xMin > (Width*0.5):
                    print("weak right")
                    rover.TurnRight()
                    rover.PressGas()
                    rover.DriveFor(0.1)
                    rover.GoStraight()
                    rover.DriveFor(0.05)
                    rover.ReleaseGas()
                    rover.prevPredictionMarker ="weak right"
                    
                elif len(rover.arucoMarkers) > 0:
                    print("Full steam ahead!")
                    rover.GoStraight()
                    rover.PressGas()
                    rover.DriveFor(0.1)
                    rover.ReleaseGas()
                    #rover.prevPredictionMarker = "straight"
                    
                else:
                    print("reset predictions")
                    rover.prevPrediction = "reset"
                    rover.prevPredictionMarker = "reset"
                   
                #print(arucoMarker)
	
				
def RunRover():
	rover = MyRover()
	
	#Load in the curve straight model (the json file) and weights
	rover.LoadCurveStraightModel("model.json")
	rover.LoadCurveStraightWeights("model.h5")
	
	#Load in the Yolo model (the cfg file) and weights
	rover.LoadYoloModel("yolov3.cfg")
	rover.LoadYoloWeights("../../yolov3_last.weights")
	
	rover.Run()


if __name__ == "__main__":
	RunRover()