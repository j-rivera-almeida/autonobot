import threading
from PIL import ImageGrab
import numpy as np
import cv2
from predictor import predictor
import DriveAPI
import time
import mss
args = {
    "classes": ["Cone","Aruco 1", "Aruco 2"],
    "weights": "yolov3_last.weights",
    "config": "yolov3.cfg"
}
cone_seen = False
pred = predictor(args["classes"],args["weights"], args["config"])
objective = 4

#1 is back up
#2 is through cone
#3 is around cone
#4 is go straight
#5 is stop
#6 is right
#7 is left
        

##class screenshotter(threading.Thread):
##    frame = None
##    def __init__(self):
##        threading.Thread.__init__(self)
##        threading.Thread(target=self.run)
##    def run(self):
##        while True:
##            image = ImageGrab.grab()
##            image_np = np.array(image)
##            self.frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
##    def get_image(self):
##        return self.frame

class screenshotter(threading.Thread):
    frame = None
    
    def __init__(self):
        threading.Thread.__init__(self)
        threading.Thread(target=self.run)
        self.monitor = {"top": 0, "left": 0, "width": 1800, "height": 1200}
    def run(self):
        while True:
            with mss.mss() as sct:
                img = np.array(sct.grab(self.monitor))
                #self.frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.frame = img[:,:,:3]
    def get_image(self):
        return self.frame
		
class predictor(threading.Thread):
    old = True
    img = None
    output = None
    def __init__(self):
        threading.Thread.__init__(self)
        threading.Thread(target=self.run)
    def run(self):
        while True:
            if self.old and self.img is not None:
                time_1 = time.time()
                self.output = pred.predict_coord(self.img)
                self.old = False
                time_2 = time.time()
                print("TIME :: ",time_2-time_1)
    def set_image(self,img):
        self.img = img
        self.old = True
    def get_output(self):
        return self.output
		
def get_area(item):
    coord1 = item[2]
    coord2 = item[3]
    coord3 = item[4]
    coord4 = item[5]
    width = coord4-coord2
    height = coord3-coord1
    area = width*height
    return area

def get_aruco_task(output):
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
            area = get_area(arucos[x])
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
            print("aruco 1 is seen")
            #objective = 3
            return temp
def get_cones(output):
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
            area = get_area(cones[x])
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
def color_cones(img,cones):
    #print(cones)
    for c in cones:
        img = draw_rect(img,[c[2],c[3],c[4],c[5]])
    return img
def get_slope(ams):
    am_left,am_right = get_lr_cone(ams)
    return float((am_right[5]-am_left[5])/(am_right[2]-am_left[4]))
def get_lr_cone(ams):
    am_left = []
    am_right = []
    if ams[0][2] < ams[1][2]:
        am_left = ams[0]
        am_right = ams[1]
    else:
        am_left = ams[1]
        am_right = ams[0]
    return [am_left,am_right]
    
def display_image(img,name):
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow(name,resized)
    cv2.waitKey(1)
def draw_rect(img,rect,color=(0,0,255)):
    img = cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),color,3)
    return img
def draw_rect_aruco(img,rect1,rect2):
    im2 = cv2.rectangle(img,(rect1[0],rect1[1]),(rect1[2],rect1[3]),(255,0,0),3)
    im2 = cv2.rectangle(im2,(rect2[0],rect2[1]),(rect2[2],rect2[3]),(0,255,255),3)
    return im2
	
sc_thread = screenshotter()
pd_thread = predictor()
sc_thread.start()
pd_thread.start()
time.sleep(0.5)
rover = DriveAPI.Rover()


while True:
    img = sc_thread.get_image()
    pd_thread.set_image(img)
    output = pd_thread.get_output()
    #display_image(img,"raw")
    #print(output)
    im_copy = np.copy(img)
    ams = get_aruco_task(output)
    cones = get_cones(output)
    im_cones = im_copy
    if cones is not None:
        im_cones = color_cones(im_copy,cones)
    
    im_aruco = im_cones
    #print(output)
    if objective is 1:
        #rover.PutInReverse()
        #rover.PressGas()
        #rover.DriveFor(0.01)
        #rover.ReleaseGas()
        print("REVERSE")
    if objective is 4:
        #rover.PutInDrive()
        #rover.PressGas()
        #rover.DriveFor(0.01)
        #rover.ReleaseGas()
        print("ONWARD")
    #1 is back up
    #2 is through cone
    #3 is around cone
    #4 is go straight
    #5 is stop
    #6 is right
    #7 is left
    if objective is 6:
        #rover.PutInDrive()
        #rover.PressGas()
        #rover.TurnRight()
        #rover.DriveFor(0.01)
        #rover.ReleaseGas()
        print("TURNING RIGHT")
    if objective is 7:
        #rover.PutInDrive()
        #rover.PressGas()
        #rover.TurnLeft()
        #rover.DriveFor(0.01)
        #rover.ReleaseGas()
        print("TURNING LEFT")
    if objective is 5:
        #rover.ReleaseGas()
        print("RELEASING GAS")
    #print(ams,objective)
    
    if ams is not -1:
        cone_seen = True
        im_aruco = draw_rect_aruco(im_cones,[ams[0][2],ams[0][3],ams[0][4],ams[0][5]],[ams[1][2],ams[1][3],ams[1][4],ams[1][5]])
        am_left,am_right = get_lr_cone(ams)
        
        
    if cones is not None and output is not None:
        cone_left,cone_right = get_lr_cone(cones)
        print(cone_left,cone_right)
        #print("L CONE R: ",str(am_left[4]))                                 
        #print("R CONE L: ",str(am_right[2]))
        if get_area(cones[0]) <= 3000:
            objective = 4
        elif get_area(cones[0]) > 3000:
            if cone_left[4] > 900:
                objective = 6
            elif cone_right[2] < 1000:
                objective = 7
            else:
                objective = 4
    if cones is not None and output is not None:
        #print("AREAS")
        #print(get_area(cones[0]),get_area(cones[1]))
        pass
    display_image(im_aruco,"marked")
