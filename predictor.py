import cv2
import numpy as np
import time

class predictor:
  def __init__(self, classes, weights, config):

    #Public
    self.classes = classes
    self.weights = weights
    self.config = config

    #initialization of network parameters
    self.NET = cv2.dnn.readNet(weights, config)
    self.INPUT_SIZE = (608,608)

    #detection of output layers
    layer_names = self.NET.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in self.NET.getUnconnectedOutLayers()]
    self.OUTPUT_LAYERS = output_layers

    #Bounding box colors
    self.COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    #Thresholds for prediction confidence
    self.CONFIDENCE_THRESH = 0.5
    self.NMS_TRESH = 0.4

  def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
     label = str(self.classes[class_id])
     color = self.COLORS[class_id]
     cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
     cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  def predict_image(self, img):
      #performs a prediciton and returns an image with drawn bounding box
      #image data
      Width = img.shape[1]
      Height = img.shape[0]
      scale = 0.00392

      #prepare image for prediction
      blob = cv2.dnn.blobFromImage(img, scale, self.INPUT_SIZE, (0,0,0), True, crop=False)
      self.NET.setInput(blob)

      #peform the prediction
      outs = self.NET.forward(self.OUTPUT_LAYERS)
      #prediction results
      class_ids = []
      confidences = []
      boxes = []

      #interpret prediciton results
      for out in outs:
          for detection in out:
              scores = detection[5:]
              class_id = np.argmax(scores)
              confidence = scores[class_id]
              if confidence > 0.5:

                  center_x = int(detection[0] * Width)
                  center_y = int(detection[1] * Height)
                  w = int(detection[2] * Width)
                  h = int(detection[3] * Height)
                  x = center_x - w / 2
                  y = center_y - h / 2
                  class_ids.append(class_id)
                  confidences.append(float(confidence))
                  boxes.append([x, y, w, h])

      #Peform non-max supression
      indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESH, self.NMS_TRESH)

      for i in indices:
          i = i[0]
          box = boxes[i]
          x = box[0]
          y = box[1]
          w = box[2]
          h = box[3]

          self.draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

      return img

  def predict_coord(self, img):
      #Performs a prediction and returns bounding box coordinates
      #image data
      Width = img.shape[1]
      Height = img.shape[0]
      scale = 0.00392

      #prepare image for prediction
      blob = cv2.dnn.blobFromImage(img, scale, self.INPUT_SIZE, (0,0,0), True, crop=False)
      self.NET.setInput(blob)

      #peform the prediction
	  
      # time_1 = time.time()
      outs = self.NET.forward(self.OUTPUT_LAYERS)
      # time_2 = time.time()
      # print("TIME :: ",time_2-time_1)
      #prediction results
      class_ids = []
      confidences = []
      boxes = []

      #interpret prediciton results
      for out in outs:
          for detection in out:
              scores = detection[5:]
              class_id = np.argmax(scores)
              confidence = scores[class_id]
              if confidence > 0.5:

                  center_x = int(detection[0] * Width)
                  center_y = int(detection[1] * Height)
                  w = int(detection[2] * Width)
                  h = int(detection[3] * Height)
                  x = center_x - w / 2
                  y = center_y - h / 2
                  class_ids.append(class_id)
                  confidences.append(float(confidence))
                  boxes.append([x, y, w, h])

      #Peform non-max supression
      indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESH, self.NMS_TRESH)

      results = []
      for i in indices:
          i = i[0]
          box = boxes[i]
          x = box[0]
          y = box[1]
          w = box[2]
          h = box[3]

          results.append([self.classes[class_ids[i]], confidences[i], round(x), round(y), round(x+w), round(y+h)])

      return results
