import numpy as np
import time
import cv2
import os
from numpy.lib.type_check import imag
from PIL import Image

import pycozmo

# Last image, received from the robot.
last_im = None

class MLObjectDetector():
    def __init__(self, yolo, confidence=0.5, threshold=0.3):
        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([yolo, "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
                                dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([yolo, "yolov3.weights"])
        configPath = os.path.sep.join([yolo, "yolov3.cfg"])
        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.confidence = 0.5
        self.threshold = 0.3
        self.detections = 0


    def detect_object(self, pil_image):
        # Convert PIL Image to OpenCV format
        image = np.array(pil_image.convert('RGB')) 
        image = image[:, :, ::-1].copy() # Convert RGB to BGR 

        # load our input image and grab its spatial dimensions
        (H, W) = image.shape[:2]
        # determine only the *output* layer names that we need from YOLO
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()
        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence,
                                self.threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                if self.LABELS[classIDs[i]] == 'book' or self.LABELS[classIDs[i]] == 'cup':
                    color = [int(c) for c in self.COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
                    cv2.imwrite("object_detect_{}.jpg".format(self.detections), image)
                    self.detections += 1
                    inch_distance = (2 * 3.14 * 180) / (w + h * 360) * 1000 + 3
                    return inch_distance
        return -1


def on_camera_image(cli, new_im):
    """ Handle new images, coming from the robot. """
    global last_im
    last_im = new_im


if __name__ == "__main__":

    with pycozmo.connect(enable_procedural_face=True) as cli:
        ml_object_detector = MLObjectDetector("yolo-coco", confidence=0.3)
        # Raise head.
        cli.set_head_angle(0.0)
        time.sleep(1)

        # Register to receive new camera images.
        cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

        # Enable camera.
        cli.enable_camera(color=True)
        time.sleep(1)

        # Run with 14 FPS. This is the frame rate of the robot camera.
        timer = pycozmo.util.FPSTimer(14)
        counter = 0
        stop = False

        # last_im.save("camera_{}.jpg".format(), "JPEG")
        no_detection_counter = 0
        previous_step = False
        current_step = False
        while not stop:
            speed = 20
            previous_step = current_step
            cli.drive_wheels(lwheel_speed=speed,
                             rwheel_speed=speed, duration=1.0)
            print(previous_step, current_step)
            if counter%3 == 0:
                distance = ml_object_detector.detect_object(last_im)
                if distance != -1:
                    current_step = True
                else:
                    current_step = False

                if previous_step == True and current_step == False or distance < 2:
                    stop = True
            counter += 1
