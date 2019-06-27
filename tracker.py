import cv2
import numpy
import dlib
import argparse
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class TrackableObject:
    def __init__(self, objectID, centroid):
        
        self.objectID = objectID
        self.centroids = [centroid]

        self.counted = False



        
class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared

        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])


        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects
 


    
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", 
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
# model -->
net = cv2.dnn.readNetFromCaffe("res/mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                              "/res/mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

W, H = None, None


ct = CentroidTracker(maxDisappeared = 40, maxDistance = 40)
trackers = []
trackableObjects = {}

skipframes = 10

cap = cv2.VideoCapture("res/example_02.mp4")

totalFrames = 0
totalDown = 0
totalUp = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    # converting to dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (H, W) = frame.shape[:2]
    
    # initially nothing
    status = "Waiting"
    rects = []
    
    if totalFrames % skipframes == 0:
        status = "Detecting"
        trackers = []
 
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]
 
            if confidence > 0.5:
            
                idx = int(detections[0, 0, i, 1])
 
                if CLASSES[idx] != "person":
                    continue
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
            
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
 
                trackers.append(tracker)
    else:
        for tracker in trackers:
            status = "Tracking"
 
            tracker.update(rgb)
            pos = tracker.get_position()
    
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
 
            rects.append((startX, startY, endX, endY))
        
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
    
    objects = ct.update(rects)
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)

        else:
            
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:
            
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

        trackableObjects[objectID] = to

#         text = "ID {}".format(objectID)
#         cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
    info = [("Up", totalUp),("Down", totalDown),]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


    totalFrames += 1
    
    

cap.release()
cv2.destroyAllWindows()
