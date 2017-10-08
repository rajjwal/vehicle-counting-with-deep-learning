# USAGE
# python vehicle_tracker_by_frame.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import sys

from collections import defaultdict

car2points_map = defaultdict(list)
time2cars_map = defaultdict(list)
detection_ts = list()
start_time = time.time()
num_cars = 0

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_median_get_points(boxA, boxB):
    new_StartX = float(boxA[0] + boxB[0]) / float(2)
    new_StartY = float(boxA[1] + boxB[1]) / float(2)
    new_EndX = float(boxA[2] + boxB[2]) / float(2)
    new_EndY = float(boxA[3] + boxB[3]) / float(2)
    return (new_StartX, new_StartY, new_EndX, new_EndY)

def uniquify_boxes(new_boxes):
    i = 0
    unique_boxes = list()
    while i < len(new_boxes):
        box = new_boxes[i]
        max_overlap = 1.0
        min_overlap = 0.3275
        sim_box = None
        increment = 1
        for j in xrange(i+1, len(new_boxes)):
            overlap = bb_intersection_over_union(box, new_boxes[i])
            if overlap >= min_overlap:
                if any(unique_boxes):
                    unique_boxes[i] = get_median_get_points(unique_boxes[i], new_boxes[i])
                else:
                    unique_boxes.append(get_median_get_points(box, new_boxes[i]))
                increment += 1
        i += increment
    return unique_boxes   

def map_cars_by_bbox(new_boxes, dtime):
    global num_cars
    if len(new_boxes) > 1:
        new_boxes = uniquify_boxes(new_boxes)
    if not any(detection_ts):
        detection_ts.append(dtime)
        for i in xrange(len(new_boxes)):
            num_cars = i
            car_name = "car#{0}".format(num_cars)
            car2points_map[car_name].append(new_boxes[i])
            time2cars_map[dtime].append((new_boxes[i], car_name))
        return None
    prev_timestp = detection_ts[-1]
    detection_ts.append(dtime)
    for curr_box in new_boxes:
        max_overlap = 1.0
        min_overlap = 0.3275
        sim_box = None
        for prev_box in time2cars_map[prev_timestp]:
            overlap = bb_intersection_over_union(curr_box, prev_box[0])
            if overlap >= max_overlap:
                car2points_map[prev_box[1]].append(curr_box)
                time2cars_map[dtime].append((curr_box, prev_box[1]))
                break
            if overlap > min_overlap:
                sim_box = prev_box
                min_overlap = overlap
        if overlap == max_overlap:
            continue
        if min_overlap != 0.3275:
            car2points_map[sim_box[1]].append(curr_box)
            time2cars_map[dtime].append((curr_box, sim_box[1]))
            continue
        else: # check prev ko prev timestp
            num_cars = num_cars + 1
            car2points_map['car#{0}'.format(num_cars)].append(curr_box)
            time2cars_map[dtime].append((curr_box, 'car#{0}'.format(num_cars)))
    return None

def map_cars_by_centers(cnts):
    min_dist = 0
    max_dist = 5
    for center in cnts:
        if not mapped_points:
            mapped_points[center].append(center)

        for start_center in mapped_points:
            pass
    if len(prev_cnts) < len(curr_cnts):
        new_points = [point for point in curr_cnts if point not in mapped_points]
    return mapped_points, new_points

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.25,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

video_path = "data/IntersectionCarVideoIII.ASF"
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
rel_classes = ['car', 'truck', 'bus']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(video_path)
time.sleep(2.0)
# fps = FPS().start()
prev_pts = list()
curr_pts = list()
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    success, frame = vs.read()
    if not success:
        print "Video read Complete! :D or Failed! :("
        break
    frame = imutils.resize(frame, width=1000)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
    
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    detection_time = time.time()
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        
        # if curr_pts:
        #     prev_pts = curr_pts
        #     curr_pts = list()
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            if idx != 7:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if any(box):
                curr_pts.append((startX, startY, endX, endY))
                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    if any(curr_pts) and prev_pts != curr_pts:
        print curr_pts
        map_cars_by_bbox(curr_pts, detection_time)
        print curr_pts
        print "\nNum total cars detected:", len(car2points_map), "\nNum cars last seen on screen:", len(curr_pts)
        prev_pts = curr_pts[:]
    curr_pts = []

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

print "\nDumping all results of file to analyze later."
with open('results/car_results.json', 'w') as outfile:
    json.dump([car2points_map], outfile)
    print '\nWrote car2points_map.'

with open('results/time2cars_map.json', 'w') as outfile:
    json.dump([time2cars_map], outfile)
    print 'Wrote time2cars_map.\n'

# stop the timer and display FPS information
# fps.stop()
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
# fps.stop()
cv2.destroyAllWindows()
vs.stop()


