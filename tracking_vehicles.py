import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import dlib
import tensorflow as tf

from utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util, recording_objects_utils
from object_detection.utils import visualization_utils as vis_util

def get_boxes(image_np, sess, detection_graph, max_boxes_to_draw=20, min_score_thresh=0.8):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    height, width = image_np.shape[:2]
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections], 
        feed_dict={image_tensor: image_np_expanded})
    boxes, scores = np.squeeze(boxes), np.squeeze(scores)
    centers = []
    detected_boxes = []
    found = False
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh:
            ymin, xmin, ymax, xmax = boxes[i].tolist()
            box = (int(width* xmin), int(height * ymin), int(width * xmax), int(height * ymax))
            center = (int((ymin + ymax) * height)/2.0, int(width * (xmin + xmax))/2.0)
            centers.append(center)
            detected_boxes.append(box)
    # if len(detected_boxes) > 0:
    #     found = True
    return detected_boxes, centers

def worker(input_q, output_q, path):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        # frame = cv2.resize(frame, (frame.cols/4, frame.rows/4))
        output_q.put(get_boxes(frame, sess, detection_graph))

    fps.stop()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', '--name-of-the-model', dest='MODEL_NAME', type=str, default='ssd_mobilenet_v1_coco_11_06_2017', help='Path to the frozen weights')
    parser.add_argument('-src', '--source', dest='video_source', type=str,
                        default="/Users/vijay/Downloads/cars_around_mountain.mp4", help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=960, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=540, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    

    CWD_PATH = os.getcwd()

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    # MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_NAME = args.MODEL_NAME
    PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q, PATH_TO_CKPT))

    # video_capture = WebcamVideoStream(src=args.video_source,
                                      # width=args.width,
                                      # height=args.height).start()
    capture = cv2.VideoCapture(args.video_source)

    if not capture.isOpened():
        print "File couldn't be opened."
        exit()

    while True: 
        retval, img = capture.read()
        if not retval:
            print "Cannot capture the frame."
            exit()

        #feeding the img to the get_boxes method
        input_q.put(img)
        #extracting the bounding boxes of the detected objects
        #for now the min_thresh score is 0.8 in get_boxes method above. One can change the threshold score as required
        bb_detected, centers = output_q.get()
        
        if(cv2.waitKey(10)==ord('p')):
            break
        img = cv2.resize(img, (args.width, args.height))
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        # if found: 
        #     break
    cv2.destroyWindow("Image")
    

    #Initialize tracker 
#     print bb_detected
#     # tracker = [dlib.correlation_tracker() for _ in xrange(len(bb_detected))]
#     # [tracker[i].start_track(img, dlib.rectangle(*rect)) for i, rect in enumerate(bb_detected)]
#     tracker = dlib.correlation_tracker()
#     tracker.start_track(img, dlib.rectangle(*bb_detected[0]))
#     dispLoc = True
#     while True:  # fps._numFrames < 120
#         success, img= capture.read()
#         if not success:
#             print "cannot capture the frame"
#             exit()
#         tracker.update(img)
#         rect = tracker.get_position()
#         pt1 = (int(rect.left()), int(rect.top()))
#         pt2 = (int(rect.right()), int(rect.bottom()))
#         cv2.rectangle(img, pt1, pt2, (255, 0, 0), 3)
#         print "Object tracked at [{}, {}] \r".format(pt1, pt2),
#         # for i in xrange(len(tracker)):
#         #     tracker[i].update(image)
#         #     rect = tracker[i].get_position()
#         #     pt1 = (int(rect.left()), int(rect.top()))
#         #     pt2 = (int(rect.right()), int(rect.bottom()))
#         #     cv2.rectangle(image, pt1, pt2, (225, 0, 0), 3)
#         if dispLoc:
#             loc = (int(rect.left()), int(rect.top()-20))
#             txt = "Object tracked at [{}, {}]".format(pt1, pt2)
#             cv2.putText(img, txt, loc , cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
#         cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
#         image = cv2.resize(img, (960, 540))
#         cv2.imshow('Image', image)

#  #       print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     fps.stop()
# #    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
# #    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    # video_capture.stop()
    # cv2.destroyAllWindows()
