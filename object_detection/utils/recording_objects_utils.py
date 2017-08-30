import numpy as np 
import tensorflow as tf 
import collections 
import os

def record_objects(image, 
					boxes, 
					gaze_pos, 
					timestamps, 
					classes, 
					scores, 
					category_index,
					path_to_records,
					max_boxes_to_draw=20,
					min_score_thresh=0.5,
					agnostic_mode=False
					):
	if not max_boxes_to_draw:
		max_boxes_to_draw = boxes.shape[0]
	with open(path_to_records, 'a') as f:
		for i in range(min(max_boxes_to_draw, boxes.shape[0])):
			if scores is None or scores[i] > min_score_thresh:
				box = tuple(boxes[i].tolist())
				gaze_x, gaze_y = gaze_pos[0], gaze_pos[1]
				ymin, xmin, ymax, xmax = box 
				if gaze_x > xmin and gaze_x < xmax and gaze_y > ymin and gaze_y < ymax:
					if not agnostic_mode:
						if classes[i] in category_index.keys():
							class_name = category_index[classes[i]]['name']
							print class_name
							f.write(str(class_name) + " " + str(timestamps) + "\n")
	f.close()

def record_vehicles(image, 
					boxes,  
					classes, 
					scores, 
					category_index,
					path_to_records,
					max_boxes_to_draw=20,
					min_score_thresh=0.8,
					agnostic_mode=False
					):
	if not max_boxes_to_draw:
		max_boxes_to_draw = boxes.shape[0]
	with open(path_to_records, 'a') as f:
		for i in range(min(max_boxes_to_draw, boxes.shape[0])):
			if scores is None or scores[i] > min_score_thresh:
				if classes[i] in category_index.keys():
					class_name = category_index[classes[i]]['name']
					print class_name
					f.write(str(class_name) + "\n")
						
	f.close()


# def reading_recorded_objects(path_to_records):
# 	objects = {}
# 	prev_object = ""
# 	prev_time = ""
# 	curr_object = ""
# 	curr_time = ""
# 	duration = 0.0
# 	with open(path_to_records, 'r') as f:
# 		count = 0 
# 		for line in f:
# 			items = line.strip().split(' ')
# 			prev_object = curr_object
# 			prev_time = curr_time
# 			curr_object = items[0]
# 			curr_time = items[1]
# 			if count == 0:
# 				count += 1
# 				continue
# 			duration += (float(curr_time) - float(prev_time))
# 			else:
# 				if prev_object != curr_object:
# 					objects[prev_object] = duration
# 					duration = 0.0
