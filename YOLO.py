import numpy as np
import argparse
import imutils
import time
import os
import cv2

def function1():
	ca = argparse.ArgumentParser()
	ca.add_argument("-i", "--input", required=True)
	ca.add_argument("-o", "--output", required=True)
	ca.add_argument("-y", "--yolo", required=True)
	ca.add_argument("-c", "--confidence", type=float, default=0.5)
	ca.add_argument("-t", "--threshold", type=float, default=0.3)
	result = vars(ca.parse_args())
	return result

def get_data(args):
	coco_names_path = os.path.sep.join([args["yolo"], "coco.names"])
	coco_names = open(coco_names_path).read().strip().split("\n")
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(coco_names), 3),
							   dtype="uint8")
	yolo_weights = os.path.sep.join([args["yolo"], "yolov3.weights"])
	yolo_config = os.path.sep.join([args["yolo"], "yolov3.cfg"])
	net_total = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
	length = net_total.getLayerNames()
	length = [length[i[0] - 1] for i in net_total.getUnconnectedOutLayers()]
	return net_total,length,COLORS,coco_names

def main():
	result= function1()
	net_total,length, COLORS,coco_names = get_data(result)
	input = cv2.VideoCapture(result["input"])
	author = None
	(o_width, o_height) = (None, None)
	frame_count = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2()\
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(input.get(frame_count))
	print("total frames in video are {}".format(total))
	while True:
		(grabbed, video) = input.read()
		if not grabbed:
			break
		if o_width is None or o_height is None:
			(o_height, o_width) = video.shape[:2]
		image_blob = cv2.dnn.blobFromImage(video, 1 / 255.0, (416, 416))
		net_total.setInput(image_blob)
		start = time.time()
		l_Outputs = net_total.forward(length)
		end = time.time()
		bounding_boxes = []
		confidence_ratio = []
		class_labels = []
		for i in l_Outputs:
			for j in i:
				scores = j[5:]
				class_label = np.argmax(scores)
				confidence = scores[class_label]
				if confidence > result["confidence"]:
					image = j[0:4] * np.array([o_width, o_height, o_width, o_height])
					(X,Y, width, height) = image.astype("int")
					x = int(X - (width / 2))
					y = int(Y - (height / 2))
					bounding_boxes.append([x, y, int(width), int(height)])
					confidence_ratio.append(float(confidence))
					class_labels.append(class_label)
		data = cv2.dnn.NMSBoxes(bounding_boxes, confidence_ratio, result["confidence"],result["threshold"])
		if len(data) > 0:
			for i in data.flatten():
				(x, y) = (bounding_boxes[i][0], bounding_boxes[i][1])
				(w, h) = (bounding_boxes[i][2], bounding_boxes[i][3])
				color = [int(c) for c in COLORS[class_labels[i]]]
				cv2.rectangle(video, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(coco_names[class_labels[i]],
				 	confidence_ratio[i])
				cv2.putText(video, text, (x, y - 5),
					cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
		if author is None:
			output = cv2.VideoWriter_fourcc(*"MJPG")
			author = cv2.VideoWriter(result["output"], output, 35,
				(video.shape[1], video.shape[0]), True)
			if total > 0:
				elap = (end - start)
				print("single frame took {:.3f} seconds".format(elap))
				print("total time to finish: {:.2f}".format(
					elap * total))
		author.write(video)
	author.release()
	input.release()
main()
