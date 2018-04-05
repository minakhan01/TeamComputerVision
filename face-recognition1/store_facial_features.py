import cv2
import numpy as np
import dlib
import pickle
import os, csv
from random import shuffle

MODEL = "dlib_face_recognition_resnet_model_v1.dat"
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
FACE_DIR = "faces/"
CSV_FILE = "dataset.csv"

face_rec = dlib.face_recognition_model_v1(MODEL)
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
detector = dlib.get_frontal_face_detector()

folder_names = os.listdir(FACE_DIR)
face_descriptors = []
row = ""
for folder_name in folder_names:
	full_folder_path  = FACE_DIR+folder_name+"/"
	images = os.listdir(full_folder_path)
	for image in images:
		full_image_path = full_folder_path+image
		print(full_image_path)
		img = cv2.imread(full_image_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		try:
			face = detector(img, 1)[0]
		except:
			print("Error")
			continue
		shape = shape_predictor(img, face)
		face_descriptor = face_rec.compute_face_descriptor(img, shape)
		face_descriptor = list(face_descriptor)
		face_descriptor.insert(0, int(folder_name))
		face_descriptors.append(face_descriptor)
		#print(face_descriptor, type(face_descriptor))
		
shuffle(face_descriptors)
shuffle(face_descriptors)
shuffle(face_descriptors)
if os.path.exists(CSV_FILE):
	os.remove(CSV_FILE)
for face_descriptor in face_descriptors:
	with open(CSV_FILE, "a", newline="") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(face_descriptor)

