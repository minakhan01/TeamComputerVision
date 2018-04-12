import cv2
import numpy as np
from webcam_video_stream import WebcamVideoStream
from dlib import get_frontal_face_detector, face_recognition_model_v1, shape_predictor
from imutils import face_utils
from keras.models import load_model

detector = get_frontal_face_detector()
FACENET_MODEL = "dlib_face_recognition_resnet_model_v1.dat"
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
face_rec = face_recognition_model_v1(FACENET_MODEL)
shape_predictor = shape_predictor(SHAPE_PREDICTOR)
face_recognizer = load_model('mlp_model_keras2.h5')

def recognize_face(face_descriptor):
	#print(face_descriptor)
	pred = face_recognizer.predict(face_descriptor)
	pred_probab = pred[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_face_names():
	face_ids = dict()
	import sqlite3
	conn = sqlite3.connect("face_db.db")
	sql_cmd = "SELECT * FROM faces"
	cursor = conn.execute(sql_cmd)
	for row in cursor:
		face_ids[row[0]] = row[1]
	return face_ids

def extract_face_info(img, img_rgb, face_names):
	faces = detector(img_rgb)
	x, y, w, h = 0, 0, 0, 0
	face_descriptor = None
	if len(faces) > 0:
		for face in faces:
			shape = shape_predictor(img, face)
			(x, y, w, h) = face_utils.rect_to_bb(face)
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
			face_descriptor = face_rec.compute_face_descriptor(img_rgb, shape)
			face_descriptor = np.array([face_descriptor,])	
			probability, face_id = recognize_face(face_descriptor)
			if probability > 0.8:
				cv2.putText(img, "FaceId #"+str(face_id), (x-20, y - 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
				cv2.putText(img, 'Name - '+face_names[face_id],  (x-20, y - 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
				cv2.putText(img, "%s %.2f%%" % ('Probability', probability*100), (x-20, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
			else:
				cv2.putText(img, 'No matching faces', (x-20, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
		
def recognize():
	face_names = get_face_names()
	img = cv2.imread('img.jpg', 1)
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	extract_face_info(img, img_rgb, face_names)
	cv2.imshow('Recognizing faces', img)
	cv2.imwrite('img_recognized.jpg', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

recognize()
