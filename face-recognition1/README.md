## Requirements
1. Python 3.x
2. Tensorflow
3. OpenCV
4. Dlib
5. numpy

## How to use the files (stepwise)

### Save few faces first

1. Run the save_face.py file

	python save_face.py

2. It will ask for face_id. Since I have already saved 3 different faces of my friends and myself, your face_id should start from 3.
3. It will also ask for a starting image number. Enter it as 1. If you have already done this and you still want to add more images then enter this as 301. Then for even more 601 and so on.
4. Now a window showing your webcam feed should appear. Make sure there is only one face in the frame or else the face capturing will stop. Also make sure that you give facial expressions during the capturing. After 300 images of your face are taken the window automatically stops.
5. You can see the faces saved in the faces/ directory.
6. In the faces directory you will see some subfolders named as '0', '1', '2' etc. These numbers represent the face_id. Inside each folder you will see 300 images of the person taken from the webcam.
7. You can add your own images that are taken from your phone or any other device inside these folders depending on your face_id.

### Storing 128 facial measurements or aka embeddings in a csv file

1. Run the store_facial_features.py file
	
	python store_facial_features.py

2. What this file does is it iteratively searches for every face in the faces folder, then computes the embeddings of each of them.
3. These embeddings are stored in a csv file called dataset.csv.
4. The format of the each row of csv file is <b>face_id</b> <b>facial measurement 1</b> <b>facial measurement 2</b>....<b>facial measurement 128</b>.

### Getting training and testing data from the csv file

1. Run the csv_to_pickle.py file
	
	python csv_to_pickle.py

2. The top 9/10th of the data in the csv file is used as training data and the rest 1/10th is used as testing data.
3. 4 new files will be created train_features, train_labels, test_features and test_labels.

### Train the model

1. Run the train_model.py file

	python train_model.py

2. The emdbedding are used to train a multilayer perceptron.
3. I used my rule of thumb to create this network. I have no idea why it works, but I know it works.
4. With the 3 faces of my friends and myself, I got 100% accuracy using this network.
5. Add many more faces to see if the network really works.
6. If even after adding many new faces the accuracy does decrease by a huge amount then the network can be used in mobile devices too since the network is very simple.
7. The checkpoint files are stored in the tmp/mlp_model/ folder.


## Future Work

1. The network created is currently not compatible with NCS. That can be done easily using the NCS SDK.
2. Real time recognition is still not implemented. Will be done later.