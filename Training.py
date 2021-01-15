import dlib
dlib.DLIB_USE_CUDA

import numpy as np
import joblib
import pickle
import face_recognition
from sklearn import svm
import os

import face_recognition
from sklearn import svm
import os
from tqdm import tqdm
import time

def train():

    encodings = []
    names = []

    # Training directory
    train_dir = os.listdir('C:/Users/Anuraag/Desktop/OLD OFFICE/FACE_REC_SVC/train_dir/')


    start_time = time.time()
    # Loop through each person in the training directory
    for person in tqdm(train_dir):
        pix = os.listdir("C:/Users/Anuraag/Desktop/OLD OFFICE/FACE_REC_SVC/train_dir/" + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file("C:/Users/Anuraag/Desktop/OLD OFFICE/FACE_REC_SVC/train_dir/" + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            #If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img + " was skipped and can't be used for training")

    joblib.dump(encodings,"face_encoding")
    joblib.dump(names,"name_encoding")

    known_face_encodings = joblib.load("face_encoding")
    known_name_encoding = joblib.load("name_encoding")
    clf = svm.SVC(gamma='scale',probability=True)
    clf.fit(known_face_encodings,known_name_encoding)

    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))