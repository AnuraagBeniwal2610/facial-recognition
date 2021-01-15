import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
#from google.colab.patches import cv2_imshow
import cv2
from datetime import datetime
import pickle
import joblib
import os
import time
from PIL import Image, ImageDraw
start_time = time.time()
import numpy as np



def predict(imageselect):

    train_dir = os.listdir('C:/Users/Anuraag/Desktop/OLD OFFICE/FACE_REC_SVC/train_dir/')

    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file(imageselect)
    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    print("Number of faces detected: ", no)
    # Predict all the faces in the test image using the trained classifier
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]

    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    thresh = loaded_model.predict_proba([test_image_enc])
    thresh_updated = thresh*100
    result = []


    for prob in thresh:
        if (prob > 0.1).any():
            result = np.where(prob > 0.1, True, False)
            """if result == True:
                
                res = [i for i, val in enumerate(result) if val]
                integers = res
                strings = [str(integer) for integer in integers]
                a_string = "".join(strings)
                an_integer = int(a_string)
                print(train_dir[an_integer])"""
            res = [i for i, val in enumerate(result) if val]
            integers = res
            strings = [str(integer) for integer in integers]
            a_string = "".join(strings)
            an_integer = int(a_string)
            return "EMPLOYEE"+" -> "+train_dir[an_integer]
            #print("EMPLOYEE"+" -> "+train_dir[an_integer])
        else:
            return "UNKNOWN USER"
            #print("UNKNOWN USER")

            
    