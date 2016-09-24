# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import cv
import cv2
import random
import string
import datetime
import numpy as np
from PIL import Image


'''
    haarcascade_eye_tree_eyeglasses.xml   haarcascade_mcs_leftear.xml
    haarcascade_eye.xml                   haarcascade_mcs_lefteye.xml
    haarcascade_frontalface_alt2.xml      haarcascade_mcs_mouth.xml
    haarcascade_frontalface_alt_tree.xml  haarcascade_mcs_nose.xml
    haarcascade_frontalface_alt.xml       haarcascade_mcs_rightear.xml
    haarcascade_frontalface_default.xml   haarcascade_mcs_righteye.xml
    haarcascade_fullbody.xml              haarcascade_mcs_upperbody.xml
    haarcascade_lefteye_2splits.xml       haarcascade_profileface.xml
    haarcascade_lowerbody.xml             haarcascade_righteye_2splits.xml
    haarcascade_mcs_eyepair_big.xml       haarcascade_smile.xml
    haarcascade_mcs_eyepair_small.xml     haarcascade_upperbody.xml
'''

cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.createLBPHFaceRecognizer(1, 8, 8, 8, 123)


def get_images():
    path = "media"
    image_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
    images = []
    labels = []
    for image_path in image_paths:
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')
        subject_number = int(image_path.split("_")[-1].split('.')[0])
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(subject_number)
            cv2.imshow("", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    return images, labels


def train_face_recognition():
    images, labels = get_images()
    cv2.destroyAllWindows()
    recognizer.train(images, np.array(labels))


def capture_image():
    capture = cv.CaptureFromCAM(1)
    frame = cv.QueryFrame(capture)
    random_part = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(11))
    now = datetime.datetime.utcnow()
    image_time = now.strftime("%Y%m%d_%H%M%S")
    cv.SaveImage("media/capture_%s_%s.jpg" % (random_part, image_time), frame)


def find_faces():
    print("Starting face detection")
    path = "media"
    image_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
    for image_path in image_paths:
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
        for (x, y, w, h) in faces:
            number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])
            number_actual = int(image_path.split("_")[-1].split('.')[0])
            if number_actual == number_predicted:
                print "{} is Correctly Recognized with confidence {}".format(number_actual, conf)
            else:
                print "{} is Incorrect Recognized as {}".format(number_actual, number_predicted)
            cv2.imshow("Recognizing Face", image[y: y + h, x: x + w])
            cv2.waitKey(1000)

if __name__ == "__main__":
    capture_image()
    train_face_recognition()
    capture_image()
    find_faces()
