from cleanfunction import myfunction
import dlib
import cv2
import numpy as np

p = myfunction(

    detection = dlib.get_frontal_face_detector(),
    prediction = dlib.shape_predictor("file/face_landmarks_25-12-62.dat"),
    read = cv2.VideoCapture("video/1.avi"),     #cv2.VideoCapture("video/1.avi") && cv2.imread("pic/1.jpg")
    count = 1,                                  #showlandmark count = 0 && detectface count = 1
    keyd = ('d'),
    keyl = ('l'),
    keyq = ('q'),
    write = "video/saveimage/ve",                 #picture "pic/saveimage/img" && "video/saveimage/ve"
    newmodel = "face_landmarks_25-12-62.dat",
    oldmodel = "face_landmarks_19-12-62.dat",
    test = f"face_landmarks_test.xml",

    )

