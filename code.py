#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 22:18:20 2023

@author: baranidharan
"""

import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load image and convert to grayscale
img = cv2.imread('/Users/baranidharan/Work Folder/Course Works/CVIP/Final Project CVIP trail/Code/image.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Loop over each detected face
for face in faces:
    # Get facial landmarks for the face
    landmarks = predictor(gray, face)
    
    # Create a mask for the face
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [np.array([(landmarks.part(n).x, landmarks.part(n).y) 
                                  for n in range(0, 68)])], (255, 255, 255))
    
    # Extract color pixel values for the face
    face_pixels = cv2.bitwise_and(img, mask)
    face_pixels = face_pixels[np.where((face_pixels!=[0,0,0]).all(axis=2))]

    # Print the extracted pixel values
    print(face_pixels)

cv2.imshow("Face",face_pixels)
cv2.waitKey(20)
cv2.destroyAllWindows