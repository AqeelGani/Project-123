import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import PIL.ImageOps
from PIL import Image
import os, time

x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
print(pd.series(y).value_counts())
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
n = len(classes)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state = 0, train_size = 7500, test_size = 2500)
xtrain = xtrain/255.0
xtest = xtest/255.0

model = LogisticRegression(solver = 'saga', multi_class = "multinomial").fit(xtrain, ytrain)

ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(accuracy)

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    try:
        ret,frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        top_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right = (int(width/2 + 56), int(height/2 + 56))

        cv2.rectangle(gray, top_left, bottom_right, (0,255,0), 2)

        roi = gray[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        im_pil = Image.fromarray(roi)
        img_bw = im_pil.convert('L')
        img_resized = img_bw.resize((28,28), Image.ANTIALIAS)

        img_inverted = PIL.ImageOps.invert(img_resized)

        min_pixel = np.percentile(img_inverted, 20)
        img_scaled  = np.clip(img_resized - min_pixel, 0, 255)
        max_pixel = np.max(img_inverted)
        img_scaled = np.asarray(img_scaled)/max_pixel
        test_sample = np.array(img_scaled).reshape(1, 784)
        test_pred = model.predict(test_sample)
        print("The Predicted Class Is :", test_pred)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()