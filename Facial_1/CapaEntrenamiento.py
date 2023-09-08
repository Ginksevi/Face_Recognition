import cv2 as cv
import numpy as np
import os
from time import time

datapath = 'Facial_1/Data'
datalist = os.listdir(datapath)
ids = []
dataface = []
id = 0
startime =time()

for data in datalist:
    finalpath = datapath + '/' + data
    print('Starting reading...')
    for file in os.listdir(finalpath):
        print('Image: ', data + '/' +file)
        ids.append(id)
        dataface.append(cv.imread(finalpath + '/' + file, 0))
    
    id = id+1
    totaltime = time()
    readtime = totaltime - startime
    print('Reading total time: ', readtime)

training_1 = cv.face.EigenFaceRecognizer_create()
print('Training start wait...')
training_1.train(dataface, np.array(ids))
finaltraining = time()
TimeTotalTraining = finaltraining - readtime
print('Time total training...', TimeTotalTraining)
training_1.write('Training EigenRecognizer.xml')
print('Complete training')