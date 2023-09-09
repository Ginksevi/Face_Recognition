import cv2 as cv
import os

datapath = 'Facial_1/Data'
datalist = os.listdir(datapath)

training_1 = cv.face.EigenFaceRecognizer_create()
training_1.read('Training EigenRecognizer.xml')
ruidos = cv.CascadeClassifier('Cv\haarcascade_frontalface_default.xml')
camara = cv.VideoCapture(0)
while True:
    _, capture = camara.read()
    gris = cv.cvtColor(capture, cv.COLOR_BGR2GRAY)
    idcapture = gris.copy()
    face = ruidos.detectMultiScale(gris, 1.5, 5)
    for(x, y, e1, e2) in face:
        rostrocapt = idcapture [y:y + e2, x:x + e1]
        rostrocapt = cv.resize(rostrocapt, (150, 150), interpolation=cv.INTER_CUBIC)
        resultado = training_1.predict(rostrocapt)
        cv.putText(capture, '{}'.format(resultado), (x, y-5), 1, 1.3, (0,0,255), 2, cv.LINE_AA)
        if resultado [1] < 9000:
            cv.putText(capture, 'No encontrado', (x, y-10), 2, 1.1, (0,0,255), 2, cv.LINE_AA)
            cv.rectangle(capture, (x, y), (x+e1, y+e2), (255,0,0), 2)
        else:
            cv.putText(capture, '{}'.format(datalist[resultado[0]]), (x, y-10), 2, 1.1, (0,0,255), 2, cv.LINE_AA)
            cv.rectangle(capture, (x, y), (x+e1, y+e2), (255,0,0), 2)

    cv.imshow('Resultados', capture)
    if cv.waitKey(1) == ord ('q'):
        break
camara.release()
cv.destroyAllWindows()