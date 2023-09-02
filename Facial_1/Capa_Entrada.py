import cv2 as cv
import numpy as np
ruido = cv.CascadeClassifier('D:\Clases\Proyectos_Python\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml')
camara = cv.VideoCapture(1)
while True:
    _, captura = camara.read()
    gris = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    rostro = ruido.detectMultiScale(gris, 1.3, 3)
    for(x, y, e1, e2) in rostro:
        cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0), 2)

    cv.imshow("Resultado", captura)
    
    if cv.waitKey(1) == ord ('q'):
        break
camara.release()
cv.destroyAllWindows()
