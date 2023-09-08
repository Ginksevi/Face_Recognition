import cv2 as cv
import os
import imutils as im

modelo = 'fotos'
ruta_1 = 'Facial_1'
ruta_final = ruta_1 + '/' + modelo
if not os.path.exists(ruta_final):
    os.makedirs(ruta_final)

ruido = cv.CascadeClassifier('Cv\haarcascade_frontalface_default.xml')
camara = cv.VideoCapture(1)
id = 0
while True:
    respuesta, captura = camara.read()
    if respuesta == False:break
    captura = im.resize(captura, width=640)
    gris = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura = captura.copy()
    rostro = ruido.detectMultiScale(gris, 1.3, 3)
    
    for(x, y, e1, e2) in rostro:
        cv.rectangle(captura, (x,y), (x+e1,y+e2), (0,0,255), 2)
        rostrocapt = idcaptura [y:y + e2, x:x + e1]
        rostrocapt = cv.resize(rostrocapt, (150, 150), interpolation=cv.INTER_CUBIC)
        cv.imwrite(ruta_final + '/imagen_{}.jpg'.format(id), rostrocapt)
        id = id + 1
    cv.imshow("Resultado", captura)
    
    if id == 200:
        break
camara.release()
cv.destroyAllWindows()
