import cv2
import numpy as np
import os 
from random import randint
import beepy
import threading


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

facecasc = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + facecasc)
elfcasc = "haarcascade_lefteye_2splits.xml"
lefteye = cv2.CascadeClassifier(cv2.data.haarcascades + elfcasc)
ergcasc = "haarcascade_righteye_2splits.xml"
righteye = cv2.CascadeClassifier(cv2.data.haarcascades + ergcasc)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (0,20)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 2
lineType               = 2
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Bayhaqi', 'Paula', 'Ilza', 'Z', 'W'] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
count = 0
mleft = 0
mright = 0


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

@threaded
def beep():
    beepy.make_sound.beep(sound="ping")


def matablink():
    global mleft
    global mright 
    mleft = randint(0, 1)
    mright = randint(0, 1)
    
def matablinktrick():
    global mleft
    global mright 
    if(mleft==1):
        msgleft = "Buka mata kiri"
    else:
        msgleft = "Tutup mata kiri"
    if(mright==1):
        msgright = "buka mata kanan"
    else:
        msgright = "tutup mata kanan"
    return msgleft + ' dan ' + msgright 
        

while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        reyes = righteye.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=5,
            minSize=(5, 5),
            )
        leyes = lefteye.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=5,
            minSize=(5, 5),
            )
        
        msg = "Tidak terdeteksi wajah"        
        
        msg = matablinktrick()
        for (ex, ey, ew, eh) in leyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
        for (ex, ey, ew, eh) in reyes:
            # msg = matablinktrick()
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        pre = False

        if(mleft == 0 ):
            if(len(leyes)==0):
                pre = True
            else:
                pre = False
        else:
            if(len(leyes)>=0):
                pre = True
            elif(len(leyes)==0):
                pre = False
                
        if(pre):
            if(mright == 0 ):
                if(len(reyes)==0):
                    pre = True
                else:
                    pre = False
            else:
                if(len(reyes)>=0):
                    pre = True
                elif(len(reyes)==0):
                    pre = False
            
        if(pre):
            thrd = beep()
            thrd.join
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(
                        img, 
                        str(id), 
                        (x+5,y-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                    )
            cv2.putText(
                        img, 
                        str(confidence), 
                        (x+5,y+h-5), 
                        font, 
                        1, 
                        (255,255,0), 
                        1
                    )  
            pre = False
        
    
    cv2.putText(img,msg, 
                bottomLeftCornerOfText,
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)   
    cv2.imshow('camera',img) 
    count += 1
    # print(count)
    if(count==500):
        count = 0
        matablink()
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()