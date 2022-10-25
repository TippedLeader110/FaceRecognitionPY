import cv2
import os
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
facecasc = "haarcascade_frontalface_default.xml"
smilecasc = "haarcascade_smile.xml"
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + facecasc)
smile_detect = cv2.CascadeClassifier(cv2.data.haarcascades + facecasc)

jumlahfoto             = 50
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (0,20)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 4
lineType               = 2
# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
take = False
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    if(count==0):
        msg = "Lihat Kedepan dan tekan SPACE"
    elif(count==jumlahfoto):
        msg = "Lihat sedikit Kekiri dan tekan SPACE"
        take=False
    elif(count==jumlahfoto*2):
        msg = "Lihat sedikit Kekanan dan tekan SPACE"
        take=False
    else:
        msg = "Mengambil Gambar"
    print(count)
    if(len(faces)==0):
        msg = "Wajah tidak fokus"
    for (x,y,w,h) in faces:
        
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)  
        
        # Save the captured image into the datasets folder
        print(take)
        if(take):
            count += 1
            cv2.imwrite("dataset/face/User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
    cv2.putText(img,msg, 
                bottomLeftCornerOfText,
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)   
    cv2.imshow('image', img)
        
        
    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    elif k == 32:
        take = True
        count += 1
    elif count >= jumlahfoto*3:
         break
     
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()