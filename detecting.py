import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
detect = cv2.face.LBPHFaceRecognizer_create()
detect.read("face_training.yml")

labels = {"avenger_name": 1}

def rescale_frame(frame, percent=105):
    scale_percent = 105
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


with open("face_labels.pickle", "rb") as a:
    og_labels = pickle.load(a)
    labels = {v: k for k, v in og_labels.items()}

vid = cv2.VideoCapture("avengersupdated.mp4")

#fourcc = cv2.VideoWriter_fourcc(*'MP4V') Bilgisayarım çok yavaş olduğu için yazdırma işlemini gerçekleştiremiyorum
#rec = cv2.VideoWriter('avengersdetected.mp4', fourcc, 20.0, (1249, 2221))

while True:
    ret, frame = vid.read()
    frame = rescale_frame(frame, percent=5)
    #cv2.imshow('frame', frame)
    frame2 = rescale_frame(frame, percent=5)
    #cv2.imshow("frame2", frame2)
    frame3 = rescale_frame(frame2, percent=5)
    #cv2.imshow("frame3", frame3)
    kernel = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 8, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 4.0

    denoise = cv2.fastNlMeansDenoisingColored(frame3, None, 7, 15, 7, 21)
    output_kernel = cv2.filter2D(denoise, -1, kernel)
    gray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = denoise[y:y+h, x:x+w]

        avenger, conf = detect.predict(roi_gray)
        if conf >= 85:
            name = labels[avenger]
            cv2.putText(denoise, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(denoise, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #rec.write(denoise)
    cv2.imshow('frame ', denoise)
    c = cv2.waitKey(10) & 0xff
    if c == 27:
        break

vid.release()
cv2.destroyAllWindows()
