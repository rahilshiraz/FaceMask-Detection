from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

prototxtpath = os.path.sep.join(['face_detector','deploy.prototxt'])
weightspath = os.path.sep.join(['face_detector','res10_300x300_ssd_iter_140000.caffemodel'])
net = cv2.dnn.readNet(prototxtpath, weightspath)

model = load_model('maskdetect.model')

cap = cv2.VideoCapture(1)
while True:
    _,frame = cap.read()
    (h,w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame,1.0,(300, 300),(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]
            if mask>0.7 or withoutMask>0.7:
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.rectangle(frame,(startX,startY),(endX+5,startY+20),color,cv2.FILLED)
                cv2.putText(frame, label, (startX, startY +15),cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1)
                cv2.rectangle(frame, (startX, startY), (endX+5, endY), color, 2)
        
    cv2.imshow("Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
