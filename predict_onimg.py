from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

def detectmask(img):
    (h,w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img,1.0,(300, 300),(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence>0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")

            face = img[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.rectangle(img,(startX,startY),(endX,startY+70),color,cv2.FILLED)
            cv2.putText(img, label, (startX, startY +65),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 4)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 4)

    img = cv2.resize(img,(480,480))
    return img

img1 = cv2.imread("test3.jpg")
img2 = cv2.imread('test2.jpg')

model = load_model('maskdetect.model')
prototxtpath = r"face_detector\deploy.prototxt"
weightspath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtpath, weightspath)

img1 = detectmask(img1)
img2 = detectmask(img2)

out = np.hstack((img1,img2))
cv2.imshow('Mask Detection',out)
cv2.waitKey(0)