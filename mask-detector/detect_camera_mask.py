import cv2
import numpy as np
import tensorflow.keras.applications.mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

classifier = cv2.CascadeClassifier(
    "cascades/haarcascade_frontalface_default.xml")
mask_model = load_model("mask_detector")
camera = cv2.VideoCapture(0)

while(True):
    ret, frame = camera.read()

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = classifier.detectMultiScale(
        gray_image, minSize=(225, 225))

    for (x, y, l, a) in detected_faces:
        face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        faces = np.array([face], dtype="float32")
        pred = mask_model.predict(faces, batch_size=32)
        (mask, withoutMask) = pred[0]
        using_mask = mask > withoutMask
        color = (0, 255, 0) if using_mask else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + l, y + a), color, 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()
