import cv2

classifier = cv2.CascadeClassifier(
    "cascades/haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)

while(True):
    ret, frame = camera.read()

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = classifier.detectMultiScale(
        gray_image, minSize=(225, 225))

    for (x, y, l, a) in detected_faces:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()
