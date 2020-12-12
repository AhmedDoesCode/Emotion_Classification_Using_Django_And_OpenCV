import cv2
from keras.preprocessing.image import img_to_array
import imutils
from keras.models import load_model
import numpy as np

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        # self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        self.video = cv2.VideoCapture(0)
        self.face_detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.emotion_classifier = load_model("Emo_stat.hdf5", compile=False)
        self.EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=7,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        frameClone = image.copy()
        f = True
        if len(faces)>0:
            faces = sorted(faces, reverse = True, key = lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = self.emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = self.EMOTIONS[preds.argmax()]
        else:
            print("X")
            f = False

        if f:
            for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):
                cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', frameClone)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
