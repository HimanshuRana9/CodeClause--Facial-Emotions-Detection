# realtime.py - live webcam demo using a saved Keras model (emotion_model.h5)
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import decode_emotion

MODEL_PATH = 'emotion_model.h5'

def preprocess_face(face_img):
    face = cv2.resize(face_img, (48,48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, -1)  # channel
    face = np.expand_dims(face, 0)   # batch
    return face

def main(model_path=MODEL_PATH, cam_index=0):
    if not os.path.exists(model_path):
        print(f'Model not found at {model_path}. Please run train.py or place a model file there.')
        return
    print('Loading model...')
    model = load_model(model_path)
    print('Model loaded. Starting webcam... (press q to quit)')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print('Cannot open webcam. Try a different --cam index.')
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            face_in = preprocess_face(roi)
            pred = model.predict(face_in, verbose=0)[0]
            label = decode_emotion(pred)
            conf = float(pred.max())
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow('Facial Emotions Detection - press q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
