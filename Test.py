import cv2 
import numpy as np 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the ResNet50 model
emotion_model = ResNet50(weights=None, include_top=False)
emotion_model.load_weights("model/emotion_model_resnet.weights.h5")  # Load the weights

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascades_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        
        # Resize the input image to match the expected input size of ResNet50 (224, 224)
        roi_gray_frame_resized = cv2.resize(roi_gray_frame, (224, 224))
        roi_gray_frame_resized = roi_gray_frame_resized.astype('float') / 255.0
        roi_gray_frame_resized = np.expand_dims(roi_gray_frame_resized, axis=0)

        emotion_prediction = emotion_model.predict(roi_gray_frame_resized)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
