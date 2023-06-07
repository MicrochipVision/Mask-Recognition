import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("trained_model.h5", compile=False) #Downalod the keras model from the link: https://tinyurl.com/yc3reb7b

# Load the labels
class_names = ["Without Mask", "With Mask"]

# Load the face cascade XML file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read the video stream
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess the ROI for the model
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = (face_roi / 255.0) - 0.5

        # Perform mask detection
        predictions = model.predict(face_roi)
        prediction_index = np.argmax(predictions[0])
        class_name = class_names[prediction_index]
        confidence = predictions[0][prediction_index]

        # Determine the label and color for drawing the bounding box
        if class_name == "Without Mask" and confidence < 0.05:
            label = "With Mask"
            color = (0, 255, 0)  # Green border
        else:
            label = f"{class_name}: {confidence*100:.2f}%"
            color = (0, 0, 255)  # Red border

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the frame with bounding boxes
    cv2.imshow("Mask Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()

# Destroy all windows
cv2.destroyAllWindows()
