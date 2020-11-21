import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
model = load_model(
    'model/model.h5', custom_objects={"KerasLayer": hub.KerasLayer})

labels_dict = {0: 'withmask', 1: 'Withoutmask'}

color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

size = 4
webcam = cv2.VideoCapture(0)  # Use camera 0

# load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 1)  # Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
        # Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (224, 224))
        normalized = resized/255.0
        reshaped = np.reshape(normalized, (1, 224, 224, 3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)
        print(result)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(im, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.rectangle(im, (x, y-40), (x+w, y), color_dict[label], -1)
        cv2.putText(im, labels_dict[label], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the image
    cv2.imshow('Mask detection', im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27:  # The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()
