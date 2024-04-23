import cv2
import numpy as np
import tensorflow as tf
model = tf.saved_model.load("https://tfhub.dev/google/hands/2")
def detect_hands(frame):
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(input_image[np.newaxis, ...], dtype=tf.float32)
    output_dict = model.signatures["serving_default"](input_tensor)

    # Extract hand keypoints
    keypoints = output_dict['output_0']

    return keypoints
cap = cv2.VideoCapture(0)  # Open default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = detect_hands(frame)

    # Draw keypoints on the frame
    for kp in keypoints[0]:
        x, y = int(kp[1] * frame.shape[1]), int(kp[0] * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('Hand Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()