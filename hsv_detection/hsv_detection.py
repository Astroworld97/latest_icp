import cv2
import numpy as np

# Define a mouse callback function to get the HSV values of the clicked pixel
def get_hsv_values(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        print("HSV values at ({}, {}): {}".format(x, y, hsv[y, x]))

# Initialize camera
cap = cv2.VideoCapture(0)

# Define a window name and set the mouse callback function
window_name = "Click to get HSV values"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, get_hsv_values)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow(window_name, frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
