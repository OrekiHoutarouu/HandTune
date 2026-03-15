import cv2

def get_webcam():
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        exit()

    return capture

def get_frame(capture):
    success, frame = capture.read()

    if success:
        return frame