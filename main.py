from modules import gesture, hand_tracker, utils, volume_controller, webcam
import cv2

capture = webcam.get_webcam()

while True:
    frame = webcam.get_frame(capture)
    cv2.imshow("Webcam", frame)
    
    if cv2.waitKey(1) == ord("q"):
        exit()