from modules import gesture, hand_tracker, utils, volume_controller, webcam
import mediapipe
import cv2

capture = webcam.get_webcam()
landmarker = hand_tracker.initialize_landmarker()

while True:
    frame = webcam.get_frame(capture)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hand_tracker.get_results(frame_rgb, landmarker)

    annotated_image = utils.draw_landmarks_on_image(frame_rgb, results)

    cv2.imshow("Volume control", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) == ord("q"):
        break
