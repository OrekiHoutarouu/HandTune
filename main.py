from modules import gesture, hand_tracker, utils, volume_controller, webcam
import mediapipe
import cv2

capture = webcam.get_webcam()
landmarker = hand_tracker.initialize_landmarker()
previous_volume = 0

while True:
    frame = webcam.get_frame(capture)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hand_tracker.get_results(frame_rgb, landmarker)
    
    distance = gesture.calculate_distance(frame, results)
    volume = gesture.calculate_volume(distance)

    previous_volume = volume_controller.change_volume(volume, previous_volume)

    annotated_image = utils.draw_landmarks_on_image(frame_rgb, results, volume)

    cv2.imshow("Volume control", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) == ord("q"):
        break
