import os 
import numpy
import cv2
import mediapipe

def get_absolute_path(path):
    root = os.getcwd()

    return os.path.join(root, path)


def draw_landmarks_on_image(rgb_image, detection_result):
    mp_connections = mediapipe.tasks.vision.HandLandmarksConnections
    mp_drawing = mediapipe.tasks.vision.drawing_utils

    annotated_image = numpy.copy(rgb_image)

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    landmark_style = mp_drawing.DrawingSpec(
        color=(255, 255, 255),
        thickness=2,
        circle_radius=4
    )

    connection_style = mp_drawing.DrawingSpec(
        color=(160, 160, 160),
        thickness=2
    )

    for i in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[i]
        handedness = handedness_list[i]

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_connections.HAND_CONNECTIONS,
            landmark_style,
            connection_style
        )

        h, w, _ = annotated_image.shape

        thumb = hand_landmarks[4]
        index = hand_landmarks[8]

        x1, y1 = int(thumb.x * w), int(thumb.y * h)
        x2, y2 = int(index.x * w), int(index.y * h)

        highlight_color = (255, 200, 0)

        cv2.circle(annotated_image, (x1, y1), 7, highlight_color, -1)
        cv2.circle(annotated_image, (x2, y2), 7, highlight_color, -1)
        cv2.line(annotated_image, (x1, y1), (x2, y2), highlight_color, 3)

        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]

        text_x = int(min(x_coords) * w)
        text_y = int(min(y_coords) * h) - 10

        label = handedness[0].category_name

        cv2.putText(
            annotated_image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            4,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated_image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return annotated_image