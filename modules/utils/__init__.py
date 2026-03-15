import os 
import numpy
import cv2
import mediapipe
import platform

def get_absolute_path(path):
    """Returns the absolute path for a given relative path.

    Args:
        path (str): Relative or non-absolute file path.

    Returns:
        str: Absolute file path.
    """
    
    root = os.getcwd()
    return os.path.join(root, path)


def get_os():
    """Returns the name of the user's operating system.

    Returns:
        str: Name of the operating system (e.g., 'Windows', 'Linux', 'Darwin').
    """

    return platform.system()


def draw_landmarks_on_image(rgb_image, detection_result, volume):
    """Draws hand landmarks, connections, handedness label, and a volume bar on an image.

    Args:
        rgb_image (numpy.ndarray): Input RGB image.
        detection_result: Detection result object with hand landmarks and handedness.
        volume (float): Calculated volume (0-100).

    Returns:
        numpy.ndarray: Annotated image with landmarks, labels, and volume bar.
    """

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

    draw_volume_bar(volume, annotated_image)

    return annotated_image


def draw_volume_bar(volume, annotated_image):
    """Draws a vertical volume bar and its label on the annotated image.

    Args:
        volume (float): Calculated volume (0-100).
        annotated_image (numpy.ndarray): Image to draw the volume bar on.
    """

    if volume is None:
        return

    bar = numpy.interp(volume, [0, 100], [400, 150])

    bar_color = (200, 200, 200)      
    fill_color = (255, 200, 0)   

    cv2.rectangle(annotated_image, (50,150), (85,400), bar_color, 2)
    cv2.rectangle(annotated_image, (50,int(bar)), (85,400), fill_color, -1)

    label = f"{int(volume)}%"

    cv2.putText(
        annotated_image,
        label,
        (35,430),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,0),
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        annotated_image,
        label,
        (35,430),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2,
        cv2.LINE_AA
    )
