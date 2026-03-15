import math
import numpy

def calculate_distance(frame, detection_result):
    """
    Calculates the distance between the thumb and index finger landmarks.

    Args:
        frame (numpy.ndarray): The input image frame.
        detection_result: An object containing hand landmarks, typically of type
            mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList.

    Returns:
        float or None: The Euclidean distance between the thumb tip (landmark 4)
            and index finger tip (landmark 8) in pixel coordinates, or None if
            no hand landmarks are detected.
    """
    
    if not detection_result.hand_landmarks:
        return
    
    landmarks = detection_result.hand_landmarks[0]

    h, w, _ = frame.shape

    thumb = landmarks[4]
    index = landmarks[8]

    x1, y1 = int(thumb.x * w), int(thumb.y * h)
    x2, y2 = int(index.x * w), int(index.y * h)

    distance = math.hypot(x2-x1, y2-y1)

    return distance


def calculate_volume(distance):
    """
    Maps the distance between thumb and index finger to a volume value.

    Args:
        distance (float): The distance between the thumb and index finger in pixels.

    Returns:
        int or None: The calculated volume (rounded to nearest multiple of 5) in the range [0, 100],
            or None if distance is not provided.
    """

    if not distance:
        return
    
    volume = numpy.interp(distance, [30, 200], [0, 100])
    volume = int(volume/5)*5

    return volume