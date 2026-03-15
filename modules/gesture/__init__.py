import math
import numpy

def calculate_distance(frame, detection_result):
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
    if not distance:
        return
    
    volume = numpy.interp(distance,[30,200],[0,100])
    volume = int(volume/5)*5

    return volume