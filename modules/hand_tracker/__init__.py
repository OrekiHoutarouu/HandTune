from modules.utils import get_absolute_path
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import os

def initialize_landmarker():
    """
    Initializes and returns a HandLandmarker instance.

    Returns:
        HandLandmarker: Initialized hand landmarker object.
    """
    
    if hasattr(sys, '_MEIPASS'):
        model_path = os.path.join(sys._MEIPASS, "models", "hand_landmarker.task")
    else:
        model_path = get_absolute_path("models/hand_landmarker.task")

    base_options = mediapipe.tasks.BaseOptions
    hand_landmarker = mediapipe.tasks.vision.HandLandmarker
    hand_landmarker_options = mediapipe.tasks.vision.HandLandmarkerOptions
    vision_running_mode = mediapipe.tasks.vision.RunningMode

    options = hand_landmarker_options(
        base_options=base_options(model_asset_path=model_path),
        running_mode=vision_running_mode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.30,
        min_hand_presence_confidence=0.30,
        min_tracking_confidence=0.30
    )
    
    landmarker = hand_landmarker.create_from_options(options)
    return landmarker


def get_results(frame_rgb, landmarker):
    """
    Detects hand landmarks in an RGB image using the provided landmarker.

    Args:
        frame_rgb (numpy.ndarray): Input image in RGB format.
        landmarker (HandLandmarker): Initialized hand landmarker object.

    Returns:
        HandLandmarkerResult: Result containing detected hand landmarks.
    """

    mediapipe_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame_rgb)
    hand_landmarker_result = landmarker.detect(mediapipe_image)

    return hand_landmarker_result