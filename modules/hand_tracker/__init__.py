from modules.utils import get_absolute_path
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def initialize_landmarker():
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
    mediapipe_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame_rgb)
    hand_landmarker_result = landmarker.detect(mediapipe_image)

    return hand_landmarker_result