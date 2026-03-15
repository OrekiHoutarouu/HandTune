import os
from modules.utils import get_os

system = get_os()

def set_volume(volume, previous_volume):
    """Sets the system volume to the specified level if it differs significantly from the previous value.

    Args:
        volume (float): The target volume level as a percentage (0-100).
        previous_volume (float): The previous volume level as a percentage (0-100).

    Returns:
        bool: True if the volume was changed, False otherwise.
    Notes:
        - On Linux, uses 'pactl' to set the default sink volume.
        - On macOS (Darwin), uses 'osascript' to set the output volume.
        - On Windows, uses pycaw to set the master volume.
        - If 'volume' is None or the change is less than 5%, the volume is not updated.
    """

    if volume is None:
        return False

    if abs(volume - previous_volume) < 5:
        return False

    if system == "Linux":
        os.system(f"pactl set-sink-volume @DEFAULT_SINK@ {volume}%")
    
    elif system == "Darwin":
        os.system(f"osascript -e 'set volume output volume {volume}'")

    elif system == "Windows":
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_,
            CLSCTX_ALL,
            None
        )

        volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
        volume_interface.SetMasterVolumeLevelScalar(volume/100, None)

    return True