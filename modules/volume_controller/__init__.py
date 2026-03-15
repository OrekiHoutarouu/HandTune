import os
from modules.utils import get_os

system = get_os()

def set_volume(volume, previous_volume):
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