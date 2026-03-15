import os

def change_volume(volume, previous_volume):
    if volume is None:
        return previous_volume

    if abs(volume - previous_volume) >= 5:
        os.system(f"pactl set-sink-volume @DEFAULT_SINK@ {volume}%")
        previous_volume = volume

    return previous_volume
