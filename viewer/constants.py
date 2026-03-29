"""
constants.py

Shared constants for the EndoSam project.
"""

# Video/clip processing constants
DEFAULT_CLIP_FPS = 10.0  # Frames per second of source clips
DEFAULT_FRAMES_PER_CLIP = 600  # Number of frames per clip segment (60 seconds at 10 FPS)

# Bounding box colors (BGR format)
BBOX_COLORS_BGR = [
    (53, 57, 229),   # Red
    (71, 160, 67),   # Green
    (229, 136, 30),  # Blue
    (0, 140, 251),   # Orange
    (170, 36, 142),  # Purple
    (193, 172, 0),   # Cyan
]

# Mask overlay settings
MASK_COLOR_BGR = (180, 50, 180)  # Purple
MASK_ALPHA = 0.35  # Transparency level

