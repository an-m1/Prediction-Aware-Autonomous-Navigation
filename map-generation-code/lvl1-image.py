import cv2
import numpy as np

width, height = 2000, 2000

# Create a solid 3-channel RGB white canvas (255, 255, 255) with no alpha channel
track = np.full((height, width, 3), 255, dtype=np.uint8)

# Draw a solid black line (0, 0, 0) in the center
line_width = 10
center_x = width // 2
cv2.rectangle(
    track,
    (center_x - line_width // 2, 0),
    (center_x + line_width // 2, height),
    (0, 0, 0),
    -1,
)

cv2.imwrite("textures/straight_line_track.png", track)