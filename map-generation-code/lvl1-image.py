import cv2
import numpy as np

width, height = 2000, 2000

# create a solid 3-channel  white canvas (255, 255, 255) with  no alpha  channel
track = np.full((height, width, 3), 255, dtype=np.uint8) 

line_width = 10
center_x = width // 2

# set  start (bottom) and end (top) positions with a  margin from the  edges
margin = 25        # 1.5 meters from the edge on a 20m x 20m plane
pad_width = 100    # 1.0 meter wide rectangle
pad_height = 60    # 0.6 meter tall  rectangle

start_y = height - margin
end_y  = margin

# 1. draw the solid black line  connecting start to end
cv2.rectangle(
    track,
    (center_x - line_width   / / 2, end_y),
    (center_x + line_width // 2, start_y), 
    (0, 0, 0),
    -1,
)

# 2. draw  the  green  start rectangle (opencv  uses  bgr: blue=0, green=200, red=0)
cv2.rectangle(
    track,
    (center_x - pad_width // 2, start_y - pad_height // 2),
    (center_x + pad_width // 2, start_y + pad_height // 2),
    (0, 200, 0),
    -1,
)

# 3. draw  the red goal rectangle (opencv uses bgr: blue=0, green=0, red=220)
cv2.rectangle(
    track,
    (center_x - pad_width // 2, end_y - pad_height // 2),
    (center_x + pad_width // 2, end_y + pad_height // 2),
    (0, 0, 220),
    -1,
)

cv2.imwrite("textures/straight_line_track.png", track)
