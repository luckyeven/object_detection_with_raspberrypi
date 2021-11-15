"""
Raspberry Pi Camera Image Capture

Author: Shifeng Song
Date: Nev 13, 2021

"""

import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray

# Pi Settings
res_width = 96           # width of camera
res_height = 96          # height of camera
rotation = 