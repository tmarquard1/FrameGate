from picamzero import Camera
from time import sleep

cam = Camera()
# cam.start_preview()
cam.take_photo("/home/tmarquard/Documents/FrameGate/camera-experiment/new_image.jpg")
# Keep the preview window open for 5 seconds
sleep(5)