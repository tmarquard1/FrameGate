from picamera2 import Picamera2
import requests
import io
from PIL import Image
import time

# Initialize the camera
picam2 = Picamera2()

# Configure the camera for video capture
config = picam2.create_video_configuration()
picam2.configure(config)

# Start the camera
picam2.start()

# Define the endpoint URL
url = 'http://heron.local:8000/predict/'
headers = {'accept': 'application/json'}

try:
    while True:
        # Capture a frame
        image = picam2.capture_array()
        img = Image.fromarray(image)
        
        # Rotate the image 180 degrees
        img = img.rotate(180)

        # Convert the image to RGB mode
        img = img.convert("RGB")
        
        # Save the image to an in-memory byte stream
        photo_stream = io.BytesIO()
        img.save(photo_stream, format='jpeg')
        photo_stream.seek(0)  # Rewind the stream to the beginning
        
        # Send the photo to the endpoint
        files = {'file': ('frame.jpg', photo_stream, 'image/jpeg')}
        response = requests.post(url, headers=headers, files=files)
        
        # Print the response from the server
        print(response.json())
        
        # Sleep for a short duration to control the frame rate
        time.sleep(.5)  # Adjust the sleep duration as needed

except KeyboardInterrupt:
    # Stop the camera when interrupted
    picam2.stop()
    print("Video capture stopped.")