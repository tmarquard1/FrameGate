from picamera2 import Picamera2
import requests
import io
from PIL import Image

# Initialize the camera
picam2 = Picamera2()

# Configure the camera
config = picam2.create_still_configuration()
picam2.configure(config)

# Start the camera
picam2.start()

# Capture the photo into an in-memory byte stream
photo_stream = io.BytesIO()
image = picam2.capture_array()
img = Image.fromarray(image)
img.save(photo_stream, format='jpeg')
photo_stream.seek(0)  # Rewind the stream to the beginning

# Stop the camera
picam2.stop()

# Send the photo to the endpoint
url = 'http://mathias.local:8000/predict/'
files = {'file': ('new_image.jpg', photo_stream, 'image/jpeg')}
headers = {'accept': 'application/json'}

response = requests.post(url, headers=headers, files=files)

# Print the response from the server
print(response.json())