from picamzero import Camera
from time import sleep
import requests
import io
from PIL import Image

cam = Camera()

# Capture the photo into an in-memory byte stream
photo_stream = io.BytesIO()
cam.capture(photo_stream, format='jpeg')
photo_stream.seek(0)  # Rewind the stream to the beginning

# Send the photo to the endpoint
url = 'http://mathias.local:8000/predict/'
files = {'file': ('new_image.jpg', photo_stream, 'image/jpeg')}
headers = {'accept': 'application/json'}

response = requests.post(url, headers=headers, files=files)

# Print the response from the server
print(response.json())