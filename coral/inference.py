from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import io
from PIL import Image, ImageDraw, ImageFont
import uvicorn
import os
from datetime import datetime
import tensorflow as tf
import pathlib
import cv2

app = FastAPI()

# Load the model from the saved path
model_path = '/home/movinet_model'
model = tf.saved_model.load(model_path)

labels_path = pathlib.Path('kinetics_600_labels.txt')
lines = labels_path.read_text().splitlines()
KINETICS_600_LABELS = np.array([line.strip() for line in lines])
images = []
init_state = None
n = 0

# Function to get top k labels and probabilities
def get_top_k(probs, k=5, label_map=KINETICS_600_LABELS):
    top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
    top_labels = tf.gather(label_map, top_predictions, axis=-1)
    top_labels = [label.decode('utf8') for label in top_labels.numpy()]
    top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
    return tuple(zip(top_labels, top_probs))

def load_images(image_files, image_size=(224, 224)):
    frames = []
    for image_file in image_files:
        img = Image.open(io.BytesIO(image_file)).resize(image_size)
        img = np.array(img) / 255.0
        frames.append(img)
    video = tf.convert_to_tensor(frames, dtype=tf.float32)
    return video

def overlay_text_on_image(image, text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_size = draw.textsize(text, font=font)
    position = (10, 10)
    draw.rectangle([position, (position[0] + text_size[0], position[1] + text_size[1])], fill="black")
    draw.text(position, text, fill="white", font=font)
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global images, init_state, n
    
    contents = await file.read()
    
    # Save the file with a timestamped name in /Downloads
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f"/Downloads/{timestamp}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Load images
    images.append(contents)  # Assuming a single image for now
    video = load_images(images)
    
    # Initialize states
    if len(images) == 1:
        print("Initializing states...")
        init_state = model.init_states(video[tf.newaxis, ...].shape)
    else:
        print(f"Updating states...{len(images)}")

    # Perform inference
    inputs = init_state.copy()
    inputs['image'] = video[tf.newaxis, n:n+1, ...]
    logits, state = model(inputs)
    probs = tf.nn.softmax(logits[0], axis=-1)
    n += 1

    # Use get_top_k function to get top predictions
    top_k_predictions = get_top_k(probs, k=1)
    top_label, top_score = top_k_predictions[0]
    results = [{"label": top_label, "score": float(top_score)}]

    # Overlay text on image
    img = Image.open(io.BytesIO(contents)).resize((224, 224))
    img = overlay_text_on_image(img, f"{top_label}: {top_score:.2f}")
    img.save(f"/Downloads/{timestamp}_overlay_{file.filename}")

    return JSONResponse(content={"predictions": results})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)