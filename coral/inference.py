from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import io
from PIL import Image
import uvicorn
import os
from datetime import datetime
import tensorflow as tf
import pathlib

app = FastAPI()

# Load the model from the saved path
model_path = '/home/movinet_model'
model = tf.saved_model.load(model_path)

labels_path = pathlib.Path('kinetics_600_labels.txt')
lines = labels_path.read_text().splitlines()
KINETICS_600_LABELS = np.array([line.strip() for line in lines])
images = []
init_state = None
n=0

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
        init_state = model.init_states(video[tf.newaxis, ...].shape)

    # Perform inference
    inputs = init_state.copy()
    inputs['image'] = video[tf.newaxis, n:n+1, ...]
    logits, state = model(inputs)
    probs = tf.nn.softmax(logits[0], axis=-1)
    n += 1

    # Use get_top_k function to get top predictions
    top_k_predictions = get_top_k(probs, k=3)
    results = [{"label": label, "score": float(score)} for label, score in top_k_predictions]
    
    return JSONResponse(content={"predictions": results})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)