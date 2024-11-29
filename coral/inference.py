from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import io
from PIL import Image
import uvicorn
import os
from datetime import datetime
import tensorflow as tf

app = FastAPI()

# Load the model from the saved path
model_path = '/home/movinet_model'
model = tf.saved_model.load(model_path)

# Load labels
with open('kinetics_600_labels.txt', 'r') as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Save the file with a timestamped name in /Downloads
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f"/Downloads/{timestamp}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Open the image and resize it
    img = Image.open(io.BytesIO(contents)).resize((224, 224))
    
    # Convert the image to a NumPy array and normalize it
    img = np.array(img) / 255.0

    # Initialize states
    init_state = model.init_states(img[tf.newaxis, ...].shape)

    # Perform inference
    inputs = init_state.copy()
    inputs['image'] = img[tf.newaxis, ...]
    logits, state = model(inputs)
    probs = tf.nn.softmax(logits[0], axis=-1)
    top_k = tf.argsort(probs, axis=-1, direction='DESCENDING')[:3]
    results = [{"label": labels[i], "score": float(probs[i])} for i in top_k.numpy()]
    
    return JSONResponse(content={"predictions": results})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)