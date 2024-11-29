from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
import numpy as np
import io
from PIL import Image
import uvicorn
import os
from datetime import datetime

app = FastAPI()
# Note this is version 1 of the model
interpreter = make_interpreter('movinet_stream_a2_edgetpu.tflite') 
interpreter.allocate_tensors()

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
    img = img/255.0

    # Perform inference
    common.set_input(interpreter, img)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=3)
    results = [{"label": labels[c.id], "score": float(c.score)} for c in classes]
    
    return JSONResponse(content={"predictions": results})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)