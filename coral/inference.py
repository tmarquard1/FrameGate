from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
import numpy as np
import io
from PIL import Image
import uvicorn

app = FastAPI()
interpreter = make_interpreter('mobilenet_v1_1.0_224_edgetpu.tflite')
interpreter.allocate_tensors()

# Load labels
with open('imagenet_labels.txt', 'r') as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((224, 224))

    common.set_input(interpreter, img)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=3)
    print(classes)
    results = [{"label": labels[c.id], "score": float(c.score)} for c in classes]
    return JSONResponse(content={"predictions": results})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # vicorn coral.inference:app --host 0.0.0.0 --port 8000 --reload