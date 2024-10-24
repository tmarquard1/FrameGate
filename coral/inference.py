from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
#from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import uvicorn

app = FastAPI()
interpreter = make_interpreter('mobilenet_v1_1.0_224_edgetpu.tflite')
interpreter.allocate_tensors()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1  # Normalize to [-1, 1]
    # img_array = preprocess_input(img_array)

    common.set_input(interpreter, img_array)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=3)

    results = [{"label": c.id, "description": c.label, "score": float(c.score)} for c in classes]
    return JSONResponse(content={"predictions": results})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # vicorn coral.inference:app --host 0.0.0.0 --port 8000 --reload