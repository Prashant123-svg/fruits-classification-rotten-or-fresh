from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from src.inference import predict
app=FastAPI(title="Fruit Classification API",version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}
  
@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = predict(image)
    return result