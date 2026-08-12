from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import traceback
from backend.src.inference import predict
app=FastAPI(title="Fruit Classification API",version="1.0.0")

@app.get("/")
def root():
    return {"status": "ok", "message": "Fruit Classification API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}
  
@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = predict(image)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
