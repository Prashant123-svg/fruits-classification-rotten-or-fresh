
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Request
from PIL import Image
import io
import traceback
from backend.src.inference import predict
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import time
from backend.metrics import REQUEST_COUNT, REQUEST_LATENCY, PREDICTION_COUNT
app = FastAPI(title="Fruit Classification API", version="1.0.0")

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    return response

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
        PREDICTION_COUNT.labels(label=result["label"]).inc()
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
