from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fast_alpr import ALPR

from PIL import Image
import io
import numpy as np

import logging
import os

# Force ONNX Runtime to use CPU provider to avoid CoreML issues
os.environ['ORT_FORCE_CPU'] = '1'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

detector_model = 'yolo-v9-t-256-license-plate-end2end'
ocr_model = 'cct-xs-v1-global-model'

app = FastAPI()
alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model)

@app.post("/process_frame")
async def process_frame(frame: UploadFile = File(...)):
    image_bytes = await frame.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Your plate detection and OCR goes here
    result = process_image_with_your_model(image)

    return JSONResponse({"plates": result})

def process_image_with_your_model(image):
    try:
        # Convert PIL Image to numpy array (RGB format)
        img_array = np.array(image)
        
        # Validate image dimensions
        if img_array.size == 0:
            logger.warning("Empty image received")
            return []
        
        logger.info(f"Processing image with shape: {img_array.shape}")
        
        # Run ALPR prediction
        model_results = alpr.predict(img_array)
        detected_plates = [result.ocr.text for result in model_results if result.ocr]
        
        logger.info(f"Detected plates: {detected_plates}")
        return detected_plates
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return []

# Mount static files AFTER defining API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
