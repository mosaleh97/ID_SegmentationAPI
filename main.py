# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from ultralytics import YOLO
import fastapi
import uvicorn
import json
import inference.get_card as get_card
import cv2
import numpy as np
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
import base64


# Read json file having the configuration
with open('config/model_config.json') as f:
    config = json.load(f)

# Load YOLOv8s model
model = YOLO(config['model_path'])  # Load YOLOv8s model

# define FastAPI app
app = fastapi.FastAPI()


# define a post request endpoint that receives the image and return the wrapped card
@app.post("/get_id_card")
async def get_id_card(image: fastapi.UploadFile = fastapi.File(...)):
    # read the image
    img = await image.read()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite('input.png', img)

    # get the mask of the image
    mask = get_card.predict_mask(img, model)

    # write the mask
    cv2.imwrite('outputresized.png', mask)

    wrapped_card = get_card.get_wrapped_card(img, mask)

    if wrapped_card is None:
        error_msg = {"detail": config['error_message']}
        return JSONResponse(content=error_msg, status_code=422)

    # write the wrapped card
    cv2.imwrite('outputwrapped.jpeg', wrapped_card)

    # Convert image to bytes
    _, img_encoded = cv2.imencode('.jpeg', wrapped_card)
    img_bytes = img_encoded.tobytes()

    # Return the image as a streaming response
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
