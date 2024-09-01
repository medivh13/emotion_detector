import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load the trained model
model = tf.keras.models.load_model("app/models/emotion_model.h5")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                await websocket.send_text("Error: Received empty image")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            predicted_emotion_index = np.argmax(prediction)
            predicted_emotion = emotion_labels[predicted_emotion_index]
            await websocket.send_text(predicted_emotion)
    except WebSocketDisconnect:
        print("Client disconnected")

# HTML page to serve
@app.get("/")
async def get():
    return templates.TemplateResponse("index.html", {"request": {}})
