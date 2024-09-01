import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

# Create a FastAPI instance
app = FastAPI()

# Set up the template directory for rendering HTML pages
templates = Jinja2Templates(directory="app/templates")

# Serve static files (like CSS or JavaScript) from the "static" folder
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load the pre-trained emotion detection model
model = tf.keras.models.load_model("app/models/emotion_model.h5")
# Define the list of emotion labels used by the model
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# WebSocket endpoint for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection
    await websocket.accept()
    try:
        while True:
            # Receive image data sent by the client
            data = await websocket.receive_bytes()
            # Convert the received bytes into a NumPy array
            np_arr = np.frombuffer(data, np.uint8)
            # Decode the NumPy array into an image
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Check if the image was successfully decoded
            if img is None:
                await websocket.send_text("Error: Received empty image")
                continue

            # Convert the image to grayscale (single channel)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Resize the image to the size expected by the model (48x48 pixels)
            img = cv2.resize(img, (48, 48))
            # Add an extra dimension for the color channel (grayscale)
            img = np.expand_dims(img, axis=-1)
            # Add another dimension for batch size
            img = np.expand_dims(img, axis=0)
            # Make a prediction using the trained model
            prediction = model.predict(img)
            print(prediction)
            # Get the index of the highest probability class
            predicted_emotion_index = np.argmax(prediction)
            # Map the index to the corresponding emotion label
            predicted_emotion = emotion_labels[predicted_emotion_index]
            # Send the detected emotion back to the client
            await websocket.send_text(predicted_emotion)
    except WebSocketDisconnect:
        # Handle the case when the client disconnects
        print("Client disconnected")

# HTTP endpoint to serve the main HTML page
@app.get("/")
async def get():
    # Render the "index.html" template and return it as the response
    return templates.TemplateResponse("index.html", {"request": {}})
