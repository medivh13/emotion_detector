# Real-Time Emotion Detection

## Description
This project is a real-time facial emotion detection application using TensorFlow, FastAPI, and WebSocket. The application utilizes a webcam to capture images of the face, which are then analyzed by the server to detect emotions. The detected emotions are displayed live on the web page.

## Prerequisites
Ensure you have Python 3.x and pip installed on your system.

## Installation and Setup

1. **Clone the Repository**
2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
3. **Run the Server**
    ```bash
   uvicorn app.main:app --reload
4. **Access the Application**
    ```bash
    Open your browser and navigate to http://localhost:8000 to view the application.
    Ensure your webcam is connected and allowed access to the site.
