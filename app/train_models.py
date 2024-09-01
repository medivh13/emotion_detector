import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Load Dataset
data_path = "data/fer2013.csv"
data = pd.read_csv(data_path)

# Preprocess Data
def preprocess_data(data):
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []

    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        faces.append(face)

    faces = np.array(faces)
    faces = np.expand_dims(faces, -1)
    emotions = to_categorical(data['emotion'], num_classes=7)
    return faces, emotions

faces, emotions = preprocess_data(data)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.2, random_state=42)

# Define the Model
def create_emotion_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
model = create_emotion_model()
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model
model.save('app/models/emotion_model.h5')
