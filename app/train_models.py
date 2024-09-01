import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Load Dataset
data_path = "data/fer2013.csv"  # Path to the dataset file
data = pd.read_csv(data_path)    # Load the dataset from the CSV file into a DataFrame

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
    faces = faces / 255.0  # Normalisasi data

    emotions = to_categorical(data['emotion'], num_classes=7)
    return faces, emotions


faces, emotions = preprocess_data(data)  # Preprocess the data

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.2, random_state=42)
# 80% of the data will be used for training, and 20% for validation

# Define the Model
def create_emotion_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train the model
model = create_emotion_model()  # Create the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
# Train the model on the training data for 10 epochs and validate on the validation data

# Save the model
model.save('app/models/emotion_model.h5')  # Save the trained model to a file
