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
    pixels = data['pixels'].tolist()  # Get the list of pixel data from the DataFrame
    width, height = 48, 48  # Set the width and height of the images
    faces = []

    # Process each pixel sequence
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]  # Convert pixel string to list of integers
        face = np.asarray(face).reshape(width, height)  # Reshape list into a 48x48 image
        faces.append(face)  # Add the processed image to the list

    faces = np.array(faces)  # Convert list of images into a numpy array
    faces = np.expand_dims(faces, -1)  # Add a channel dimension (grayscale images)
    faces = faces / 255.0  # Normalize pixel values to the range [0, 1]

    emotions = to_categorical(data['emotion'], num_classes=7)  # Convert emotion labels to one-hot encoded format
    return faces, emotions

faces, emotions = preprocess_data(data)  # Preprocess the data

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.2, random_state=42)
# 80% of the data will be used for training, and 20% for validation

# Define the Model
def create_emotion_model():
    model = models.Sequential([
        # ReLU (Rectified Linear Unit) adalah fungsi aktivasi yang sering digunakan dalam jaringan saraf tiruan, 
        # terutama dalam lapisan konvolusi dan lapisan tersembunyi dari model pembelajaran mendalam. 
        # Fungsi ini diperkenalkan untuk mengatasi beberapa masalah yang muncul dengan fungsi aktivasi lainnya, 
        # seperti sigmoid atau tanh.
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),  # First convolutional layer
        layers.BatchNormalization(),  # Normalize activations to improve training
        layers.MaxPooling2D((2, 2)),  # Reduce the spatial dimensions of the image
        layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
        layers.BatchNormalization(),  # Normalize activations
        layers.MaxPooling2D((2, 2)),  # Reduce dimensions
        layers.Conv2D(64, (3, 3), activation='relu'),  # Third convolutional layer
        layers.BatchNormalization(),  # Normalize activations
        layers.MaxPooling2D((2, 2)),  # Reduce dimensions
        layers.Flatten(),  # Flatten the 3D feature maps to 1D
        layers.Dense(64, activation='relu'),  # Fully connected layer with 64 neurons
        layers.Dropout(0.5),  # Dropout layer to prevent overfitting
        layers.Dense(7, activation='softmax')  # Output layer with 7 neurons (one for each emotion class)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
model = create_emotion_model()  # Create the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
# Train the model on the training data for 10 epochs and validate on the validation data

# Save the model
model.save('app/models/emotion_model.h5')  # Save the trained model to a file
