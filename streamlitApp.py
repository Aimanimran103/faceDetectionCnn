import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your pre-trained Face Detection CNN model
model = tf.keras.models.load_model('path_to_face_detection_model.h5')

# Title of the app
st.title("Face Detection using CNN")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Convert to OpenCV format
    img = np.array(image)

    # Preprocess the image (resize, normalize)
    img_resized = cv2.resize(img, (224, 224))  # Assuming the model input size is 224x224
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_resized)

    # Assuming the model outputs bounding box coordinates
    for box in prediction:
        x, y, w, h = box
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

    # Display the detection results
    st.image(img, caption="Face Detection Result", use_column_width=True)
import matplotlib.pyplot as plt

def plot_history(history):
    st.subheader("Model Performance")
    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    st.pyplot(plt)

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    st.pyplot(plt)

# Call this function with the training history if available
# plot_history(history)
