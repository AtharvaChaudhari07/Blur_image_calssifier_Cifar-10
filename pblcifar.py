import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Load the saved TensorFlow model
model = tf.keras.models.load_model('PBL.hdf5')

# Preprocess image for model input
def preprocess_image(image):
    image = image.resize((32, 32))  # Resize image to match CIFAR-10 input shape
    image = np.array(image)
    image = image.astype('float32') / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Create the Streamlit application
def main():
    st.title("CIFAR-10 Image Classification")

    # Create a file uploader component
    uploaded_file = st.file_uploader("Upload an image")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        # Perform inference using the loaded model
        prediction = model.predict(preprocessed_image)
        class_index = np.argmax(prediction)
        class_label = "Class label: " + str(class_index)

        # Display the predicted class label
        st.write(classes[class_index])

if __name__ == '__main__':
    main()
