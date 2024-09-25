import os
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Load VGG19 model without the top layers
base_model = VGG19(include_top=False, input_shape=(128, 128, 3))

# Define the new architecture
x = base_model.output
flat = Flatten()(x)

# Ensure these dense layers match the saved model architecture
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)

# Create the final model
model_03 = Model(inputs=base_model.input, outputs=output)

# Load model weights from the Desktop
model_path = '/Users/mohdalfaid/Desktop/pneumonia/vgg_unfrozen.h5'
try:
    model_03.load_weights(model_path)
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {e}")

# Function to classify the result
def get_class_name(class_no):
    if class_no == 0:
        return "Normal"
    elif class_no == 1:
        return "Pneumonia"

# Function to predict result based on image
def get_result(image):
    # Open and process the uploaded image
    image = Image.open(image)
    image = image.resize((128, 128))

    # Convert image to array and ensure it has three channels
    image = np.array(image)
    
    if image.ndim == 2:  # Grayscale image
        image = np.stack((image,) * 3, axis=-1)  # Convert to RGB
    elif image.shape[2] == 1:  # Single channel image
        image = np.concatenate([image, image, image], axis=-1)  # Convert to RGB

    # Ensure the shape is (1, 128, 128, 3)
    input_img = np.expand_dims(image, axis=0)  # Add batch dimension

    # Print the shape for debugging
    print("Input image shape:", input_img.shape)

    input_img = input_img.astype('float32') / 255.0  # Normalize the image
    result = model_03.predict(input_img)
    result_class = np.argmax(result, axis=1)
    return result_class[0]  # Return the class index directly

# Streamlit app
st.title("Pneumonia Detection Web App")
st.write("Upload an X-ray image to classify whether it's Normal or Pneumonia.")

# Image upload and prediction
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Get result and display
    label = get_result(uploaded_file)
    result = get_class_name(label)
    st.write(f"Prediction: **{result}**")
