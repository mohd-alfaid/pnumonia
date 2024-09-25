import os
import numpy as np
from PIL import Image
import cv2
import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Load model weights from Desktop
model_03 = VGG19(include_top=False, input_shape=(128, 128, 3))
x = model_03.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(model_03.inputs, output)

# Update the path to the weights file
model_03.load_weights('/Users/mohdalfaid/Desktop/pneumonia/vgg_unfrozen.h5')

# Function to classify the result
def get_class_name(class_no):
    if class_no == 0:
        return "Normal"
    elif class_no == 1:
        return "Pneumonia"

# Function to predict result based on image
def get_result(image):
    image = Image.open(image)
    image = image.resize((128, 128))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model_03.predict(input_img)
    result_class = np.argmax(result, axis=1)
    return result_class

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
