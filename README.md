# Pneumonia Detection Web App

This web app detects pneumonia from chest X-ray images using a pre-trained VGG19 model. The app is built using Streamlit, making it easy to deploy and use.

## Features
- Upload chest X-ray images in `.jpg`, `.png`, or `.jpeg` formats.
- The app classifies the image as either "Normal" or "Pneumonia" using the pre-trained VGG19 model.
- Streamlit-based user interface for quick and easy deployment.

## How It Works
1. A VGG19 convolutional neural network (CNN) is pre-trained to process chest X-ray images.
2. The app accepts a chest X-ray image and preprocesses it to fit the input size for the model (128x128).
3. The model predicts whether the image corresponds to a normal or pneumonia condition.
4. The result is displayed on the app interface.

## Requirements
To run the web app, make sure you have the following installed:

- Python 3.7 or higher
- TensorFlow 2.x
- Streamlit
- OpenCV
- Pillow (PIL)
- NumPy

You can install all the required libraries using the `requirements.txt` file.

## Installation

1. Clone the repository or download the project:
    ```bash
    git clone https://github.com/yourusername/pneumonia-detection-webapp.git
    cd pneumonia-detection-webapp
    ```

2. Install the dependencies using the provided `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app using Streamlit:
    ```bash
    streamlit run app.py

- **app.py**: The main application code.
- **vgg_unfrozen.h5**: Pre-trained weights for the VGG19 model (make sure to include this file in the directory).
- **README.md**: Instructions and documentation.
- **requirements.txt**: List of dependencies.

## Model Details
The VGG19 model used in this project has been pre-trained for binary classification:
- **Input shape**: 128x128x3 (RGB images)
- **Output classes**: 2 (Normal, Pneumonia)

## How to Use
1. Upload a chest X-ray image using the file uploader.
2. Wait for the app to classify the image.
3. The app will display whether the X-ray is **Normal** or **Pneumonia**.

## Acknowledgments
The model is based on the VGG19 architecture and uses TensorFlow/Keras for training. Special thanks to the contributors of open-source tools and libraries used in this project.
