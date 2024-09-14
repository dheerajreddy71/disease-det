# Library imports
import numpy as np
import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models

# Load the PyTorch model
model = models.resnet18()  # Replace with your model architecture
model.load_state_dict(torch.load('plant_disease_model.pth'))
model.eval()

# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust mean and std if necessary
])

# Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Convert to PIL Image for transformation
        pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
        
        # Apply transformations
        input_image = transform(pil_image).unsqueeze(0)  # Add a batch dimension
        
        # Make Prediction
        with torch.no_grad():
            outputs = model(input_image)
            _, predicted = torch.max(outputs, 1)
            result = CLASS_NAMES[predicted.item()]
        
        # Display the image and result
        st.image(opencv_image, channels="BGR")
        st.title(f"This is a {result.split('-')[0]} leaf with {result.split('-')[1]}")
