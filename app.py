import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- Configuration ---
CLASS_NAMES = ['animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel', 'fish sea_food red_mullet', 'fish sea_food red_sea_bream', 'fish sea_food sea_bass', 'fish sea_food shrimp', 'fish sea_food striped_red_mullet', 'fish sea_food trout']

# Cache the model
@st.cache_resource
def load_my_model():
    return load_model('fish_classifier_97.h5')

model = load_my_model()

# --- Streamlit UI ---
st.title(" Multiclass Fish Image Classification")
st.write("Upload an image of a fish, and the custom CNN will predict its species!")

# File uploader
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    st.write("Classifying...")
    
    # 2. Preprocess the image to match our Colab training data
    # Convert to RGB in case a PNG has a transparent channel
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0  # Crucial: Rescale just like ImageDataGenerator
    img_array = np.expand_dims(img_array, axis=0) # Create a batch of 1
    
    # 3. Make Prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    # 4. Output results
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    
    st.success(f"**Prediction:** {predicted_class_name}")
    st.info(f"**Confidence Score:** {confidence:.2f}%")
