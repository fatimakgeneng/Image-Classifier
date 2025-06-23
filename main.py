import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from PIL import Image


def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Image Classifier", page_icon="🖼", layout="centered")

    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what is in it")





# Sidebar: Developer Info
    st.sidebar.markdown("<h1 style='text-align: center; margin-top: -40px; margin-bottom: 10px;'>👩‍💻 Developed By</h1>", unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        <div style='display: flex; justify-content: center;'>
            <img src='https://media.licdn.com/dms/image/v2/D4E03AQG8lxYDv2d_jA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1682492456789?e=1756339200&v=beta&t=FUzyJGaPbESA-sLaQfGiBBeIHeHXmvf3AJex-YLl53M' style='width: 150px; border-radius: 10px;'/>
        </div>
        """,
        unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        <div style='text-align: center; padding-top: 5px;'>
            <h2 style='margin-bottom: 1px;'>Fatima Khan</h2>
        
        👩‍🔬 *Certified Agentic & Robotic AI Engineer | Generative AI Expert | JAMStack Developer*

        Passionate about building smart, responsive apps using cutting-edge web technologies and AI models.  
        Co-founder of **The Constructors Development Group** — shaping the future with code & curiosity.
        </div>
        """,
        unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        <div style='display: flex; justify-content: center; gap: 20px; padding-top: 2px;'>
            <a href='https://www.linkedin.com/in/fatimakgeneng/' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg' width='24' />
            </a>
            <a href='https://github.com/fatimakgeneng' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg' width='24' />
            </a>
        </div>
        """,
        unsafe_allow_html=True)




    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image=st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )
        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing..."):
                image=Image.open(uploaded_file)
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions:")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")

if __name__ == "__main__":
    main()