import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model
from spotify import get_songs_by_emotion

st.set_page_config(page_title="MoodMatch", page_icon="🎭") #should be at the starting of page 
st.title("MoodMatch: Emotion-Based Music Recommender")

@st.cache_resource
def load_emotion_model():
    return load_model("fer_model.h5") #fetching the saved model in the same directory 

model = load_emotion_model()

def predict_emotion_from_image(pil_image):  #complete model to predict mood, uses previous model 
    emotion_labels = ['Angry', 'Calm', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] #FER2013  has these 7 classes so mapping to these labels
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image_rgb = np.array(pil_image.convert("RGB"))          #cv is computer vision
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)

    if len(faces) == 0:
        return None, image_rgb

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = cv2.resize(roi_gray, (48, 48))
    cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0) / 255.0

    prediction = model.predict(cropped_img)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]

    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    cv2.putText(image_rgb, emotion, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return emotion, image_rgb

input_option = st.radio("Choose input method:", ["Use Webcam", "Upload Image"])
image = None

if input_option == "Use Webcam":
    image_data = st.camera_input("Take a photo")
    if image_data:
        image = Image.open(image_data)

elif input_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

if image:
    detected_emotion, annotated_image = predict_emotion_from_image(image)

    if detected_emotion:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(annotated_image, caption=f"Detected: {detected_emotion}", channels="RGB")

        st.success(f"Detected Mood: **{detected_emotion}**")

        st.write(f"Finding top songs for mood: **{detected_emotion}**...")
        time.sleep(2)
        songs = get_songs_by_emotion(detected_emotion, limit=5)

        if songs:
            st.markdown("### Top Songs for Your Mood:")
            for i, song in enumerate(songs, 1):
                st.markdown(f"**{i}. {song['title']}** by *{song['artist']}*")
                st.write(f"Released: {song['release_date']}")
                st.markdown("---")
        else:
            st.warning("Sorry, couldn't find songs right now. Try again later.")
    else:
        st.warning("No face detected in the image. Please try a clearer photo.")
else:
    st.info("Upload or capture a face image to get started.")
