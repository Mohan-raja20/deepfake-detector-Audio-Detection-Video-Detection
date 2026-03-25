import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import cv2
from deepface import DeepFace

# -----------------------------
# Audio Functions
# -----------------------------
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1), mfcc, sr

def plot_mfcc(mfcc, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax)
    ax.set_title('MFCC')
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)

# Load audio model
model = joblib.load("rf_model.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔍 Deepfake Detector")

tab1, tab2 = st.tabs(["Audio Detection", "Video Detection"])

# -----------------------------
# Tab 1: Audio Detection
# -----------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        # Save uploaded file
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())

        # Extract features
        features, mfcc, sr = extract_mfcc("temp.wav")
        prediction = model.predict(features.reshape(1, -1))[0]
        confidence = model.predict_proba(features.reshape(1, -1))[0]

        # Show results
        label = "🧑 Real" if prediction == 0 else "🤖 Fake"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: Real = {confidence[0]:.2f}, Fake = {confidence[1]:.2f}")

        # Show MFCC plot
        st.subheader("MFCC Visualization")
        plot_mfcc(mfcc, sr)

# -----------------------------
# Tab 2: Video Detection
# -----------------------------
with tab2:
    video_file = st.file_uploader("Upload a Video", type=["mp4"])
    if video_file is not None:
        tmp_path = "uploaded_video.mp4"
        with open(tmp_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            st.image(frame, channels="BGR", caption="Sample Frame")
            try:
                analysis = DeepFace.analyze(frame, actions=['deepfake'])
                st.write("Prediction:", analysis['deepfake']['label'])
                st.write("Confidence:", analysis['deepfake']['confidence'])
            except Exception as e:
                st.error(f"Error analyzing frame: {e}")
