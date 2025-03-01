import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
import math

model = load_model('C:/Users/shivo/OneDrive/Desktop/Honors Proj/model_cnn_DA.keras')

new_input = Input(shape=(130, 13, 1)) 

output = model.layers[-1].output 

new_model = Model(inputs=new_input, outputs=output)

new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

new_model.save('modified_model.keras')


# Genre list used in the prediction
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Function to extract MFCCs from audio data
def extract_mfccs_from_audio(audio, sr, duration=30, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    # Prepare to extract features
    samples_per_track = sr * duration
    samps_per_segment = int(samples_per_track / num_segments)
    mfccs_per_segment = math.ceil(samps_per_segment / hop_length)

    mfccs = []
    for seg in range(num_segments):
        start_sample = seg * samps_per_segment
        end_sample = start_sample + samps_per_segment

        mfcc = librosa.feature.mfcc(y=audio[start_sample:end_sample], sr=sr, n_mfcc=n_mfcc,
                                    n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        if len(mfcc) == mfccs_per_segment:
            mfccs.append(mfcc.tolist())

    return np.array(mfccs)

# Function to make genre prediction
def make_prediction(model, X):
    preds_num = []
    preds_name = []

    for X_current in X:
        # Adjust shape to match model's input expectations
        if X_current.shape[0] > 130:
            X_current = X_current[:130]  # Truncate if longer
        elif X_current.shape[0] < 130:
            # Optionally pad if shorter, but truncating usually works for consistent inputs
            X_current = np.pad(X_current, ((0, 130 - X_current.shape[0]), (0, 0)), mode='constant')

        X_current = X_current[..., np.newaxis]  # Add the extra channel dimension
        X_current = X_current[np.newaxis, ...]  # Add batch dimension
        pred = model.predict(X_current)
        pred = np.argmax(pred, axis=1)  # predicted index
        preds_num.append(pred[0])
        preds_name.append(genres[pred[0]])

    return preds_num, preds_name



# UI design
st.title('🎶 Music Genre Classification using CNN')

st.markdown("""
### Welcome to the Music Genre Classifier!

This app allows you to upload an audio file, and it will predict the genre using a **Convolutional Neural Network (CNN)** model trained on music features.

#### Instructions:
1. Upload an audio file in `.wav`, `.mp3`, or `.ogg` format.
2. Click **Classify Genre** to get the predicted genre.
3. Explore more by following the provided links!
""")

# File uploader section
st.markdown("### 1. Upload an audio file below:")
uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])

if uploaded_file is not None:
    # Load audio and display playback
    audio_bytes = uploaded_file.read()
    audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=22500)  # Load with target sample rate

    # Audio playback
    st.audio(audio_bytes, format='audio/wav')

    # Genre prediction button
    if st.button('🎤 Classify Genre'):
        # Extract MFCC features from the uploaded audio
        mfccs = extract_mfccs_from_audio(audio_data, sr)

        # Make predictions
        _, predicted_genre_name = make_prediction(model, mfccs)

        # Display predicted genre
        st.markdown("### 2. Predicted Genre:")
        st.success(f'The predicted genre is: **{predicted_genre_name[0]}** 🎧')

        # Learn more links
        st.markdown("### 3. Learn More:")
        cnn_link = "https://www.google.com/search?q=Convolutional+Neural+Networks+CNN"
        genre_link = f"https://www.google.com/search?q={predicted_genre_name[0]}+music+genre"

        st.markdown(f"- [Learn about CNN]({cnn_link})")
        st.markdown(f"- [Learn more about {predicted_genre_name[0]} music genre]({genre_link})")
