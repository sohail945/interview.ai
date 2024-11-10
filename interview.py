import streamlit as st
import moviepy.editor as mp
import speech_recognition as sr
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import librosa
import matplotlib.pyplot as plt

# Load Pre-trained Models (Replace paths with actual model paths)
nlp_model_path = "C:\\Users\\Muhammad Shoaib\\Desktop\\Interview_ai\\NLP_model.h5"
audio_model_path = "C:\\Users\\Muhammad Shoaib\\Desktop\\Interview_ai\\Audio_model.h5"
visual_model_path = "C:\\Users\\Muhammad Shoaib\\Desktop\\Interview_ai\\Visual_model.h5"

nlp_model = tf.keras.models.load_model(nlp_model_path)
audio_model = tf.keras.models.load_model(audio_model_path)
visual_model = tf.keras.models.load_model(visual_model_path)

# Set up tokenizer for text processing
max_length = 40
tokenizer = Tokenizer(num_words=14500)

# Text Extraction Function
def extract_text_from_video(video_file):
    video_clip = mp.VideoFileClip(video_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile("audio.wav")

    r = sr.Recognizer()
    with sr.AudioFile("audio.wav") as source:
        audio = r.record(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

# Audio Feature Extraction (Mel-spectrogram)
def extract_audio_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    log_mel_spectrogram_resized = cv2.resize(log_mel_spectrogram, (124, 124))
    mel_spectrogram_rgb = np.repeat(log_mel_spectrogram_resized[..., np.newaxis], 3, axis=-1)
    return np.expand_dims(mel_spectrogram_rgb, axis=0)

# Visual Feature Extraction (frames from video)
def extract_visual_features(video_path, sequence_length=15, img_size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(total_frames // sequence_length, 1)
    for i in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_skip)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, img_size)
            frames.append(frame)
    cap.release()
    return np.expand_dims(np.array(frames).astype('float32') / 255.0, axis=0) if frames else None

# Prediction Function (Late Fusion)
def predict_interview_score(video_file):
    text = extract_text_from_video(video_file)
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence = pad_sequences(text_sequence, maxlen=max_length)
    nlp_pred = nlp_model.predict(text_sequence)[0][0] if text_sequence is not None else None

    audio_features = extract_audio_features("audio.wav")
    audio_pred = audio_model.predict(audio_features)[0][0] if audio_features is not None else None
    audio_pred /= 100

    visual_features = extract_visual_features(video_file)
    visual_pred = visual_model.predict(visual_features)[0][0] if visual_features is not None else None

    predictions = [pred for pred in [nlp_pred, audio_pred, visual_pred] if pred is not None]
    final_score = np.mean(predictions) if predictions else None

    return final_score

# Streamlit Interface
st.set_page_config(page_title="Interview.ai", page_icon="🎤", layout="wide")

# Custom Title and Tagline
st.markdown("""
    <style>
        .title { font-size: 40px; color: #2A2A2A; font-weight: bold; }
        .tagline { font-size: 20px; color: #555555; }
        .score { font-size: 30px; font-weight: bold; color: #0073e6; }
        .header { text-align: center; margin-bottom: 20px; }
    </style>
    <div class="header">
        <div class="title">Interview.ai</div>
        <div class="tagline">Predict your personality for an interview based on video content</div>
    </div>
""", unsafe_allow_html=True)

# File upload section
st.markdown("### Upload Your Video for Interview Prediction")
uploaded_video = st.file_uploader("Upload a video (MP4 format, max size 200MB)", type=["mp4"])

if uploaded_video is not None:
    # Check file size (in bytes; 200MB = 209715200 bytes)
    if uploaded_video.size > 209715200:
        st.error("Error: File size exceeds 200MB limit.")
    else:
        # Save the uploaded video temporarily
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())

        # Display video preview
        st.video("uploaded_video.mp4")

        # Display prediction status
        st.markdown("#### Predicting interview score...")
        
        # Get the final score
        final_score = predict_interview_score("uploaded_video.mp4")

        if final_score is not None:
            st.markdown(f'<p class="score">Final Interview Score: {final_score:.2f}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="score" style="color: red;">Error: Unable to predict the score.</p>', unsafe_allow_html=True)
