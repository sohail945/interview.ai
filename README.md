# Interview.ai

**Interview.ai** is an AI-powered platform designed to automate and enhance the hiring process by evaluating candidates' personality traits and interview performance. It leverages a multi-modal approach combining video, audio, and text analysis to generate an interview score that reflects the candidate's communication skills and overall personality, helping recruiters make more informed decisions.

## Key Features
- **Multi-modal Analysis:** Processes video, audio, and text data to predict personality traits and interview performance.
- **Video Analysis:** Uses computer vision models to assess facial expressions, body language, and visual cues.
- **Audio Analysis:** Extracts features from the audio to evaluate speech patterns, tone, and emotion.
- **Text Analysis:** Utilizes natural language processing (NLP) to analyze the candidate's responses and language usage.
- **Real-time Scoring:** Provides a real-time score for candidates based on a combination of all three data modalities.

## Project Overview
Interview.ai integrates state-of-the-art machine learning models to provide a comprehensive assessment of a candidate's suitability for a role based on their interview responses. By analyzing various aspects such as visual cues (via CNN models), audio signals (via audio feature extraction), and text responses (via NLP models), the platform can generate a score that assists recruiters in the decision-making process.

The system includes:
- **Video Model (Visual Analysis):** A CNN-based model to extract meaningful insights from video data.
- **Audio Model (Speech Analysis):** An audio classification model to process speech data and extract relevant features like tone and clarity.
- **Text Model (NLP Analysis):** A deep learning model trained to analyze and process the text data, such as transcriptions from the video.

## Installation

### Prerequisites
- Python 3.x
- TensorFlow
- OpenCV
- librosa
- Streamlit
- moviepy
- SpeechRecognition

### Step 1: Clone the Repository
Clone this repository to your local machine using the following command:
```bash
git clone https://github.com/yourusername/Interview.ai.git
cd Interview.ai
