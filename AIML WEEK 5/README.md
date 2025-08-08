**Alcohol Detection from Speech Using CNN**
**Project Overview**

This project builds a Convolutional Neural Network (CNN) model to detect alcohol intoxication based on speech audio samples. Using Mel-frequency cepstral coefficients (MFCC) extracted from audio files, the model classifies voice recordings as either "Sober" or "Intoxicated".
Features

    Audio preprocessing and feature extraction using librosa

    CNN-based deep learning model for classification

    Train/test split with model evaluation metrics

    Flask API for live speech prediction from microphone input

**Dataset**

    Speech samples divided into two categories:

        data/sober/ - Audio recordings of sober speech

        data/intoxicated/ - Audio recordings of intoxicated speech

    All audio files should be in .wav format


Create and activate a Python virtual environment:

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows CMD
.\.venv\Scripts\activate.bat
# Linux/Mac
source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

**Usage**
Step 1: Feature Extraction

Extract MFCC features from audio samples and save them for training:

python feature_extraction.py

Step 2: Train the Model

Train the CNN model on extracted features:

python train_model.py

Step 3: Evaluate Model

(Optional) Evaluate the trained model’s performance:

python evaluate_model.py

Step 4: Run Flask API Server

Start the Flask server for live alcohol detection via microphone input:

python app.py

**Project Structure**

AlcoholSpeechDetection/
│
├── data/
│   ├── sober/
│   └── intoxicated/
│
├── features.npy
├── labels.npy
├── alcohol_speech_model.h5
│
├── feature_extraction.py
├── train_model.py
├── evaluate_model.py
├── app.py
│
├── requirements.txt
└── README.md

**Dependencies**

    numpy

    pandas

    librosa

    sounddevice

    soundfile

    scikit-learn

    tensorflow

    matplotlib

    flask

    tqdm

    seaborn

**Notes**
    Ensure your dataset contains valid .wav files.

    The model expects extracted features saved as .npy files.

    Use Python 3.7 or higher for best compatibility.
    