from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('alcohol_speech_model.h5')

def extract_mfcc_from_audio(audio_path, n_mfcc=40, max_pad_len=174):
    audio, sample_rate = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save uploaded file
    file_path = 'temp.wav'
    file.save(file_path)
    
    # Extract features and predict
    mfcc = extract_mfcc_from_audio(file_path)
    X = mfcc[np.newaxis, ..., np.newaxis]
    pred_prob = model.predict(X)[0][0]
    pred_label = 'Intoxicated' if pred_prob > 0.5 else 'Sober'
    
    return jsonify({'prediction': pred_label, 'probability': float(pred_prob)})

if __name__ == '__main__':
    app.run(debug=True)
