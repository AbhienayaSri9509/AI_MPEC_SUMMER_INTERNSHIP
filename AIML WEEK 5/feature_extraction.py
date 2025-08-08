
import os
import numpy as np
import librosa

def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def process_folder(folder_path, label, features, labels):
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            file_path = os.path.join(folder_path, file)
            mfcc = extract_mfcc(file_path)
            features.append(mfcc)
            labels.append(label)

if __name__ == "__main__":
    data_dir = 'data'  # Your folder containing 'sober' and 'intoxicated' subfolders
    features = []
    labels = []

    sober_path = os.path.join(data_dir, 'sober')
    process_folder(sober_path, 0, features, labels)

    intox_path = os.path.join(data_dir, 'intoxicated')
    process_folder(intox_path, 1, features, labels)

    features = np.array(features)
    labels = np.array(labels)

    np.save('features.npy', features)
    np.save('labels.npy', labels)
    print("Features and labels saved successfully.")
