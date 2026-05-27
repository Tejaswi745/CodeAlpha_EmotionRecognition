import os

dataset_path = "C:/Users/USER/OneDrive/Desktop/emotion_project/dataset"

files = []

for actor in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor)
    
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            file_path = os.path.join(actor_path, file)
            
            parts = file.split("-")
            emotion = parts[2]
            
            files.append((file_path, emotion))

print(files[:5])
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised"
}

data = []

for file_path, emotion in files:
    data.append((file_path, emotion_map[emotion]))

print(data[:5])
import librosa
import numpy as np

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    
    return mfcc_scaled

features = []
labels = []

for file_path, emotion in data:
    try:
        feature = extract_features(file_path)
        features.append(feature)
        labels.append(emotion)
    except:
        pass

print("Features shape:", np.array(features).shape)
print("Labels count:", len(labels))