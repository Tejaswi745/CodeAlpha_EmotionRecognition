import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels
emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def predict_live():
    print("Speak now...")

    duration = 3  # seconds
    sample_rate = 22050

    # Record audio
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()

    # Flatten audio
    audio = audio.flatten()

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = mfcc.reshape(1, -1)

    # Predict emotion
    prediction = model.predict(mfcc)

    # Print result
    print("Predicted Emotion:", emotions[np.argmax(prediction)])
    print("Prediction:", prediction)
    print("Max index:", np.argmax(prediction))


# Run the function
if __name__ == "__main__":
    predict_live()