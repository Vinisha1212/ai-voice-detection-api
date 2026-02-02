import librosa
import numpy as np

def extract_audio_features(audio_path):
    """
    Converts an audio file into numerical features
    """

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # 1️⃣ MFCC (voice fingerprint)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)

    # 2️⃣ Pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0

    # 3️⃣ Zero Crossing Rate (voice sharpness)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # 4️⃣ Energy (loudness)
    energy = np.mean(librosa.feature.rms(y=y))

    # Combine all features into one array
    features = np.hstack([
        mfcc_mean,
        pitch_mean,
        zcr,
        energy
    ])

    return features
