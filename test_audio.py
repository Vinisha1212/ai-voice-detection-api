from audio_features import extract_audio_features

features = extract_audio_features("sample.mp3")
print(len(features))
print(features)
