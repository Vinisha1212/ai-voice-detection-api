import base64

mp3_file_path = "sample.mp3"  

with open(mp3_file_path, "rb") as audio_file:
    audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

print(audio_base64)
