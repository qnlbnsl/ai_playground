import argparse
import io
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import tempfile
import os
from functions.transcribe import transcribe
temp_dir = "C:\\Users\\kunal\\Projects\\ai_playground\\Temp"  # tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "temp.wav")


def check_stop_word(predicted_text: str, stop_word: str) -> bool:
    import re
    pattern = re.compile('[\W_]+', re.UNICODE)
    return pattern.sub('', predicted_text).lower() == stop_word


def async_transcribe(
        model="base",
        language="english",
        mic_energy=500,
        pause_duration=0.5,
        mic_dynamic_energy="False",
        stop_word="stop",
        sample_rate=16000,
):
    # there are no english models for large
    if model != "large" and language == 'english':
        model = model + ".en"
    # audio_model = whisper.load_model("base")
    # load the speech recognizer with CLI settings
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = mic_energy
    recognizer.pause_threshold = pause_duration
    recognizer.dynamic_energy_threshold = mic_dynamic_energy
    print(save_path)
    with sr.Microphone(sample_rate=sample_rate) as source:
        print("Let's get the talking going!")
        while True:
            # record audio stream into wav
            audio = recognizer.listen(source)
            data = io.BytesIO(audio.get_wav_data())
            audio_clip = AudioSegment.from_file(data, format='wav')
            audio_clip.export(save_path, format="wav")
            predicted_text = recognizer.recognize_whisper(audio_data=data, model=model, language=language)
            # predicted_text = transcribe(save_path)
            # predicted_text = result["text"]
            print("Text: " + predicted_text)

            if check_stop_word(predicted_text, stop_word):
                break
