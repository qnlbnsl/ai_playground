import os

import whisper
from pyChatGPT import ChatGPT
import gradio as gr

secret_token = os.getenv('OPENAI_TOKEN')  # Enter your session token here!
model = whisper.load_model("base")
model.device


def transcribe(audio):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    result_text = result.text


    return result_text


def gradio():
    output_1 = gr.Textbox(label="Speech to Text")
    output_2 = gr.Textbox(label="ChatGPT Output")

    return gr.Interface(
        title='OpenAI Whisper and ChatGPT ASR Gradio Web UI',
        fn=transcribe,
        inputs=[
            gr.inputs.Audio(source="microphone", type="filepath")
        ],

        outputs=[
            output_1, output_2
        ],
        live=True)
