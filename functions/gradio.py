from datetime import time

from functions.transcribe import transcribe
import gradio as gr
import time
# from functions.classifier import classify


def main(audio, state=""):
    time.sleep(2)
    audio_text = transcribe(audio)
    state += audio_text + "\n "
    # result = classify(audio_text)
    return state, state


def gradio():
    # Define number of outputs
    # state = gr.Textbox(label="Speech to Text")
    # output_2 = gr.Textbox(label="Classifier Output")

    return gr.Interface(
        title='OpenAI Whisper and ChatGPT ASR Gradio Web UI',
        fn=main,
        inputs=[
            gr.Audio(source="microphone", type="filepath", streaming=True), "state"
        ],

        outputs=[
            "textbox",
            "state"
        ],
        live=True)
