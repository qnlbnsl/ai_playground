import warnings
from functions.gradio import gradio
from functions.async_whisper import async_transcribe
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")

# Press Ctrl+F8 to toggle the breakpoint.
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # gradio().launch()
    async_transcribe()
