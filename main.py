import warnings
from functions.classifier import *
from functions.transcribe import gradio
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")

# Press Ctrl+F8 to toggle the breakpoint.
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    classify("TEXT")
    gradio().launch()
