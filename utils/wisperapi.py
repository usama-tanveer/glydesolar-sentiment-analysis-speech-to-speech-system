from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class WhisperAPI:
    def __init__(self):
        """
        Initialize the WhisperAPI client.
        """
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file using OpenAI Whisper API.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            str: Transcription of the audio.
        """
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                return transcription
        except Exception as e:
            print(f"Error during transcription: {e}")
            return "Error during transcription"