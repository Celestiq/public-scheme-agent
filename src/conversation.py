from sarvamai import SarvamAI
import base64
import os
from dotenv import load_dotenv
load_dotenv()

class ConvoAI:
    def __init__(self):
        self.client = SarvamAI(
            api_subscription_key=os.getenv("SARVAMAI_API_KEY"),
        )

    def transcribe_audio(self, audio_file_path: str):
        with open(audio_file_path, "rb") as audio:
            response = self.client.speech_to_text.transcribe(
                file=audio
            )
        self.source_language = response.language_code
        print(response.language_code)
        return response.transcript, response.language_code

    def translate_text(self, text: str):
        if self.source_language != "en-IN":
            translation = self.client.text.translate(
                input=text,
                source_language_code=self.source_language,
                target_language_code="en-IN"
            )
            return translation.translated_text
        else:
            return text

    def convert_text_to_source_language(self, text: str):
        if self.source_language != "en-IN":
            translation = self.client.text.translate(
                input=text,
                source_language_code="en-IN",
                target_language_code=self.source_language
            )
            return translation.translated_text
        else:
            return text

    def text_to_speech(self, text: str):
        audio = self.client.text_to_speech.convert(
            text=text,
            target_language_code=self.source_language,
        )
        base64_string = audio.audios[0]
        wav_data = base64.b64decode(base64_string)

        return wav_data