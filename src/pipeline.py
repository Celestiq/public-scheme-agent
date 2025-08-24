from conversation import ConvoAI
from rag import RAGPipeline
from config import KNOWLEDGE_BASE_PATH

convo = ConvoAI()
rag = RAGPipeline(KNOWLEDGE_BASE_PATH)

def get_response_from_audio(audio_file_path: str):
    transcription, _ = convo.transcribe_audio(audio_file_path)
    print("Transcription: ", transcription)
    en_question = convo.translate_text(transcription)
    print("Translated Question: ", en_question)
    en_answer = rag.invoke(en_question)
    print("RAG Answer: ", en_answer)
    source_language_answer = convo.convert_text_to_source_language(en_answer)
    audio_answer = convo.text_to_speech(source_language_answer)

    with open("output.wav", "wb") as f:
        f.write(audio_answer)

    return en_answer

if __name__ == "__main__":
    audio_path = "question.wav"  # Path to your input audio file
    answer = get_response_from_audio(audio_path)
    print("Answer:", answer)