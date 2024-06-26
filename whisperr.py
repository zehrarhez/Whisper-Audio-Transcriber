import whisper
import gradio as gr
import os

# Whisper modelini yükle
model = whisper.load_model("base")

def transcribe(audio):
    if isinstance(audio, str):  # Handle file path case
        file_path = audio
    else:  # Handle file object case
        file_path = audio.name
    try:
        # Ses dosyasını yükle ve transkripte et
        result = model.transcribe(file_path)
        text = result['text']
        
        # .srt formatında altyazı dosyası oluştur
        srt_content = ''
        for i, segment in enumerate(result['segments']):
            start = segment['start']
            end = segment['end']
            start_formatted = f"{int(start // 3600):02}:{int(start % 3600 // 60):02}:{int(start % 60):02},{int(start % 1 * 1000):03}"
            end_formatted = f"{int(end // 3600):02}:{int(end % 3600 // 60):02}:{int(end % 60):02},{int(end % 1 * 1000):03}"
            srt_content += f"{i + 1}\n{start_formatted} --> {end_formatted}\n{segment['text']}\n\n"

        # Metin ve .srt dosyasını kaydet
        with open("transcript.txt", "w", encoding='utf-8') as txt_file:
            txt_file.write(text)
            
        with open("transcript.srt", "w", encoding='utf-8') as srt_file:
            srt_file.write(srt_content)
        
        # Dosya yollarını döndür
        return text, "transcript.txt", "transcript.srt"
    except Exception as e:
        print(audio)
        return str(e), None, None

# Gradio arayüzünü oluştur
iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=[
        gr.Textbox(label="Transcript"),
        gr.File(label="Download .txt"),
        gr.File(label="Download .srt")
    ],
    title="Whisper Transcription",
    description="Upload an audio file to transcribe it to text using OpenAI's Whisper model.",
)

# Arayüzü çalıştır
iface.launch(share=True)
