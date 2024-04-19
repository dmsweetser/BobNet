import os
import whisper
import gtts
import pyaudio
import wave
from bob_net import *

class TalkToMe:
    
    def __init__(self):
        self.transcription_model = whisper.load_model("tiny.en")        
        self.bob_net = BobNet()

    def text_to_speech(text, filename):
        if text == "":
            text = "Sorry - something went wrong."
        tts = gtts.gTTS(text=text, lang='en', tld='com.au')
        tts.save(filename + ".mp3")
        os.system("ffplay -autoexit " + filename + ".mp3")

    def record_audio(filename, duration=10):        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        RECORD_SECONDS = duration
        frames = []
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        audio.terminate()
        waveFile = wave.open(filename, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

    def listen(audio_file):
        result = self.transcription_model.transcribe(audio_file, fp16=False, initial_prompt="A user is conversing with a language model named BobNet.")
        return result["text"] if result else None

    def start(self):
        self.text_to_speech("How can I help? Say CHAT WITH BOBNET, TEACH BOBNET or PAUSE BOBNET", "output")
        
        while True:
            self.record_audio("input.wav")
            response = self.listen("input.wav")
            if response:
                question = response.strip()
                if "chat with bobnet" in question.lower():
                    response = self.bob_net.infer(question)
                    self.text_to_speech(response, "output")
                elif "teach bobnet" in question.lower():
                    self.text_to_speech("Go ahead sensei.", "output")
                    self.record_audio("input.wav")
                    lesson = self.listen("input.wav")
                    if lesson:
                        self.bob_net.ingest_single_training_text(lesson)
                        self.text_to_speech("Thanks.", "output")
                elif "pause bobnet" in question.lower():
                    input("Paused - press any key to start again...")
                    continue
                elif question == "":
                    continue

if __name__ == "__main__":
    talk_to_me = TalkToMe()
    talk_to_me.start()
