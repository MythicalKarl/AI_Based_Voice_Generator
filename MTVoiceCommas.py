import sounddevice as sd
import numpy as np
import threading
import keyboard
from scipy.io import wavfile
import os
import speech_recognition as sr
import elevenlabs
import io
import soundfile as sf
import nltk
import psutil
import concurrent.futures
from pydub import AudioSegment
from language_tool_python import LanguageTool
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize  # Add this import
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
# Initialize LanguageTool
language_tool = LanguageTool('en-US')

elevenlabs.set_api_key("Your api key")

# Set the audio parameters
sample_rate = 44100

# Specify the path to the desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
filename = os.path.join(desktop_path, "output.wav")

# Get the current process
current_process = psutil.Process()

# Set the CPU priority to "high"
current_process.nice(psutil.HIGH_PRIORITY_CLASS)

# Initialize variables
recording = False
audio_data = []

# Callback function for recording audio
def audio_callback(indata, frames, time, status):
    if recording:
        audio_data.append(indata.copy())

# Function to start recording
def start_recording():
    os.system('cls')
    global recording, audio_data
    print("Recording audio...")
    recording = True
    audio_data = []
    with sd.InputStream(callback=audio_callback, channels=2, samplerate=sample_rate):
        while recording:  # Continue recording while recording is True
            sd.sleep(100)  # Sleep for a short duration to check for key release
    print("Finished recording.")

# Function to stop recording
def stop_recording():
    global recording
    recording = False

def add_commas(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Iterate through sentences and add commas based on POS tagging
    corrected_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)

        # Check for specific patterns and add commas accordingly
        # Example: If a sentence starts with an introductory phrase, add a comma
        if tagged_words:
            if tagged_words[0][1] in ["RB", "DT", "IN"]:
                sentence = sentence.replace(tagged_words[0][0], tagged_words[0][0] + ",")

        # Add the sentence to the list of corrected sentences
        corrected_sentences.append(sentence)

    # Combine the corrected sentences back into a single text
    corrected_text = ' '.join(corrected_sentences)
    
    return corrected_text

# Function to save recorded audio to a file
def save_audio():
    global audio_data
    if audio_data:
        audio_array = np.concatenate(audio_data, axis=0)
        wavfile.write(filename, sample_rate, audio_array)
        print(f"Saved audio to {filename}")


def transcribe_audio():
    recognizer = sr.Recognizer()

    # Convert audio to PCM WAV format
    audio = AudioSegment.from_wav(filename)

    # Raise the volume by a factor (e.g., 1.5 for 150% volume)
    volume_factor = 1.5
    audio = audio.apply_gain(volume_factor * 10)  # Convert factor to dB

    pcm_wav_path = os.path.join(desktop_path, "output_pcm.wav")
    audio.export(pcm_wav_path, format="wav")

    with sr.AudioFile(pcm_wav_path) as source:
        audio = recognizer.record(source)

    voice = elevenlabs.Voice(
       voice_id="2EiwWnXFnvU5JabPnv8n",
       settings=elevenlabs.VoiceSettings(
           stability=0.15,
           similarity_boost=1
       )
    )

    try:
        transcription = recognizer.recognize_google(audio)
        print("Transcription:", transcription)

        # Apply grammar corrections using LanguageTool
        corrected_text = language_tool.correct(transcription)

        # Use PunktSentenceTokenizer for sentence tokenization
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(corrected_text)

        # Punctuate sentences based on content
        punctuated_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()  # Remove leading/trailing spaces

            if not sentence.endswith(('?', '!', '.')):
                # Check if the sentence is a question based on its content
                if sentence.lower().startswith(('who', 'what', 'when', 'where', 'why', 'how', 'are','excuse' )):
                    sentence += '?'  # Add a question mark if not present
                else:
                    sentence += '.'  # Add a period if not a question or statement with punctuation

            punctuated_sentences.append(sentence)

        punctuated_text = ' '.join(punctuated_sentences)

        # Add commas based on grammatical rules
        punctuated_text_with_commas = add_commas(punctuated_text)

        with open(os.path.join(desktop_path, "transcription.txt"), "w") as transcription_file:
            transcription_file.write(punctuated_text_with_commas)
        print("Transcription saved to transcription.txt")

        # Transcribe and generate audio
        with concurrent.futures.ThreadPoolExecutor() as executor:
            transcribe_future = executor.submit(elevenlabs.generate, text=punctuated_text_with_commas, voice=voice)
            transcribed_audio = transcribe_future.result()  # Get the result of the transcription audio

        sd.default.device = 'Virtual Cable (VB-Audio Virtual Cable), Windows DirectSound'
        sd.play(*sf.read(io.BytesIO(transcribed_audio)))
        sd.wait()

        sd.default.device = 'Microphone (Sharkoon RUSH ER30), Windows DirectSound'

    except sr.UnknownValueError:
        print("Google Web Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech Recognition service; {e}")


# Thread for recording, saving, and transcribing
def record_save_transcribe():
    start_recording()
    save_audio()
# Thread for recording, saving, and transcribing
def record_save_transcribe():
    start_recording()
    save_audio()

    # Create and start a thread for transcription
    transcription_thread = threading.Thread(target=transcribe_audio)
    transcription_thread.Daemon = True
    transcription_thread.start()

# ...

# Start recording when the specified key is held down and stop when released
def key_handler(e):
    global recording_thread
    if e.event_type == keyboard.KEY_DOWN and e.name == 'e' and not recording:
        recording_thread = threading.Thread(target=record_save_transcribe)
        recording_thread.start()
    elif e.event_type == keyboard.KEY_UP and e.name == 'e' and recording:
        stop_recording()

# Register the key handler
keyboard.hook(key_handler)

# Main loop
running = True
while running:
    # Check if the user wants to exit the program
    user_input = input("Press 'q' and Enter to quit: ")
    if user_input.lower() == 'q':
        running = False  # Set the flag to exit the loop
