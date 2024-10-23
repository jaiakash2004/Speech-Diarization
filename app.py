from flask import Flask, request, render_template, send_file
import whisper
import torch
import os
import numpy as np
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import wave
import contextlib
import datetime
import subprocess

app = Flask(__name__)

# Load the Whisper model and embedding model once on startup
model_size = 'small'
model = whisper.load_model(model_size)

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu")
)

audio = Audio()

# Create directories for uploads and audio clips if they don't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")
if not os.path.exists("audio_clips"):
    os.makedirs("audio_clips")

# Helper function for audio processing
def segment_embedding(segment, path, duration):
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    return embedding_model(waveform[None])

def diarize_audio(path, num_speakers=2):
    if path[-3:] != 'wav':
        subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
        path = 'audio.wav'

    result = model.transcribe(path)
    segments = result["segments"]

    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, path, duration)

    embeddings = np.nan_to_num(embeddings)

    # Perform clustering
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_

    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    # Create the transcript file with speaker labels
    transcript_path = "transcript.txt"
    with open(transcript_path, "w") as f:
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + str(datetime.timedelta(seconds=round(segment["start"]))) + '\n')
            f.write(segment["text"][1:] + ' ')

    return transcript_path, segments

def create_audio_clip(segment, original_file_path):
    start = segment["start"]
    end = segment["end"]
    speaker = segment["speaker"].replace(" ", "_")
    clip_name = f"static/audio_clips/{speaker}_clip_{int(start)}_{int(end)}.mp3"  # Ensure unique file name per segment
    subprocess.call(['ffmpeg', '-i', original_file_path, '-ss', str(start), '-to', str(end), clip_name, '-y'])
    return clip_name



@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400
        
        # Save the file temporarily
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        # Process the file for diarization
        transcript_path, segments = diarize_audio(file_path)

        # Prepare transcript data for rendering
        transcript = {}
        for segment in segments:
            speaker = segment["speaker"]
            if speaker not in transcript:
                transcript[speaker] = []
            audio_clip_path = create_audio_clip(segment, file_path)  # Function to create audio clip for each segment
            transcript[speaker].append({
                "timestamp": str(datetime.timedelta(seconds=round(segment["start"]))),
                "text": segment["text"][1:],
                "audio_url": audio_clip_path
            })

        return render_template("index.html", transcript=transcript)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
