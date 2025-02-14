import os
import tempfile
import subprocess

from flask import Flask, request, jsonify
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech
from googleapiclient.discovery import build

app = Flask(__name__)

# We assume Cloud Run will provide credentials via Workload Identity or
# we can specify a service account JSON via an env var if needed for local dev.
SERVICE_ACCOUNT_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json")
credentials = None
if os.path.exists(SERVICE_ACCOUNT_PATH):
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)

drive_service = build('drive', 'v3', credentials=credentials) if credentials else None
speech_client = speech.SpeechClient(credentials=credentials) if credentials else speech.SpeechClient()

@app.route("/", methods=["GET"])
def index():
    return "Cloud Run video transcriber is up. Send a POST request to /transcribe."

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON data found"}), 400
    
    drive_link = data.get("drive_link")
    file_id = data.get("file_id")

    # If drive_link provided, extract file_id from link
    if drive_link and not file_id:
        try:
            parts = drive_link.split('/file/d/')
            if len(parts) > 1:
                file_part = parts[1]
                file_id = file_part.split('/')[0]
            else:
                return jsonify({"error": "Could not extract file ID from drive_link"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    if not file_id:
        return jsonify({"error": "No file_id provided or found."}), 400

    try:
        # 1) Download video from Google Drive to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            # Use the drive API to download
            downloader = drive_service.files().get_media(fileId=file_id)
            with open(temp_video.name, 'wb') as out_file:
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        pass  # optionally track progress here
            
            temp_audio_path = temp_video.name.replace(".mp4", ".wav")

        # 2) Convert video to audio using ffmpeg
        command = [
            "ffmpeg",
            "-i", temp_video.name,
            "-ac", "1",           # mono channel
            "-ar", "16000",       # sample rate
            "-y",                 # overwrite
            temp_audio_path
        ]
        subprocess.run(command, check=True)

        # 3) Transcribe audio
        with open(temp_audio_path, "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US"
        )
        response = speech_client.recognize(config=config, audio=audio)

        # Cleanup
        if os.path.exists(temp_video.name):
            os.remove(temp_video.name)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        # 4) Build transcript string
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript

        return jsonify({"transcript": transcript}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
