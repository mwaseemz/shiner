import os
import tempfile
import subprocess
import logging
import io

from flask import Flask, request, jsonify
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import google.auth  # For default credentials
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load credentials: use service account file if present; otherwise, use default credentials.
SERVICE_ACCOUNT_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json")
if os.path.exists(SERVICE_ACCOUNT_PATH):
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
else:
    credentials, _ = google.auth.default()

# Initialize APIs.
drive_service = build('drive', 'v3', credentials=credentials)
speech_client = speech.SpeechClient(credentials=credentials)
storage_client = storage.Client(credentials=credentials)

# Set your GCS bucket name. Either set this environment variable in Cloud Run or replace the default.
GCS_BUCKET = os.getenv("GCS_BUCKET", "new_bucket_make")  # <-- Replace with your bucket name if not using env variable.

@app.route("/", methods=["GET"])
def index():
    return "Cloud Run video transcriber is up. Send a POST request to /transcribe."

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        data = request.get_json(silent=True)
        if not data:
            logging.error("No JSON data received.")
            return jsonify({"error": "No JSON data found"}), 400

        drive_link = data.get("drive_link")
        file_id = data.get("file_id")

        # Extract file_id from drive_link if necessary.
        if drive_link and not file_id:
            try:
                parts = drive_link.split('/file/d/')
                if len(parts) > 1:
                    file_part = parts[1]
                    file_id = file_part.split('/')[0]
                else:
                    logging.error("Could not extract file ID from drive_link.")
                    return jsonify({"error": "Could not extract file ID from drive_link"}), 400
            except Exception as e:
                logging.exception("Error parsing drive_link")
                return jsonify({"error": str(e)}), 400

        if not file_id:
            logging.error("No file_id provided or found.")
            return jsonify({"error": "No file_id provided or found."}), 400

        # Download the video from Google Drive.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video_path = temp_video.name
        logging.info("Starting file download from Drive...")
        request_drive = drive_service.files().get_media(fileId=file_id)
        with io.FileIO(temp_video_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request_drive)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logging.info(f"Download progress: {int(status.progress() * 100)}%")
        logging.info("File download complete.")

        # Convert the video to audio using ffmpeg.
        temp_audio_path = temp_video_path.replace(".mp4", ".wav")
        logging.info("Starting ffmpeg conversion...")
        command = [
            "ffmpeg",
            "-i", temp_video_path,
            "-ac", "1",           # mono channel
            "-ar", "16000",       # sample rate
            "-y",                 # overwrite output file
            temp_audio_path
        ]
        subprocess.run(command, check=True)
        logging.info("ffmpeg conversion complete.")

        # Check the size of the audio file.
        audio_size = os.path.getsize(temp_audio_path)
        logging.info(f"Audio file size: {audio_size} bytes")

        # Prepare Speech-to-Text configuration.
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US"
        )
        
        # For audio larger than 10 MB, use asynchronous recognition.
        if audio_size > 10 * 1024 * 1024:  # 10 MB in bytes.
            # Upload the audio file to GCS.
            bucket = storage_client.bucket(GCS_BUCKET)
            blob_name = os.path.basename(temp_audio_path)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(temp_audio_path)
            gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
            logging.info(f"Uploaded audio to {gcs_uri}")

            # Use asynchronous (long-running) transcription.
            audio = speech.RecognitionAudio(uri=gcs_uri)
            operation = speech_client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=300)  # Adjust timeout as needed.
            # Optionally, delete the blob after processing.
            blob.delete()
        else:
            # Use synchronous transcription.
            with open(temp_audio_path, "rb") as audio_file:
                content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)
            response = speech_client.recognize(config=config, audio=audio)
        
        logging.info("Transcription complete.")

        # Clean up temporary files.
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        # Build the transcript.
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript

        return jsonify({"transcript": transcript}), 200

    except Exception as e:
        logging.exception("Unhandled exception in /transcribe")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

