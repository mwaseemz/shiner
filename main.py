import os
import tempfile
import subprocess
import logging
import io
import threading
import time
import requests

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

# Set your GCS bucket name if using asynchronous transcription.
GCS_BUCKET = os.getenv("GCS_BUCKET", "new_bucket_make")  # Replace with your bucket name

# Set the webhook URL to which the transcript will be sent.
WEBHOOK_URL = "https://hook.us2.make.com/hbc5ver5l4bquuf8fm0jrals3faw142d"

def process_transcription(data):
    try:
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
                    return
            except Exception as e:
                logging.exception("Error parsing drive_link")
                return

        if not file_id:
            logging.error("No file_id provided or found.")
            return

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

        # Check audio file size.
        audio_size = os.path.getsize(temp_audio_path)
        logging.info(f"Audio file size: {audio_size} bytes")

        # Prepare Speech-to-Text configuration.
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US"
        )

        # Transcribe audio.
        if audio_size > 10 * 1024 * 1024:  # Larger than 10 MB.
            logging.info("Audio file exceeds 10MB; using asynchronous transcription.")
            # Upload audio to GCS and use asynchronous transcription.
            bucket = storage_client.bucket(GCS_BUCKET)
            blob_name = os.path.basename(temp_audio_path)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(temp_audio_path)
            gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
            logging.info(f"Uploaded audio to {gcs_uri}")

            audio = speech.RecognitionAudio(uri=gcs_uri)
            operation = speech_client.long_running_recognize(config=config, audio=audio)
            logging.info("Asynchronous transcription operation started.")
            # Poll progress every 30 seconds.
            start_time = time.time()
            while not operation.done():
                elapsed = time.time() - start_time
                logging.info(f"Asynchronous transcription in progress: {int(elapsed)} seconds elapsed...")
                time.sleep(30)
            response = operation.result(timeout=3600)  # Increase timeout as needed.
            logging.info("Asynchronous transcription operation completed.")
            # Optionally, delete the blob.
            blob.delete()
        else:
            logging.info("Using synchronous transcription.")
            # Synchronous transcription.
            with open(temp_audio_path, "rb") as audio_file:
                content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)
            response = speech_client.recognize(config=config, audio=audio)
            logging.info("Synchronous transcription completed.")

        logging.info("Transcription complete. Building transcript...")
        # Build transcript string.
        transcript = ""
        for idx, result in enumerate(response.results):
            part = result.alternatives[0].transcript
            transcript += part
            logging.info(f"Transcript part {idx+1}: {part}")

        logging.info(f"Final transcript length: {len(transcript)} characters")

        # Clean up temporary files.
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        # Send the transcript to the webhook.
        payload = {"transcript": transcript}
        webhook_response = requests.post(WEBHOOK_URL, json=payload)
        logging.info(f"Webhook response: {webhook_response.status_code} - {webhook_response.text}")

    except Exception as e:
        logging.exception("Unhandled exception in background transcription process")
        # Optionally, send error details to the webhook.
        error_payload = {"error": str(e)}
        try:
            requests.post(WEBHOOK_URL, json=error_payload)
        except Exception as ex:
            logging.exception("Failed to send error payload to webhook")

@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    data = request.get_json(silent=True)
    if not data:
        logging.error("No JSON data received.")
        return jsonify({"error": "No JSON data found"}), 400
    # Start the background transcription process.
    threading.Thread(target=process_transcription, args=(data,)).start()
    # Immediately return a success response.
    return jsonify({"status": "accepted", "message": "Transcription processing started."}), 200

@app.route("/", methods=["GET"])
def index():
    return "Cloud Run video transcriber is up. Send a POST request to /transcribe."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
