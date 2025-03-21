from flask import Flask, render_template, request, Response, redirect, url_for, session
import requests
import os
import time
from dotenv import load_dotenv
#from asyncio.windows_events import NULL
from datetime import datetime
from shutil import register_unpack_format
import sys
from refresh import Refresh
from contextlib import redirect_stderr, redirect_stdout
import re
from urllib.request import urlopen
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
import time
import cv2

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24)

def get_available_camera():
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i  # Return first available camera index
    return None  # No camera found

camera_index = get_available_camera()
camera = cv2.VideoCapture(camera_index)

face_classifier = cv2.CascadeClassifier('/home/hr/Desktop/EMOTION-BASED-MUSIC-AND-VIDEO-RECOMMENDATION-SYSTEM/haarcascade_frontalface_default.xml')
classifier =load_model('/home/hr/Desktop/EMOTION-BASED-MUSIC-AND-VIDEO-RECOMMENDATION-SYSTEM/fer.h5')
emotion_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


@app.route('/')
def main():
    return render_template('newLayout/index.html')


@app.route('/index')
def index():
    return redirect(url_for('app.main'))


@app.route('/about')
def about():
    return render_template('newLayout/about.html')


@app.route('/discography')
def discography():
    return render_template('newLayout/discography.html')


@app.route('/tours')
def tours():
    return render_template('newLayout/tours.html')


@app.route('/videos')
def videos():
    return render_template('newLayout/videos.html')


@app.route('/contact')
def contact():
    return render_template('newLayout/contact.html')


@app.route('/blog')
def blog():
    return render_template('newLayout/blog.html')


@app.route('/blog-details')
def blog_details():
    return render_template('newLayout/blog-details.html')


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global label  # Store detected emotion globally
    camera = cv2.VideoCapture(camera_index)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    prediction = classifier.predict(roi)[0]
                    
                    # Update the detected emotion
                    label = emotion_labels[prediction.argmax()]
                    session['detected_emotion'] = label 
                    
                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(f"Detected Emotion: {label}")

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/recommend', methods=["POST", "GET"])
def recommend():
    choice = request.form.get('choice')  # "music" or "video"
    form_emotion = request.form.get('emotion')  # Emotion from form input
    camera_emotion = session.get('detected_emotion')  # Emotion from camera detection

    # Use camera emotion if available, otherwise fallback to form emotion
    emotion = camera_emotion if camera_emotion else form_emotion

    # Default to Neutral if no valid emotion is detected
    emotion_keywords = {
        "Happy": "trending happy songs",
        "Sad": "sad emotional music",
        "Neutral": "relaxing music",
        "Angry": "rock energetic music",
        "Surprise": "unexpected viral hits",
        "Fear": "dark suspenseful music"
    }

    if not emotion or emotion.capitalize() not in emotion_keywords:
        emotion = "Neutral"  # Default to Neutral if no valid emotion is detected
        choice = "music"  # Default to music if no valid emotion is detected

    search_query = emotion_keywords[emotion.capitalize()]

    print(f"Selected Emotion: {emotion}, Choice: {choice}")

    if choice == "video":
        results = get_youtube_videos(search_query)
        return render_template('newLayout/recommend.html', recommendations=results, type='video')
    else:
        results = get_spotify_tracks(search_query)
        return render_template('newLayout/recommend.html', recommendations=results, type='music')


def get_youtube_videos(query):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        'key': YOUTUBE_API_KEY,
        'q': query + " Hindi OR English OR Punjabi",
        'part': 'snippet',
        'maxResults': 35,
        'type': 'video',
        'videoCategoryId': '10',  # Category ID 10 is for music
        'order': 'viewCount',
        'relevanceLanguage': 'en', 
        'regionCode': 'IN' 
    }
    response = requests.get(search_url, params=search_params)
    results = response.json().get('items', [])

    if not results:
        return []

    return [{'id': video['id']['videoId'], 'title': video['snippet']['title']} for video in results]

def get_spotify_tracks(query):
    auth_url = "https://accounts.spotify.com/api/token"

    auth_response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': SPOTIFY_CLIENT_ID,
        'client_secret': SPOTIFY_CLIENT_SECRET,
    })

    if auth_response.status_code != 200:
        return []

    access_token = auth_response.json().get('access_token')
    if not access_token:
        return []

    search_url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "q": query, 
        "type": "track",
        "limit": 50,
        "market": "IN" 
    }

    response = requests.get(search_url, headers=headers, params=params)

    if response.status_code != 200:
        return []

    response_json = response.json().get('tracks', {}).get('items', [])

    if not response_json:
        return []

    return [{
        'id': track.get('id'),
        'name': track.get('name'),
        'artist': track['artists'][0]['name'] if track.get('artists') else "Unknown Artist",
        'preview_url': track.get('preview_url')
    } for track in response_json]




if __name__ == '__main__':
    app.run(debug=True)