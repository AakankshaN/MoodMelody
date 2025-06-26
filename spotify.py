import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random

load_dotenv()

SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')

def initialize_spotify():
    if SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET:
        try:
            client = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=SPOTIPY_CLIENT_ID,
                client_secret=SPOTIPY_CLIENT_SECRET
            ))
            print("Spotify API initialized successfully.")
            return client
        except Exception as e:
            print(f"Error initializing Spotify API: {e}")
    else:
        print("Client ID/Secret missing in environment variables.")
    return None

sp = initialize_spotify()

EMOTION_GENRE_MAPPING = {
    "Happy": ["pop", "dance pop", "feel good"],
    "Sad": ["acoustic", "piano", "sad"],
    "Angry": ["metal", "hard rock", "punk"],
    "Surprise": ["edm", "electronic", "techno"],
    "Fear": ["dark ambient", "cinematic", "soundtrack"],
    "Disgust": ["jazz", "blues", "lo-fi"],
    "Neutral": ["chill", "lo-fi", "instrumental"]
}

def get_songs_by_emotion(emotion: str, limit: int = 5) -> list:
    if sp is None:
        print("Spotify API not initialized.")
        return []

    genre_options = EMOTION_GENRE_MAPPING.get(emotion.capitalize(), ["pop"])
    chosen_genre = random.choice(genre_options)
    query = f"{chosen_genre}"

    print(f" Searching for '{emotion}' mood using genre: '{chosen_genre}'")

    try:
        results = sp.search(q=query, type='track', limit=min(limit * 5, 50))
        items = results.get('tracks', {}).get('items', [])

        songs = []
        for item in items:
            song = {
                "title": item['name'],
                "artist": item['artists'][0]['name'] if item['artists'] else "Unknown",
                "release_date": item['album'].get('release_date', 'N/A'),
            }
            songs.append(song)
            if len(songs) >= limit:
                break

        if not songs:
            print(f"No songs found for emotion '{emotion}' with genre '{chosen_genre}'.")
        return songs

    except spotipy.exceptions.SpotifyException as se:
        print(f" Spotify API error: {se}")
        return []
    except Exception as e:
        print(f" Unexpected error: {e}")
        return []

# use when testing this api independently
# if __name__ == "__main__":
#     test_emotions = ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral", "Random"]

#     for emotion in test_emotions:
#         print(f"\nEmotion: {emotion}")
#         songs = get_songs_by_emotion(emotion, limit=3)
#         if songs:
#             for i, song in enumerate(songs, 1):
#                 print(f"{i}. {song['title']} by {song['artist']} (Released: {song['release_date']})")
#         else:
#             print("No recommendations found.")
#         print("-" * 40)
