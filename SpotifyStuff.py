import spotipy
from pprint import pprint
from spotipy import oauth2
import requests
from info import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI
from spotipy.oauth2 import SpotifyClientCredentials



client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

username = 'syke117'
jazz10000 = '2hJBSU1hLZUVDI9WT37pmI'
rock10000 = '3PO7Cfl1eoMcjhnq1IwByd'
hiphop10000 = '4ga5O625dPrhIp7G6i71y8'

# playlists = [jazz10000, rock10000, hiphop10000]
playlists = [jazz10000]
for playlist in playlists:
    uri = 'spotify:user:' + username + ':playlist:' + playlist
    results = sp.user_playlist(username, playlist, fields='tracks')
    for track in results['tracks']['items']:
        print(track['track']['id'])
    # pprint(tracks)

