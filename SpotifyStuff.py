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

woop = '1Zyxj9mEn9EI6t0RfYGDTY' # For testing

playlists = [jazz10000, rock10000, hiphop10000]
playlist_names = ['jazz', 'rock', 'hip_hop']
# playlists = [jazz10000]
for index, playlist in enumerate(playlists):
    print('Doing ' + playlist_names[index] + '...')
    uri = 'spotify:user:' + username + ':playlist:' + playlist
    results = sp.user_playlist(username, playlist, fields='tracks')
    # print(results['tracks'])
    with open(playlist_names[index] + '_ids.txt', 'w') as f:
        for track in results['tracks']['items']:
            # print(track['track']['id'])
            f.write(track['track']['uri'] + '\n')
        results = sp.next(results['tracks'])
        while (results['next']):
            for track in results['items']:
                # pprint(track['track']['id'])
                f.write(track['track']['uri'] + '\n')
            # print(results['next'])
            try:
                results = sp.next(results)

            except:
                break

