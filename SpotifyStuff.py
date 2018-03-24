import spotipy
import pprint
from spotipy import oauth2
import requests
from info import CLIENT_ID, CLIENT_SECRET
from spotipy.oauth2 import SpotifyClientCredentials


client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

username = 'syke117'
# username = '1116650208'
jazz10000 = '2hJBSU1hLZUVDI9WT37pmI'
rock10000 = '3PO7Cfl1eoMcjhnq1IwByd'
hiphop10000 = '4ga5O625dPrhIp7G6i71y8'
metal10000 = '0s883mgjcHaWOv4TF4YI7C'
classical10000 = '5mzuM0jq1oylBxjGUKRb4K'
woop = '1Zyxj9mEn9EI6t0RfYGDTY' # For testing

pp = pprint.PrettyPrinter()

def get_songs_from_playlists():
    playlists = [classical10000]
    playlist_names = ['classical']
    # playlists = [jazz10000]
    for index, playlist in enumerate(playlists):
        print('Doing ' + playlist_names[index] + '...')
        uri = 'spotify:user:' + username + ':playlist:' + playlist
        results = sp.user_playlist(username, playlist, fields='tracks')
        # print(results['tracks'])
        with open(playlist_names[index] + '_ids.txt', 'w') as f:
            for track in results['tracks']['items']:
                # print(track['track']['id'])
                f.write(track['track']['id'] + '\n')
            results = sp.next(results['tracks'])
            while (results['next']):
                for track in results['items']:
                    # pprint(track['track']['id'])
                    f.write(track['track']['id'] + '\n')
                # print(results['next'])
                try:
                    results = sp.next(results)

                except:
                    break
    return

######
#
# This function will scan through a list of spotify IDs and return song features in groups of 50 at a time.
# Then it will strip away all the excess info and write the audio features to a file
#
######
def get_track_analysis(file_num):
    if file_num == 0:
        filename = "classical_ids.txt"
    elif file_num == 1:
        filename = "rock_ids.txt"
    elif file_num == 2:
        filename = "hip_hop_ids.txt"
    elif file_num == 3:
        filename = "classical_ids.txt"
    stepsize = 50

    file = open(filename, "r").readlines()
    write_file = open("analysis_"+filename, "w")

    file_len = sum(1 for line in file)
    stripped = [x.strip() for x in file]
    # print(sp.audio_features(stripped[0]))
    for pos in range(0, file_len, stepsize):
        WAKANDA_FOREVER = sp.audio_features(stripped[pos:pos+stepsize])
        for dictionary in WAKANDA_FOREVER:
            if dictionary is not None:
                write_file.write(str(dictionary["danceability"]) + "," +
                                 str(dictionary["energy"]) + "," +
                                 str(dictionary["key"]) + "," +
                                 str(dictionary["loudness"]) + "," +
                                 str(dictionary["mode"]) + "," +
                                 str(dictionary["speechiness"]) + "," +
                                 str(dictionary["acousticness"]) + "," +
                                 str(dictionary["instrumentalness"]) + "," +
                                 str(dictionary["liveness"]) + "," +
                                 str(dictionary["valence"]) + "," +
                                 str(dictionary["tempo"]) + "," +
                                 str(dictionary["time_signature"]) + "," +
                                 str(file_num) + '\n')

    return

# get_songs_from_playlists()
for i in range(4):
    print('doing ' + str(i))
    get_track_analysis(i)