import librosa
import time
import pandas as pd
import numpy as np


METADATA_DIR = "data/fma_metadata"
MP3_DIR = "data/fma_small"

def load_csv(name, tracks_size="small"):
    if name == "tracks":
        df = pd.read_csv(f"{METADATA_DIR}/tracks.csv", index_col=0, header=[0, 1])
        df = df[df["set", "subset"] == tracks_size]
    elif name == "genre":
        df = pd.read_csv(f"{METADATA_DIR}genres.csv")

    return df

def load_waveform(track_id, sr=22050):
    id_str = f"{track_id:06}"
    try:
        waveform, _ = librosa.load(f"{MP3_DIR}/{id_str[0:3]}/{id_str}.mp3", mono=True, sr=sr, res_type="polyphase")
    except Exception as e:
        print(f"Unable to load track with id: {id_str}. Error: {e}")
        waveform = np.array([])

    return waveform, sr

def time_since(start):
    now = time.time()
    diff = now - start
    mins = int(diff / 60)
    secs = round(diff - mins * 60, 1)
    return f"{mins}min {secs}s"

sr = 22050
def create_feature_array(tracks, f_type, sr=sr):
    print(f"Creating array for {f_type}")
    start = time.time()
    track_ids = []
    failed_track_ids = []
    f_shape = {"cqt": (84, 320)}
    feature_arr = np.empty((0, f_shape[f_type][0], f_shape[f_type][1]))

    count = 0
    for track in tracks.iterrows():
        track_id = track[0]
        waveform, sr = load_waveform(track_id, sr=sr)

        count += 1
        if count % 100 == 0:
            print(f"{time_since(start)}, Processing {count} out of {len(tracks)}")

        if waveform.any():
            if f_type == "cqt":
                single_arr = np.abs(librosa.cqt(waveform, sr=sr, hop_length=2048, bins_per_octave=12, n_bins=7*12, res_type="polyphase"))
                single_arr = single_arr[:, :320]
            track_ids.append(track_id)
            feature_arr = np.append(feature_arr, [single_arr], axis=0)
        else:
            failed_track_ids.append(track_id)
            continue
    
    return track_ids, feature_arr, failed_track_ids