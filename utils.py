import librosa
import time
import torch
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

def create_feature_array(tracks, f_type, sr=22050, hl=1024):
    print(f"Creating array for {f_type}")
    start = time.time()
    track_ids = []
    failed_track_ids = []
    f_shape = {"cqt": (84, 640), "stft": (513, 640)}
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
                single_arr = np.abs(librosa.cqt(waveform, sr=sr, hop_length=hl, bins_per_octave=12, n_bins=7*12, res_type="polyphase"))
            if f_type == "stft":
                single_arr = np.abs(librosa.stft(waveform, hop_length=hl, n_fft=1024))
            if single_arr.shape[1] >= 640:
                single_arr = single_arr[:, :640]
            else:
                failed_track_ids.append(track_id)
                continue
            track_ids.append(track_id)
            feature_arr = np.append(feature_arr, [single_arr], axis=0)
        else:
            failed_track_ids.append(track_id)
            continue
    
    return track_ids, feature_arr, failed_track_ids

def load_features(f_type, astorch=True):
    types = ["cqt", "chroma_cqt", "chroma_cens", "stft", "linear_stft", "mel_stft", "chroma_stft"]
    if f_type not in types:
        print(f"{f_type} is not a valid f_")
    else:
        features_dir = f"data/features/{f_type}"
        test_set = np.load(f"{features_dir}/test_{f_type}.npy")
        validate_set = np.load(f"{features_dir}/validate_{f_type}.npy")
        feature_shape = test_set[0].shape
        train_set = np.empty((0, feature_shape[0], feature_shape[1]))
        n_chunks = 8
        for i in range(n_chunks):
            chunk = np.load(f"{features_dir}/train_{f_type}_{i}.npy")
            train_set = np.append(train_set, chunk, axis=0)
        
        return torch.from_numpy(train_set).float(), torch.from_numpy(validate_set).float(), torch.from_numpy(test_set).float()

def load_ids():
    ids_dir = "data/features/ids"
    test_ids = np.load(f"{ids_dir}/test_ids.npy")
    validate_ids = np.load(f"{ids_dir}/validate_ids.npy")
    train_ids = np.array([])
    n_chunks = 8
    for i in range(n_chunks):
        chunk = np.load(f"{ids_dir}/train_ids_{i}.npy")
        train_ids = np.append(train_ids, chunk)
    
    return train_ids.astype(int), validate_ids.astype(int), test_ids.astype(int)
