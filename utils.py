import librosa
import time
import os
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
        df = pd.read_csv(f"{METADATA_DIR}/genres.csv", index_col=0)
    elif name == "features":
        df = pd.read_csv(f"{METADATA_DIR}/features.csv", index_col=0, header=[0, 1, 2])

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


def normalise_feature(f_type):
    if not os.path.isdir(f"data/features/norm_{f_type}"):
        os.makedirs(f"data/features/norm_{f_type}")
    
    print("normalising test array")
    test_arrs = np.load(f"data/features/{f_type}/test_{f_type}.npy")
    norm_test_arrs = normalise_array(test_arrs)
    np.save(f"data/features/norm_{f_type}/test_norm_{f_type}.npy", norm_test_arrs)

    print("normalising validate array")

    validate_arrs = np.load(f"data/features/{f_type}/validate_{f_type}.npy")
    norm_validate_arrs = normalise_array(validate_arrs)
    np.save(f"data/features/norm_{f_type}/validate_norm_{f_type}.npy", norm_validate_arrs)

    n_chunks = 8
    for i in range(n_chunks):
        print(f"normalising train array {i}")
        train_arrs = np.load(f"data/features/{f_type}/train_{f_type}_{i}.npy")
        norm_train_arrs = normalise_array(train_arrs)
        np.save(f"data/features/norm_{f_type}/train_norm_{f_type}_{i}.npy", norm_train_arrs)


def normalise_array(arr):
    shape = arr.shape
    arr_T = np.transpose(arr, (0, 2, 1))
    arr_flatten = arr_T.reshape(shape[0] * shape[2], shape[1])
    average = np.average(arr_flatten, axis=0)
    std = np.std(arr_flatten, axis=0)

    norm_arr = (arr - average[:, np.newaxis]) / std[:, np.newaxis]
    return norm_arr

