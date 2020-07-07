import time
import utils
import torch
import numpy as np
import torch.utils.data as data


class FeatureDataset(data.Dataset):
    def __init__(self, feature_type, data_type):
        self.feature_type = feature_type
        self.data_type = data_type
        data_dir = f"data/features/{feature_type}/"
        ids_dir = f"data/features/ids/"

        tracks = utils.load_csv("tracks")["track"]
        self.genres = np.unique(tracks["genre_top"].to_numpy()).tolist()

        if data_type in ["test", "validate"]:
            filename = f"{data_type}_{feature_type}.npy"
            self.npy = np.load(f"{data_dir}{filename}", mmap_mode="r")
            self.ids = np.load(f"{ids_dir}{data_type}_ids.npy")
            self.genre_data = tracks["genre_top"].loc[self.ids].tolist()

        elif data_type == "train":
            self.npys = []
            self.ids = []
            self.genre_datas = []
            for i in range(8):
                filename = f"{data_type}_{feature_type}_{i}.npy"
                ids = np.load(f"{ids_dir}{data_type}_ids_{i}.npy")
                npy = np.load(f"{data_dir}{filename}", mmap_mode="r")
                genre_data = tracks["genre_top"].loc[ids].tolist()

                self.npys.append(npy)
                self.ids.append(ids)
                self.genre_datas.append(genre_data)

        del tracks

    def __len__(self):
        if self.data_type in ["test", "validate"]:
            return len(self.npy)
        elif self.data_type == "train":
            return sum([len(npy) for npy in self.npys])

    def __getitem__(self, idx):
        if self.data_type in ["test", "validate"]:
            return (
                np.copy(self.npy[idx]),
                self.genres.index(self.genre_data[idx]),
                self.genre_data[idx],
                self.ids[idx]
            )
        elif self.data_type == "train":
            main_idx = None
            sub_idx = None
            lengths = [len(npy) for npy in self.npys]
            current_length = 0
            for i, length in enumerate(lengths):
                if idx < (current_length + length):
                    main_idx = i
                    sub_idx = idx - current_length
                    break
                current_length += length
        
        return (
            np.copy(self.npys[main_idx][sub_idx]),
            self.genres.index(self.genre_datas[main_idx][sub_idx]),
            self.genre_datas[main_idx][sub_idx],
            self.ids[main_idx][sub_idx]
        )