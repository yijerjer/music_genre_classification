import utils
import torch
import random
import time
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim


class DataLoader():
    def __init__(self, feature_type, cuda=False):
        self.feature_type = feature_type
        self.cuda = cuda
        self.tracks = utils.load_csv("tracks")["track"]
        self.genres = list(set(self.tracks["genre_top"]))

        self.train_ids, self.validate_ids, self.test_ids = utils.load_ids()
        self.train_set, self.validate_set, self.test_set = utils.load_features(feature_type)
        if self.cuda:
            self.train_set = self.train_set.cuda()
            self.validate_set = self.validate_set.cuda()
            self.test_set = self.test_set.cuda()
        
    def random_training_example(self, batch_size=4):
        max_int = len(self.train_ids) - 1
        random_ints = [random.randint(0, max_int) for x in range(batch_size)]
        ids = self.train_ids[random_ints]
        genres = self.tracks.loc[ids]["genre_top"].tolist()
        genre_tensor = torch.tensor([self.genres.index(genre) for genre in genres])
        feature_tensor = self.train_set[random_ints][:, :, :]

        if self.cuda:
            feature_tensor = feature_tensor.cuda()
            genre_tensor = genre_tensor.cuda()

        return feature_tensor, genres, genre_tensor


class ModelUtils():
    def __init__(self, model, criterion, optimizer, data_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dl = data_loader

    def category_from_output(self, categories, output_tensor):
        _, top_i = output_tensor.topk(1)
        category_i = top_i.flatten()
        return [categories[i] for i in category_i], category_i

    def train_single(self, input_tensor, genre_tensor):
        self.optimizer.zero_grad()
        output = self.model(input_tensor)
        loss = self.criterion(output, genre_tensor)
        loss.backward()
        self.optimizer.step()

        return output, loss.item()

    def train(self, n_epochs=50, n_iters=10000, save_pth_every=0):
        current_loss = 0
        self.all_losses = []
        print_every = 2500
        plot_every = 500

        if save_pth_every > 0:
            if not os.path.isdir("pths"):
                os.makedirs("pths")
            if not os.path.isdir(f"pths/{self.model.name}_{self.dl.feature_type}"):
                os.makedirs(f"pths/{self.model.name}_{self.dl.feature_type}")

        start = time.time()
        for epoch in range(n_epochs):
            for i in range(1, n_iters + 1):
                input_tensor, genre, genre_tensor = self.dl.random_training_example()
                output, loss = self.train_single(input_tensor, genre_tensor)
                current_loss += loss

                if i % print_every == 0:
                    print(f"{utils.time_since(start)}, epoch {epoch}, iter {i}, loss: {current_loss}, Predict: {self.category_from_output(self.dl.genres, output)}, Actual: {genre}")
                if i % plot_every == 0:
                    self.all_losses.append(current_loss)
                    current_loss = 0

            if save_pth_every > 0 and epoch % save_pth_every == 0:
                torch.save(self.model.state_dict(), f"pths/{self.model.name}_{self.dl.feature_type}/epoch_{epoch}.pth")
                print(f"Saved model state as {self.model.name}_{self.dl.feature_type}/epoch_{epoch}.pth")
        
        np.save(f"{self.model.name}_{self.dl.feature_type}_losses.npy")
            
    def create_confusion_mat(self, which_set="test"):
        if which_set == "test":
            data_set = self.dl.test_set
            data_ids = self.dl.test_ids
        elif which_set == "validate":
            data_set = self.dl.validate_set
            data_ids = self.dl.validate_ids
        elif which_set == "train":
            data_set = self.dl.train_set
            data_ids = self.dl.train_ids

        self.confusion_mat = torch.zeros(len(self.dl.genres), len(self.dl.genres))

        for i in range(int(len(data_set) / 4)):
            idxs = [j for j in range(i * 4, (i + 1) * 4)]
            data_tensor = data_set[idxs]
            track_ids = data_ids[idxs]
            actual_genres = self.dl.tracks.loc[track_ids]["genre_top"]
            actual_i = [self.dl.genres.index(genre) for genre in actual_genres]
            output = self.model(data_tensor)
            _, predicted_i = self.category_from_output(self.dl.genres, output)
            for j in range(4):
                self.confusion_mat[actual_i[j]][predicted_i[j]] += 1
        
        return self.confusion_mat
    
    def get_accuracy(self, which_set="test"):
        try:
            self.confusion_mat
        except NameError:
            self.confusion_mat = self.create_confusion_mat(self.dl, which_set=which_set)

        return self.confusion_mat.diag().sum() / self.confusion_mat.sum()



class ChromaCNN(nn.Module):
    def __init__(self, regu_type=""):
        super(ChromaCNN, self).__init__()
        if regu_type in ["", "batchnorm", "dropout"]:
            self.name = f"chroma_cnn_{regu_type}"
            self.regu_type = regu_type
        else:
            raise ValueError(f"{regu_type} regu_type is invalid")

        self.conv1 = nn.Conv1d(12, 12, 5)
        self.conv2 = nn.Conv2d(1, 1, 5)
        self.conv3 = nn.Conv1d(4, 4, 5)
        self.maxpool1 = nn.MaxPool1d(2)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(4 * 36, 80)
        self.fc2 = nn.Linear(80, 36)
        self.fc3 = nn.Linear(36, 8)

        if self.regu_type == "batchnorm":
            self.bn1 = nn.Linear(8)
        elif self.regu_type == "dropout":
            self.dropout = nn.Dropout()
    
    def forward(self, input):
        output = self.maxpool1(functional.relu(self.conv1(input)))
        output = output[:, None, :, :]
        output = self.maxpool2(functional.relu(self.conv2(output)))
        output = torch.squeeze(output, dim=1)
        output = self.maxpool1(functional.relu(self.conv3(output)))

        output = output.view(-1, 4 * 36)
        output = functional.relu(self.fc1(output))
        output = functional.relu(self.fc2(output))
        output = self.fc3(output)
        if self.regu_type == "batchnorm":
            output = self.bn1(output)
        elif self.regu_type == "dropout":
            output = self.dropout(output)

        return output

