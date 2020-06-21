import utils
import torch
import random
import time
import os
import numpy as np
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self, feature_type, cuda=False):
        self.feature_type = feature_type
        self.cuda = cuda
        self.tracks = utils.load_csv("tracks")["track"]
        self.genres = np.unique(self.tracks["genre_top"].to_numpy()).tolist()

        self.train_ids, self.validate_ids, self.test_ids = utils.load_ids()
        self.train_set, self.validate_set, self.test_set = utils.load_features(feature_type)
        if self.cuda:
            self.train_set = self.train_set.cuda()
            self.validate_set = self.validate_set.cuda()
            self.test_set = self.test_set.cuda()

    def random_training_batch(self, batch_size=32):
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

    def train_batch(self, input_tensor, genre_tensor):
        self.optimizer.zero_grad()
        output = self.model(input_tensor)
        loss = self.criterion(output, genre_tensor)
        loss.backward()
        self.optimizer.step()

        return output, loss.item()

    def train(self, n_epochs=70, n_iters=200, batch_size=32, save_pth_every=5):
        current_loss = 0
        self.all_losses = []
        print_every = 5
        plot_every = int(n_iters / 4)

        if not os.path.isdir("losses"):
            os.makedirs("losses")

        if save_pth_every > 0:
            if not os.path.isdir("pths"):
                os.makedirs("pths")
            if not os.path.isdir(f"pths/{self.model.name}_{self.dl.feature_type}"):
                os.makedirs(f"pths/{self.model.name}_{self.dl.feature_type}")

        start = time.time()
        for epoch in range(1, n_epochs + 1):
            for i in range(1, n_iters + 1):
                input_tensor, genre, genre_tensor = self.dl.random_training_batch(batch_size=batch_size)
                output, loss = self.train_batch(input_tensor, genre_tensor)
                current_loss += loss

                if epoch % print_every == 0 and i % 200 == 0:
                    print(f"{utils.time_since(start)}, epoch {epoch}, iter {i}, loss: {current_loss}")
                if i % plot_every == 0:
                    self.all_losses.append(current_loss)
                    current_loss = 0

            if save_pth_every > 0 and epoch % save_pth_every == 0:
                torch.save(self.model.state_dict(), f"pths/{self.model.name}_{self.dl.feature_type}/epoch_{epoch}.pth")
                print(f"Saved model state as {self.model.name}_{self.dl.feature_type}/epoch_{epoch}.pth")
        
        np.save(f"losses/{self.model.name}_{self.dl.feature_type}.npy", self.all_losses)
            
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

        for i in range(int(len(data_set) / 32)):
            idxs = [j for j in range(i * 32, (i + 1) * 32)]
            data_tensor = data_set[idxs]
            track_ids = data_ids[idxs]
            actual_genres = self.dl.tracks.loc[track_ids]["genre_top"]
            actual_i = [self.dl.genres.index(genre) for genre in actual_genres]
            output = self.model(data_tensor)
            _, predicted_i = self.category_from_output(self.dl.genres, output)
            for j in range(32):
                self.confusion_mat[actual_i[j]][predicted_i[j]] += 1
        
        return self.confusion_mat
    
    def get_accuracy(self, which_set="test"):
        try:
            self.confusion_mat
        except NameError:
            self.confusion_mat = self.create_confusion_mat(self.dl, which_set=which_set)

        return self.confusion_mat.diag().sum() / self.confusion_mat.sum()
    
    def get_accuracies_from_pths(self, n_epochs=70, save_pth_every=5, cuda=False):
        try:
            accuracies = (self.train_accuracy, self.validate_accuracy, self.test_accuracy)
            n_vals = n_epochs / save_pth_every
            if len(accuracies[0]) != n_vals or len(accuracies[1]) != n_vals or len(accuracies[2]) != n_vals:
                raise AttributeError
            return accuracies
        except AttributeError:
            PTHS_DIR = f"pths/{self.model.name}_{self.dl.feature_type}"
            n_pths = int(n_epochs / save_pth_every)

            self.train_accuracy = []
            self.validate_accuracy = []
            self.test_accuracy = []
            self.epochs = []

            print(f"Creating confusion matrix for epoch:", end=" ")
            for n in range(1, n_pths + 1):
                epoch_num = n * save_pth_every
                self.epochs.append(epoch_num)

                if cuda:
                    state_dict = torch.load(f"{PTHS_DIR}/epoch_{epoch_num}.pth")
                else:
                    state_dict = torch.load(f"{PTHS_DIR}/epoch_{epoch_num}.pth", map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)
                self.model.eval()

                print(f"{epoch_num},", end=" ")
                self.create_confusion_mat(which_set="test")
                self.test_accuracy.append(self.get_accuracy())
                self.create_confusion_mat(which_set="validate")
                self.validate_accuracy.append(self.get_accuracy())
                self.create_confusion_mat(which_set="train")
                self.train_accuracy.append(self.get_accuracy())
            
            print("Done.")
            return self.train_accuracy, self.validate_accuracy, self.test_accuracy

    def plot_accuracies(self):
        try:
            plt.plot(self.epochs, self.test_accuracy, label="test")
            plt.plot(self.epochs, self.validate_accuracy, label="validate")
            plt.plot(self.epochs, self.train_accuracy, label="train")
            plt.legend()
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.ylim([0, 1])
            plt.xlim([0, max(self.epochs)])

            print("Maximum accuracy and epoch:")
            print(f"Train - Max accuracy of {round(max(self.train_accuracy).item(), 3)} at {self.epochs[self.train_accuracy.index(max(self.train_accuracy))]}")
            print(f"Validate - Max accuracy of {round(max(self.validate_accuracy).item(), 3)} at {self.epochs[self.validate_accuracy.index(max(self.validate_accuracy))]}")
            print(f"Test - Max accuracy of {round(max(self.test_accuracy).item(), 3)} at {self.epochs[self.test_accuracy.index(max(self.test_accuracy))]}")
    
        except AttributeError as e:
            print(f"{e} Need to run get_accuracies_from_pths first.")

    def load_losses(self):
        if os.path.isfile(f"losses/{self.model.name}_{self.dl.feature_type}.npy"):
            return np.load(f"losses/{self.model.name}_{self.dl.feature_type}.npy")
        else:
            print("Can't find any losses")


