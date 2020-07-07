import utils
import torch
import random
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader


class ModelUtils():
    def __init__(self, model, criterion, optimizer, train_dataset, cuda=False, load_workers=0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_set = train_dataset
        self.cuda = cuda
        self.device = torch.device("cuda:0" if cuda else "cpu")
        self.load_workers = load_workers

    def train_batch(self, input_tensor, genre_tensor, has_hidden=False):
        self.optimizer.zero_grad()
        if has_hidden:
            batch_size = input_tensor.shape[0]
            hidden = self.model.init_hidden(batch_size, cuda=self.cuda)
            output = self.model(input_tensor, hidden)
        else:
            output = self.model(input_tensor)
        loss = self.criterion(output, genre_tensor)
        loss.backward()
        self.optimizer.step()

        return output, loss.item()

    def train(self, n_epochs=70, batch_size=32, save_pth_every=5, print_every=5, pth_dir="pths", loss_dir="losses", has_hidden=False):
        current_loss = 0
        self.all_losses = []
        print_every = print_every
        data_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=self.load_workers, pin_memory=True)

        if not os.path.isdir(loss_dir):
            os.makedirs(loss_dir)

        if save_pth_every > 0:
            if not os.path.isdir(pth_dir):
                os.makedirs(pth_dir)

            pths_dir = f"{pth_dir}/{self.model.name}_{self.train_set.feature_type}"
            if not os.path.isdir(pths_dir):
                os.makedirs(pths_dir)

        start = time.time()
        total_time = 0
        transfer_time = 0
        train_time = 0

        if save_pth_every > 0:
            file_path = f"{pth_dir}/{self.model.name}_{self.train_set.feature_type}/epoch_0.pth"
            torch.save(self.model.state_dict(), file_path)
            print(f"Saved model state as {file_path}")

        for epoch in range(1, n_epochs + 1):
            total_start = time.time()
            for data in data_loader:
                transfer_start = time.time()
                input_tensor, genre_tensor, genre, _ = data
                input_tensor, genre_tensor = input_tensor.float().to(self.device), genre_tensor.to(self.device)
                transfer_time += time.time() - transfer_start
                
                train_start = time.time()
                output, loss = self.train_batch(input_tensor, genre_tensor, has_hidden=has_hidden)
                current_loss += loss
                train_time += time.time() - train_start

            total_time += time.time() - total_start
            if epoch % print_every == 0:
                print(f"{utils.time_since(start)}, EPOCH {epoch}/{n_epochs}, Loss: {round(current_loss, 2)}, Total time: {round(total_time, 3)}, Transfer time: {round(transfer_time, 3)}, Train time: {round(train_time, 3)}")
                total_time = 0
                transfer_time = 0
                train_time = 0
            
            self.all_losses.append(current_loss)
            current_loss = 0

            if save_pth_every > 0 and epoch % save_pth_every == 0:
                file_path = f"{pth_dir}/{self.model.name}_{self.train_set.feature_type}/epoch_{epoch}.pth"
                torch.save(self.model.state_dict(), file_path)
                print(f"Saved model state as {file_path}")
        
        np.save(f"{loss_dir}/{self.model.name}_{self.train_set.feature_type}.npy", self.all_losses)


class ModelAnalytics():
    def __init__(self, model, datasets, n_epochs=70, save_pth_every=5, batch_size=32, load_workers=0, cuda=False):
        self.model = model
        self.cuda = cuda
        self.device = torch.device("cuda:0" if cuda else "cpu")

        train_set, validate_set, test_set = datasets
        self.train_dl = DataLoader(train_set, batch_size=batch_size, num_workers=load_workers)
        self.validate_dl = DataLoader(validate_set, batch_size=batch_size, num_workers=load_workers)
        self.test_dl = DataLoader(test_set, batch_size=batch_size, num_workers=load_workers)

        self.genres = train_set.genres
        self.feature_type = train_set.feature_type
        n_genres = len(train_set.genres)
        self.n_pths = int(n_epochs / save_pth_every) + 1
        self.save_pth_every = save_pth_every
        
        self.epochs = []
        self.train_accuracy = []
        self.validate_accuracy = []
        self.test_accuracy = []
        self.train_confusion_mats = torch.zeros(self.n_pths, n_genres, n_genres)
        self.validate_confusion_mats = torch.zeros(self.n_pths, n_genres, n_genres)
        self.test_confusion_mats = torch.zeros(self.n_pths, n_genres, n_genres)
        self.fpr, self.tpr, self.roc_auc = dict(), dict(), dict()

    def genre_from_output(self, output_tensor):
        _, top_i = output_tensor.topk(1)
        genre_i = top_i.flatten()
        return [self.genres[i] for i in genre_i], genre_i

    def evaluate_pths(self, has_hidden=False, pth_dir="pths"):
        pths_dir = f"{pth_dir}/{self.model.name}_{self.feature_type}"
        print(f"Evaluating pths for epochs: ", end=" ")
        with torch.no_grad():
            start = time.time()
            for n in range(self.n_pths):
                epoch_num = n * self.save_pth_every
                self.epochs.append(epoch_num)

                pth_file = f"{pths_dir}/epoch_{epoch_num}.pth"
                state_dict = torch.load(pth_file, map_location=torch.device(self.device))
                self.model.load_state_dict(state_dict)
                self.model.eval()

                for which_set in ["train", "validate", "test"]:
                    for data in eval(f"self.{which_set}_dl"):
                        input_tensor, genre_tensor, genre, _ = data
                        input_tensor = input_tensor.float().to(self.device)
                        genre_tensor = genre_tensor.float()

                        if has_hidden:
                            batch_size = input_tensor.shape[0]
                            hidden = self.model.init_hidden(batch_size, cuda=self.cuda)
                            model_out = self.model(input_tensor, hidden)
                        else:
                            model_out = self.model(input_tensor)
                        predict_score, predicted_i = self.genre_from_output(model_out)
                        self.add_to_confusion_matrix(n, genre_tensor, predicted_i, which_set=which_set)

                        del input_tensor, genre_tensor, model_out
                    
                print(f"{epoch_num} ({utils.time_since(start)}),", end=" ")

            print("Done.")

            self.get_accuracies()

            max_test_accuracy = max(self.test_accuracy)
            max_test_index = self.test_accuracy.index(max_test_accuracy)
            self.roc_epoch = self.epochs[max_test_index]
            pth_file = f"{pths_dir}/epoch_{self.roc_epoch}.pth"
            state_dict = torch.load(pth_file, map_location=torch.device(self.device))
            self.model.load_state_dict(state_dict)
            self.model.eval()

            roc_data = {"actuals": torch.empty(0), "outputs": torch.empty(0, 8)}
            for data in self.test_dl:
                input_tensor, genre_tensor, genre, _ = data
                input_tensor = input_tensor.float().to(self.device)
                genre_tensor = genre_tensor.float()
                if has_hidden:
                    batch_size = input_tensor.shape[0]
                    hidden = self.model.init_hidden(batch_size, cuda=self.cuda)
                    model_out = self.model(input_tensor, hidden)
                else:
                    model_out = self.model(input_tensor)

                roc_data["actuals"] = torch.cat((roc_data["actuals"], genre_tensor), dim=0)
                roc_data["outputs"] = torch.cat((roc_data["outputs"], model_out.cpu()), dim=0)

                del input_tensor, genre_tensor, model_out

            self.fpr, self.tpr, self.roc_auc = self.get_roc_curves(roc_data["actuals"], roc_data["outputs"])

    def make_plots(self, loss_dir="losses"):
        self.print_auc_accuracy()
        plt.figure(figsize=(8, 8))

        plt.subplot(2, 2, 1)
        losses = self.load_losses(loss_dir=loss_dir)
        if losses.any():
            plt.plot(losses)
            plt.xlabel("Loss")
            plt.title("Training Loss")
            plt.tight_layout()
        
        plt.subplot(2, 2, 2)
        self.plot_accuracies()
        plt.tight_layout()

        plt.subplot(2, 2, 3)
        self.plot_roc()
        plt.tight_layout()

        plt.subplot(2, 2, 4)
        self.plot_confusion_matrix()
        plt.tight_layout()
    
    def print_auc_accuracy(self):
        accuracies = [self.train_accuracy, self.validate_accuracy, self.test_accuracy]
        max_accuracies = [max(accuracy) for accuracy in accuracies]
        max_index = [accuracy.index(max_accuracies[i]) for i, accuracy in enumerate(accuracies)]
        max_accuracies = [round(max_val.item(), 4) for max_val in max_accuracies]

        print("ACCURACIES")
        for i, which_set in enumerate(["train", "validate", "test"]):
            print(f"Maximum {which_set} accuracy: {max_accuracies[i]} at epoch {self.epochs[max_index[i]]}")

        max_test_index = self.epochs.index(self.roc_epoch)
        print(f"\nAT EPOCH {self.roc_epoch}")
        final_accuracy = round(self.test_accuracy[max_test_index].item(), 4)
        final_auc = round(self.roc_auc['macro'], 4)
        print(f"Macro Test AUC: {final_auc}, Accuracy: {final_accuracy}")

        print("\nINDIVIDUAL TEST AUC AND ACCURACY")
        confusion_mat = self.test_confusion_mats[max_test_index]
        for i, genre in enumerate(self.genres):
            genre_accuracy = round((confusion_mat[i][i] / confusion_mat[i].sum()).item(), 4)
            genre_auc = round(self.roc_auc[i], 4)
            print(f" - {genre}: AUC of {genre_auc}, Accuracy of {genre_accuracy}")

    def get_accuracies(self):
        if len(self.train_accuracy) != self.n_pths:
            for i in range(self.n_pths):
                for which_set in ["train", "validate", "test"]:
                    confusion_mat = eval(f"self.{which_set}_confusion_mats")[i]
                    accuracy = self.accuracy_from_confusion_matrix(confusion_mat)
                    eval(f"self.{which_set}_accuracy").append(accuracy)

        return self.epochs, self.train_accuracy, self.validate_accuracy, self.test_accuracy

    def add_to_confusion_matrix(self, n_pth, actual_genre, predicted_genre, which_set="test"):
        confusion_mat = eval(f"self.{which_set}_confusion_mats")[n_pth]
        for i, genre_i in enumerate(actual_genre):
            confusion_mat[int(genre_i)][int(predicted_genre[i])] += 1

    def accuracy_from_confusion_matrix(self, confusion_mat):
        return confusion_mat.diag().sum() / confusion_mat.sum()
    
    def get_roc_curves(self, actual_genre, model_output):
        print(actual_genre.shape, model_output.shape)
        n_genres = len(self.genres)
        actual_genre, model_output = actual_genre.numpy(), model_output.cpu().detach().numpy()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_genres):
            fpr[i], tpr[i], _ = roc_curve((actual_genre == i), model_output[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # actual_genre_micro = np.concatenate([(actual_genre == i) for i in range(n_genres)])
        # fpr["micro"], tpr["micro"], _ = roc_curve(actual_genre_micro, model_output.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        fpr["macro"] = np.unique(np.concatenate([fpr[i] for i in range(n_genres)]))
        tpr["macro"] = np.zeros_like(fpr["macro"])
        for i in range(n_genres):
            tpr["macro"] += np.interp(fpr["macro"], fpr[i], tpr[i])
        tpr["macro"] /= n_genres
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return fpr, tpr, roc_auc

    def load_losses(self, loss_dir="losses"):
        loss_dir = f"{loss_dir}/{self.model.name}_{self.feature_type}.npy"
        if os.path.isfile(loss_dir):
            return np.load(loss_dir)
        else:
            print(f"Can't find {loss_dir}")
            return np.array([])

    def plot_accuracies(self):
        epochs, train_acc, validate_acc, test_acc = self.get_accuracies()

        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, validate_acc, label="Validate")
        plt.plot(epochs, test_acc, label="Test")

        plt.legend()
        plt.ylim([0, 1])
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
    
    def plot_roc(self):
        plt.plot([0, 1], [0, 1], color='k', linewidth=0.5)
        for i, genre in enumerate(self.genres):
            plt.plot(self.fpr[i], self.tpr[i], linewidth=0.8, linestyle="dashed", label=f"{genre}")
        
        # plt.plot(self.fpr["micro"], self.tpr["micro"], linewidth=1.5, label=f"Micro ROC")
        plt.plot(self.fpr["macro"], self.tpr["macro"], linewidth=1.5, label=f"Macro ROC")
        
        plt.legend()
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"Test ROC Curve at Epoch {self.roc_epoch}")
    
    def plot_confusion_matrix(self):
        confusion_mat = self.test_confusion_mats[self.epochs.index(self.roc_epoch)]
        n_genres = len(self.genres)

        plt.imshow(confusion_mat)
        plt.xticks(np.arange(n_genres), labels=self.genres, rotation="vertical")
        plt.yticks(np.arange(n_genres), labels=self.genres)
        for i in range(n_genres):
            for j in range(n_genres):
                plt.text(j, i, int(confusion_mat[i][j]), va="center", ha="center")
        
        plt.title(f"Confusion Matrix at Epoch {self.roc_epoch}")