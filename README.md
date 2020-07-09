# Music Genre Classification with CNNs and RNNs (in PyTorch)

#### This project is an exploration of recent research into the use of Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), as well as a combination of the two, for classification of music genres. Experiments into the choice of audio features and architecture of the neural networks were performed to compare the relative performance of various implementations.

#### Themes: Music Information Retrieval, Audio Signal Processing, (Fourier) Transforms

Note: I am neither an expert in the field of Music Information Retrieval (MIR) nor am I particularly well-versed in deep learning. This project is my attempt to get my feet wet and start exploring both of these fields. Big shoutout to arXiv, Medium and countless other blogs, and DeepMind x UCL lecture series for guiding me through the world of MIR and Deep Learning, and Google Colab for the free access to its GPU!

## Introduction

In the field of machine learning, problems such as image classification or natural language processing often take centre stage, and rightly so given how common images and language are used in daily life. The research into music information retrieval (MIR) is likely only a small fraction of the research into these other areas, but it is nonetheless a fascinating one. Problems within MIR, such as genre classification, are often multi-faceted, sitting within the intersections between various research areas of machine learning. Take music genre classification as an example (and because this is also the problem that is going to be tackled later on). Music can be represented pictorially via the frequency domain or temporally through its inherent nature of being a time series data. This potentially means that tools from both image classification and natural language processing can be used to tackle the problem of identifying music genres.

### Related Work

Neural networks, including convolutional neural networks (CNNs) and recurrent neural networks (RNNs), have been used in the past decade to tackle the problem of music genre classification with varying degrees of success. Li et al [1] first used a CNN on the MFCC of tracks and produced convincing results that showed the prowess of neural networks on genre classification. They were subsequently followed by others such as Nakashika et al [2], who also used a CNN on other audio features such as mel-spectrograms and STFT. In more recent years, with the advent of more advanced CNN architecture, Bian et al [3] adopted the Resnet and Densenet architectures to produce a better performing CNN that outperforms a vanilla CNN architecture. The Inception CNN architecture conjured by Szegedy et al [4], was also used by Liu et al [5] to create a novel CNN architecture known as the Bottom-up Broadcast Neural Network specifically for genre classification.  

In addition, given the temporal nature of music, the combination of a CNN and an RNN led to the creation of two different proposed architecture - the sequential convolutional recurrent neural network by Choi et al [6], and the paralleling recurrent convolutional neural network by Feng et al [7]. These two different architectures employ the classic neural networks (convolutional and recurrent) from the domains of both image classification and natural language processing in order to capture the frequential and temporal features of a music track.

This project aims to replicate some of the architectures mentioned above and compare the relative performance of these architectures.


## Method

### Dataset
This project uses the Free Music Archive (FMA) dataset, which can be found here: https://github.com/mdeff/fma. More specifically, this project uses the _small dataset_ containing 8000 tracks with 8 equally distributed genres, and provides a 30 second clip of the tracks in an `.mp3` file format. 

Moreover, this dataset provides pre-computed features, which includes the statistical features of various types of transforms of the raw audio data, such as Constant-Q Transform (CQT) and Mel-Spectrogram Cepstral Coefficients (MFCC). Twelve non-deep learning machine learning algorithms were used on these pre-calculated features to obtain a baseline accuracy. A simple overview of the pre-computed features and baseline models can be found at [exploration.ipynb](https://nbviewer.jupyter.org/github/yijerjer/music_genre_classification/blob/master/notebooks/exploration.ipynb) and [base_models.ipynb](https://nbviewer.jupyter.org/github/yijerjer/music_genre_classification/blob/master/notebooks/base_models.ipynb) respectively.


### Features

Using the raw audio `.mp3` files provided by the FMA dataset, two main features, Constant-Q Transform (CQT) and Short-Time Fourier Transform (STFT), were extracted from the raw audio data, which produced a 2D feature of a track in the frequency and temporal domain. A hop length of 1024 was used in both cases, and a n_fft of 1024 was used for the STFT. These values were used in order to balance between achieving a high enough resolution of the features and maintaining a reasonably sized input to the neural network later on. Variations of these two features were also obtained, where the Chroma CQT and Chroma energy normalised statistics (CENS) were obtained from the CQT, and the Chroma STFT and Mel-scaled STFT were obtained from the STFT. In total, there were six features that were extracted - CQT, Chroma CQT, CENS, STFT, Chroma STFT and Mel-scaled STFT, and these can be seen in the image below. (Note that in the end, the linear scaled STFT was not used due to its large size and the lack of computational resources for it to be used as an input)

![alt text](https://github.com/yijerjer/music_genre_classification/blob/master/plots.png?raw=true)

### Architecture

#### CNN

Taking inspiration from the various CNN architectures in Choi et al's [6] paper, 3 simple CNN architectures were devised and experimented with. Below are overviews of the three different architectures:
1. Frequency CNN: This CNN contains four 1D convolutions of length 5 in the temporal axis with appropriate max pooling layers, until a 1D array along the frequency axis is left per feature map. The 1D arrays are then flattened and fed into a fully connected neural network.

2. Temporal CNN: The first convolution is a 2D convolution which spans across the entire frequency axis and with length 5 in the temporal axis. The following three convolutions are of length 5 in the temporal axis. The CNN leaves behind a single value per feature map, which is then fed into a fully connected neural network. 

3. Square CNN: This convolutional neural network contains four 2D square convolutions alongside max pooling layers that are not necessarily square, until a single value is left per feature map. This is then fed into a fully connected neural network for classification.

In addition, experiments using the Resnet and Densenet CNN architectures were performed on the Frequency CNN architecture as well. Taking inspiration from the results of the paper by Bian et al [3], two implementations were experimented with here:
1.  the Resnet/Densenet replaces the second convolution in the Frequency CNN, and
2. the Resnet/Densenet replaces all of the four convolutions in the Frequency CNN.

The exact implementation of each CNN for each of the six features can be found in [cnn_models.py](https://github.com/yijerjer/music_genre_classification/blob/master/cnn_models.py).

#### Bottom-up Broadcast Neural Network

The implementation here was identical to the architecture proposed by Liu et al [5]. The core of the architecture contains three Inception blocks, which are blocks that contain four parallel streams of different convolutions and/or max pooling layers that are concatenate at the end of each Inception block. The exact details of the implementation can be found in [bbnn_models.py](https://github.com/yijerjer/music_genre_classification/blob/master/bbnn_models.py). 

#### Convolutional + Recurrent Neural Network

From the experiment with CNNs in the above subsection, the best performing feature was used in this section. The sequential convolutional recurrent neural network by Choi et al [6] and paralleling recurrent convolutional neural network by Feng et al [7] were implemented here. Here are simple overviews of each of their architecture:
* Sequential: the 2D feature is fed into a CNN with square convolutions which outputs a 2D array that contains a smaller number of points along the temporal axis. This is then fed into a many-to-one RNN. The output from the RNN is subsequently fed into a fully connected layer for classification.
* Parallel: the 2D feature is fed into a CNN identical to the Square CNN mentioned two subsection above. At the same time, the 2D feature undergoes an average pooling layer, before being fed into an RNN. The outputs from the CNN and RNN are then concatenated together, and inputted into a fully connected neural network for classification.

In the recurrent part of both neural networks, experiments were performed with LSTM and GRU cells, as well as the neural networks being unidirectional or bidirectional.

### Methodology and Other Implementation Details

For each architecture and each feature, three repeats were performed in order to obtain an approxiamate value of the mean accuracies of the model. The models were trained for between 50 - 100 epochs, and the models were saved at every 5 epochs. The test accuracy was determined for each model by picking out the maximum accuracy in all of the saved epochs. 

All of the models aboved used the Adam optimizer with a learning rate of 0.001. Better performance was observed when compared to the SGD optimizer in initial experiments. Each layer of convolution is followed by batch normalisation and a ReLU activation function.


## Results and Discussion

### Baseline Models

The results of the baseline classifiers can be found here: [base_models.ipynb](https://nbviewer.jupyter.org/github/yijerjer/music_genre_classification/blob/master/notebooks/base_models.ipynb). In addition, the notebook also contains a t-SNE plot of the pre-computed features, which gives a good visual representation of how the features are distributed.

From the 12 `sklearn` classifiers used here, the top three best performing classifiers in decreasing order of accuracy were 1. SVM with an RBF kernel, 2. MLP clasifier with a single hidden layer of length 100, and 3. a Ridge Classifier, with test accuracies of . The ROC curves and the confusion matrices of each of the 12 models can be found in the notebook mentioned above.

### CNNs

The accuracies of each CNN architecture for each feature are listed in the table below.

| Feature     | Frequency | Temporal | Square   |
| -------     | :-------: | :------: | :------: |
| Chroma CQT  | 36.0±0.4  |	32.8±0.4 | 33.5±0.9 |
| Chroma CENS | 31.6±0.2  | 28.4±0.2 | 29.9±0.5 |
| Chroma STFT | 37.6±0.8  | 35.1±0.5 | 35.5±0.8 |
| Mel STFT    | 40.9±0.6  | 36.8±0.4 | 38.6±1.1 |
| CQT         | **45.5±0.4**  | **39.3±0.5** | **43.1±0.8** |

It is clear that the CQT performs the best amongst the other features. The experiments, along with information such as the training loss, ROC curve and confusion matrices can be found in the following notebooks: [chroma_cnn_architecture.ipynb](https://nbviewer.jupyter.org/github/yijerjer/music_genre_classification/blob/master/notebooks/chroma_cnn_architecture.ipynb), [mel_cnn_architecture.ipynb](https://nbviewer.jupyter.org/github/yijerjer/music_genre_classification/blob/master/notebooks/mel_cnn_architecture.ipynb) and [cqt_cnn_architecture.ipynb](https://nbviewer.jupyter.org/github/yijerjer/music_genre_classification/blob/master/notebooks/cqt_cnn_architecture.ipynb).

Below is a table with the results of using the CQT (the best performing feature) on the implementation of the Resnet and Desnenet architectures, which were mentioned earlier in the CNN Methods section.

| Architecture                                      | Accuracy |
| --------------------------------------------------| :------: |
| Basic convolution for all four convolution layers | 45.5±0.4 |
| Resnet at second convolution                      | 42.8±0.5 |
| Densenet at second convolution                    | 45.8±0.2 |
| Resnet for all four convolution layers            | 44.8±0.6 |
| Densenet for all four convolution layers          | **46.7±0.5** |

This experiment here shows that Densenet is able to marginally outperform a basic convolutional neural network, however the Resnet fails to do so. It should be noted that Resnet and Densenet are more computationally expensive than a basic convolution layer due to their added complexity.


### Convolutional + Recurrent Neural Networks

The CQT of the tracks were used here given that this was the best performing feature in the subsection above. The accuracies of the different convolutional + recurrent architectures and variations of them are listed in the table below.

| Architecture | LSTM     | LSTM + Bidrectional | GRU          | GRU + Bidrectional |
| ------------ | :------: | :-----------------: | :------:     | :----------------: |
| Sequential   | 41.5±0.5 | 40.1±0.7            | **44.9±0.2** | 43.4±0.5           |
| Parallel     | 42.0±0.4 | **44.9±0.1**        | 43.1±0.6     | 41.6±0.5           |

For the sequential convolutional recurrent neural network, the unidirectional GRU RNN produces the best accuracy, whilst the bidirectional LSTM RNN produces the best accuracy for the parallel recurrent convolution neural network.

### Bottom-up Broadcast Neural Network

An accuracy of **51.1±0.1** was achieved with this neural network. Note that this neural network was the most computationally expensive compared to the rest, and took a significantly longer amount of time for training.

## Discussion and Conclusion



## Further Work
* Data Augmentation
* Additional two papers on autoencoders, and random projections of mel-spectrograms
* Use other audio features such as MFCC, and potentially combine multiple features together 

Have a look at CQT exploration by genre.

## References

[1] T. Li, A. Chan, A. Chun, "Automatic Musical Pattern Feature Extraction Using Convolutional Neural Network", 2010

[2] T. Nakashika, C. Garcia, T.Takiguchi, "Local feature-map integration using convolutional neural networks for music genre classification", 2012

[3] W. Bian, J. Wang, B. Zhuang, J. Yang, S. Wang, J. Xiao, "Audio-Based Music Classification with DenseNet And Data Augmentation", 2019

[4] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V.Vanhoucke, A. Rabinovich, "Going Deeper with Convolutions", 2014

[5] C. Liu, L. Feng, G. Liu, H. Wang, S. Liu, "Bottom-up Broadcast Neural Network For Music Genre Classification", 2019

[6] K. Choi, G. Fazekas, M. Sandler, K. Cho, "Convolutional Recurrent Neural Networks for Music Classification", 2016

[7] L. Feng, S. Liu, J. Yao, "Music Genre Classification with Paralleling Recurrent Convolutional Neural Network", 2017


[exploration.ipynb](https://nbviewer.jupyter.org/github/yijerjer/music_genre_classification/blob/master/notebooks/exploration.ipynb): Exploration of the FMA dataset and understanding the provided librosa features 
