# Music Genre Classification with CNNs and RNNs

#### This project is an exploration of recent research into the use of Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), as well as a combination of the two, for classification of music genres. Experiments into the choice of audio features and architecture of the neural networks were performed to compare the relative performance of various implementations.

#### Themes: Music Information Retrieval, Audio Signal Processing, (Fourier) Transforms

#### Note: I am neither an expert in the field of Music Information Retrieval (MIR) nor am I particularly well-versed in deep learning. This project is my attempt to get my feet wet and start exploring both of these fields. Big shoutout to arXiv, Medium and countless other blogs, and DeepMind x UCL lecture series for guiding me through the world of MIR and Deep Learning.

## Introduction

In the field of machine learning, problems such as image classification or natural language processing often take centre stage, and rightly so given how common images and language are used in daily life. The research into music information retrieval (MIR) is likely only a small fraction of the research into these other areas, but it is nonetheless a fascinating one. Problems within MIR, such as genre classification, are often multi-faceted, sitting within the intersections between various research areas of machine learning. Take music genre classification as an example (and because this is also the problem that is going to be tackled later on). Music can be represented pictorially via the frequency domain or temporally through its inherent nature of being a time series data. This potentially means that tools from both image classification and natural language processing can be used to tackle the problem of identifying music genres.

### Related Work

Neural networks, including convolutional neural networks (CNNs) and recurrent neural networks (RNNs), have been used in the past decade to tackle the problem of music genre classification with varying degrees of success. Li et al [1] first used a CNN on the MFCC of tracks and produced convincing results that showed the prowess of neural networks on genre classification. They were subsequently followed by others such as Nakashika et al [2], who also used a CNN on other audio features such as mel-spectrograms and STFT. In more recent years, with the advent of more advanced CNN architecture, Bian et al [3] adopted the Resnet and Densenet architectures to produce a better performing CNN that outperforms a vanilla CNN architecture. The Inception CNN architecture conjured by Szegedy et al [4], was also used by Liu et al [5] to create a novel CNN architecture known as the Bottom-up Broadcast Neural Network specifically for genre classification.  

In addition, given the temporal nature of music, the combination of a CNN and an RNN led to the creation of two different proposed architecture - the linear convolutional recurrent neural network by Choi et al [6], and the paralleling recurrent convolutional neural network by Feng et al [7]. These two different architectures employ the classic neural networks (convolutional and recurrent) from the domains of both image classification and natural language processing in order to capture the frequential and temporal features of a music track.

This project aims to replicate some of the architectures mentioned above and compare the relative performance of these architectures.


## Method

### Dataset
This project uses the Free Music Archive (FMA) dataset, which can be found here: https://github.com/mdeff/fma. More specifically, this project uses the _small dataset_ containing 8000 tracks, with 8 equally distributed 'parent' genre.

This dataset also provides pre-computed features, which includes the statistical features of various types of transforms of the raw audio data, such as Constant-Q Transform (CQT) and Mel-Spectrogram Cepstral Coefficients (MFCC). Non-deep learning machine learning algorithms were used on these pre-calculated features to obtain a baseline accuracy. A simple overview of the baseline models can be found here: [base_models.ipynb]() 


### Architecture

### Methodology


## Results and Discussion

### Baseline Models

## Conclusions

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
