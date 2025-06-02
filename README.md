# musicgenreclassification
This project implements deep learning models to classify music genres using log-transformed Mel spectrograms. Leveraging convolutional and recurrent neural networks, the models learn temporal and spectral features from short audio segments. This work builds on the GTZAN dataset, a widely used benchmark in music information retrieval.

**Note**: The dataset used is not included in the repository. See the "Data" section below for a description

## Project Overview

The project aims to classify musical segments into one of 8 genres using image-like representations of Mel spectrograms. The notebook implements and compares three deep learning architectures.

- **Section 1**: A shallow **parallel CNN** with two convolutional branches
- **Section 2**: A hybrid **CNN + LSTM** model for capturing sequential structure
- **Section 3**: A deep **CNN + Bidirectional LSTM** with data augmentation and batch normalization to improve generalization

Each model is trained on pre-processed spectrogram data and evaluated for genre classification accuracy

## Architectures Used

| Model Stage | Description | Accuracy |
|-------------|-------------|----------|
| **P1.1** | Shallow parallel CNN with two branches (different filter sizes) | ~72% |
| **P1.2** | CNN + LSTM hybrid to combine local and temporal features | ~59% |
| **P2** | Deeper CNN + Bidirectional LSTM with data augmentation and batch norm | **~85%** |

## Data

The data represents **log-transformed Mel spectrograms** derived from the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

- The original GTZAN dataset contains 1,000 songs (30 sec each) across 10 genres.
- For this project, it was:
  - Reduced to 8 genres (800 songs total)
  - Extracted 15 spectrograms per song
  - Used 80x80 grayscale images (shape: `(80, 80, 1)`)
  - Trained models on 80% of the data (training set), with the remaining 20% as validation

Each spectrogram encodes:
- **Time** (x-axis)
- **Frequency** (y-axis)
- **Intensity** (pixel brightness)

The dataset is preprocessed and stored as TensorFlow datasets outside this repository.

## Key Techniques

- **Data Augmentation**: Gaussian noise added to spectrograms
- **Convolutional Layers**: Feature extraction from 2D Mel spectrograms
- **LSTMs and BiLSTMs**: Capturing sequential patterns over time
- **Regularization**: Batch normalization, dropout, and tuning
- **Training Tools**: TensorFlow, Keras, GPU acceleration (Colab)
