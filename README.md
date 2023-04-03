# LTSM Music Generation

This project consists of three Python scripts that work together to process audio files, train a model using the processed data, and generate new songs based on the trained model.

## Scripts

### 1. create_tfrecord.py

This script processes audio files in the specified directory, converts them to mel-spectrograms, and saves the spectrograms as JPEG images. It also creates a TFRecord file containing the processed audio data.

### 2. train_model.py

This script trains an LSTM model using the data stored in the TFRecord file created by `create_tfrecord.py`. The model is saved to the specified directory.

### 3. generate_songs.py

This script generates new songs based on the trained LSTM model. The generated songs are saved as WAV files in the specified directory.

## Usage

1. Run `create_tfrecord.py` to process the audio files and create the TFRecord file.
2. Run `train_model.py` to train the LSTM model using the data from the TFRecord file.
3. Run `generate_songs.py` to generate new songs based on the trained model.

## Dependencies

- TensorFlow
- Keras
- Librosa
- SciPy
- Matplotlib
- Google Cloud Storage (optional, for saving and loading data to/from Google Cloud Storage)

## Notes

- You may need to modify the script parameters to match your specific use case, such as input and output directories, audio file format, sample rate, and the number of generated songs.
- For training and generating songs, you can choose to use TPU, GPU, or CPU by adjusting the `use_tpu` parameter in `train_model.py` and `generate_songs.py`.
- You can also choose to read and write data from/to local storage or Google Cloud Storage by adjusting the `read_local` and `write_local` parameters in the respective scripts.


# A Deep Learning Approach to Audio Generation Using Mel-spectrogram and LSTM Networks

## Abstract

We present a deep learning-based approach to generating audio using Mel-spectrogram representations and Long Short-Term Memory (LSTM) networks. The proposed method consists of three main stages: preprocessing, model training, and song generation. In the preprocessing stage, audio files are converted to Mel-spectrograms using the Short-Time Fourier Transform (STFT), which are then logarithmically scaled to better represent human perception. The dataset is prepared by dividing the Mel-spectrograms into overlapping input-target pairs. In the model training stage, an LSTM model is trained on the prepared dataset to learn the temporal dependencies between timesteps. Finally, in the song generation stage, new songs are generated by conditioning the trained LSTM model on an input seed and iteratively predicting subsequent timesteps. Our method demonstrates the potential of deep learning techniques for generating novel and coherent audio content.

## Introduction

The generation of audio content is an important task in various applications such as music synthesis, sound design, and virtual reality. Traditional methods for audio synthesis often involve handcrafted algorithms that require expert knowledge and may not generalize well to different types of sounds. In this paper, we present a data-driven approach to audio generation based on deep learning techniques. We leverage the expressive power of LSTM networks to model the complex temporal dependencies inherent in audio signals. Our method processes audio files as Mel-spectrograms, a perceptually relevant representation, and trains an LSTM model to generate new Mel-spectrograms, which are then converted back to audio waveforms.

## Method

### Preprocessing

The preprocessing stage involves the following steps:

1. **Loading audio files**: Audio files are loaded and resampled to the desired sample rate using the Librosa library.
2. **Computing the STFT**: The STFT is calculated for each audio file to obtain a time-frequency representation.
3. **Converting the STFT to a Mel-spectrogram**: The STFT is transformed into a Mel-spectrogram using the Mel filter bank to achieve a perceptually meaningful representation.
4. **Applying logarithmic scaling**: The Mel-spectrogram is converted to a logarithmic scale to enhance its perceptual relevance.
5. **Dataset creation**: Log-scaled Mel-spectrograms are divided into overlapping input-target pairs and stored in a list.
6. **TFRecord file generation**: The input-target pairs are serialized as TensorFlow `Example` objects and written to a TFRecord file for efficient storage and retrieval during model training.

### Model Training

The model training stage consists of the following steps:

1. **Environment setup**: The appropriate computing environment is configured based on the selected device (TPU, GPU, or CPU).
2. **LSTM model construction**: A Keras-based LSTM model is built, comprising multiple LSTM layers, dropout layers, and a final Dense layer with linear activation. The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss function.
3. **Dataset loading**: The dataset is loaded from the TFRecord file, parsed, shuffled, batched, and repeated to form an infinite stream of input-target pairs.
4. **Model training**: The LSTM model is trained on the dataset for a specified number of epochs and steps per epoch.
5. **Model saving**: The trained model is saved to a specified directory for use in song generation.

### Song Generation

The song generation stage involves the following steps:

1. **Environment setup**: The computing environment is configured based on the selected device (TPU, GPU, or CPU).
2. **Model loading**: The trained LSTM model is loaded from the specified directory.
3. **New song generation**: A seed input is extracted from the dataset, and the trained LSTM model is used to predict the next timestep based on this input. The predicted timestep is then appended to the input, and the process is iteratively repeated for the desired number of timesteps. This generates a new Mel-spectrogram, which represents a novel and coherent audio sequence.
4. **Converting Mel-spectrogram back to audio**: The generated Mel-spectrogram is transformed back to the time-domain audio waveform using the inverse Short-Time Fourier Transform (iSTFT) provided by the Librosa library. The resulting audio is saved as an output file in the desired format (e.g., WAV, MP3).

## Conclusion

We have presented a deep learning approach for audio generation based on Mel-spectrogram representations and LSTM networks. Our method demonstrates the potential of leveraging deep learning techniques to generate novel and coherent audio content. The proposed method can be applied to various applications, including music synthesis, sound design, and virtual reality, where generating high-quality audio content is crucial. Future work may explore alternative deep learning architectures, such as Transformer models, and incorporate techniques for generating audio with finer temporal control and incorporating higher-level musical structures.
