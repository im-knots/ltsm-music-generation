# LTSM Music Generation

PLEASE NOTE THIS IS A WORK IN PROGRESS. I HAVENT FIGURED OUT HOW TO TUNE THE MODEL FULLY YET :D

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


## create_tfrecord.py
This script processes the audio files located in the "audio" directory. It loads each audio file, computes its Short-Time Fourier Transform (STFT), and converts it into a Mel-spectrogram. Then, it creates input-target pairs by taking a chunk of the Mel-spectrogram and its shifted version. The Mel-spectrogram is saved as a JPEG image in the "spectrograms" directory. Finally, it writes the input-target pairs to a TFRecord file, which is either saved locally or uploaded to Google Cloud Storage.

## check_tfrecord.py
This script reads the created TFRecord file, parses the input-target pairs, and visualizes them as images in the "tfrecord_check" directory. It helps verify that the data has been correctly processed and stored in the TFRecord file.

## train_model.py
This script trains a deep learning model using the data stored in the TFRecord file. The model consists of several layers, including bidirectional LSTMs, layer normalizations, and an attention layer. The model is compiled with the Adam optimizer and the mean squared error loss function. The training and validation datasets are prepared using the input-target pairs from the TFRecord file. The model is trained on a TPU or GPU, and the weights are saved after each epoch. The final model is saved either locally or on Google Cloud Storage.

## generate_songs.py
This script generates new audio samples based on the trained model. It first loads the model from local storage or Google Cloud Storage, depending on the specified parameters. It then generates new Mel-spectrograms by feeding the model with an initial seed and extending the prediction iteratively. The generated Mel-spectrogram is then converted back to an audio signal using an inverse STFT. The generated audio samples are saved as .wav files either locally or on Google Cloud Storage.


