import os
import numpy as np
import librosa
import tensorflow as tf
import concurrent.futures
import matplotlib.pyplot as plt
import threading

# Parameters
input_directory = "audio"
write_local = True
sr = 22050
timesteps = 50
n_mels = 128
num_workers = 8
overlap = 50
spectrogram_directory = "spectrograms"

save_spectrogram_lock = threading.Lock()

def create_example(input_data, target_data):
    feature = {
        'input': tf.train.Feature(float_list=tf.train.FloatList(value=input_data.flatten())),
        'target': tf.train.Feature(float_list=tf.train.FloatList(value=target_data.flatten()))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tfrecord_file(data, tfrecord_filename):
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for input_data, target_data in data:
            example = create_example(input_data, target_data)
            writer.write(example.SerializeToString())
    print(f"Creating tfrecord file {tfrecord_filename}")

def save_spectrogram(log_mel_spectrogram, output_path):
    with save_spectrogram_lock:
        plt.figure(figsize=(10, 4))
        plt.imshow(log_mel_spectrogram, aspect="auto", origin="lower")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def process_file(filename):
    print(f"Processing file: {filename}")
    audio, _ = librosa.load(os.path.join(input_directory, filename), sr=sr)
    spectrogram = np.abs(librosa.stft(audio))
    mel_spectrogram = librosa.feature.melspectrogram(S=spectrogram, sr=sr, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    # Save the mel-spectrogram as a .jpeg image
    os.makedirs(spectrogram_directory, exist_ok=True)
    save_spectrogram(log_mel_spectrogram, os.path.join(spectrogram_directory, f"{os.path.splitext(filename)[0]}.jpeg"))

    data = []
    for i in range(0, log_mel_spectrogram.shape[1] - timesteps, overlap):
        data.append((log_mel_spectrogram[:, i : i + timesteps].T, log_mel_spectrogram[:, i + timesteps]))
    return data

def process_and_save_data():
    input_data = []
    target_data = []

    audio_files = [filename for filename in os.listdir(input_directory) if filename.endswith(".flac")]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for file_data in executor.map(process_file, audio_files):
            for inp, tgt in file_data:
                input_data.append(inp)
                target_data.append(tgt)

    data = list(zip(input_data, target_data))
    create_tfrecord_file(data, tfrecord_filename)

def upload_to_gcs():
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(tfrecord_filename)
    blob.upload_from_filename(tfrecord_filename)

# Set the output path for the TFRecord file
if write_local:
    tfrecord_filename = "audio_data.tfrecord"
    process_and_save_data()
else:
    gcs_bucket = "gs://your-bucket-name"
    tfrecord_filename = os.path.join(gcs_bucket, "audio_data.tfrecord")
    process_and_save_data()
    upload_to_gcs()
