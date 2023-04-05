import os
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write
import librosa
from google.cloud import storage

# Parameters
output_directory = "generated"
model_directory = "model"
gcs_bucket_name = "knots-audio-processing"
tfrecord_path = os.path.join("gs://", gcs_bucket_name, "audio_data.tfrecord")
timesteps = 5000
n_mels = 128
song_length = 8000  # in timesteps
n_songs = 2
read_local = False
write_local = False
use_tpu = True
sr = 22050
prediction_shift = 10

def check_use_tpu():
    print("Setting up the environment...")
    if use_tpu:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
            print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        except ValueError:
            raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)

    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy("CPU:0")
    return strategy

def parse_example(example_proto):
    feature_description = {
        "input": tf.io.FixedLenFeature([timesteps, n_mels], tf.float32),
        "target": tf.io.FixedLenFeature([n_mels], tf.float32),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    return parsed_features["input"], parsed_features["target"]

def audio_data_generator(tfrecord_path, batch_size=1):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset

def save_to_gcs(bucket_name, file_path, destination_path):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(file_path)
    os.remove(file_path)


if __name__ == '__main__':
    strategy = check_use_tpu()

    if not read_local:
        model_directory = os.path.join("gs://", gcs_bucket_name, model_directory)
        model = tf.keras.models.load_model(model_directory)
    else:
        model = tf.keras.models.load_model(model_directory)

    if not write_local:
        client = storage.Client()
    else:
        os.makedirs(output_directory, exist_ok=True)

    print("Generating new songs...")

    global_batch_size = 1 * strategy.num_replicas_in_sync
    dist_dataset = strategy.experimental_distribute_datasets_from_function(
        lambda _: audio_data_generator(tfrecord_path, global_batch_size)
    )

    for i in range(n_songs):
        input_data = next(iter(dist_dataset))
        seed = input_data[0].values[0].numpy()[0]

        generated_spectrogram = []

        for step in range(song_length):
            if step % prediction_shift == 0:
                prediction = model.predict(seed.reshape(1, timesteps, n_mels))
                generated_spectrogram.extend(prediction[:prediction_shift])
                seed = np.vstack((seed[prediction_shift:], prediction[:prediction_shift]))

        generated_mel_spectrogram = np.array(generated_spectrogram).T
        generated_power_spectrogram = librosa.db_to_power(generated_mel_spectrogram)
        generated_spectrogram = librosa.feature.inverse.mel_to_stft(generated_power_spectrogram, sr=sr)
        generated_audio = librosa.griffinlim(generated_spectrogram)

        if write_local:
            output_filename = os.path.join(output_directory, f"generated_song_{i + 1}.wav")
            write(output_filename, sr, generated_audio.astype(np.float32))
        else:
            output_filename = f"generated_song_{i + 1}.wav"
            write(output_filename, sr, generated_audio.astype(np.float32))
            destination_path = os.path.join(output_directory, output_filename)
            save_to_gcs(gcs_bucket_name, output_filename, destination_path)

    print(f"Generated {n_songs} songs in the '{output_directory}' directory.")
