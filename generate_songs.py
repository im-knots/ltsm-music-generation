import os
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write
from keras.models import load_model
import librosa
from google.cloud import storage

# Parameters
output_directory = "generated"
model_directory = "model"
timesteps = 50
n_mels = 128
song_length = 1000  # in timesteps
n_songs = 2
read_local = False
write_local = False
use_tpu = True

if not read_local:
    gcs_bucket = "gs://knots-audio-processing"
    model_directory = os.path.join(gcs_bucket, model_directory)

if not write_local:
    gcs_bucket_name = "knots-audio-processing"
    output_directory = os.path.join("gs://", gcs_bucket_name, output_directory)
    client = storage.Client()

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

# Load the saved model
model = load_model(model_directory)

def save_to_gcs(bucket_name, file_path, destination_path):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(file_path)
    os.remove(file_path)


print("Generating new songs...")
os.makedirs(output_directory, exist_ok=True)

for i in range(n_songs):
    input_data, _ = next(audio_data_generator(batch_size=1))
    seed = input_data[0]

    generated_spectrogram = []

    for _ in range(song_length):
        prediction = model.predict(seed.reshape(1, timesteps, n_mels))
        generated_spectrogram.append(prediction[0])
        seed = np.vstack((seed[1:], prediction))

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
