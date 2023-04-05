import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# Parameters
tfrecord_file = "audio_data.tfrecord"
output_directory = "tfrecord_check"
timesteps = 5000
n_mels = 128
num_examples_to_display = 20

def parse_example(example_proto):
    feature_description = {
        'input': tf.io.FixedLenFeature([timesteps, n_mels], tf.float32),
        'target': tf.io.FixedLenFeature([timesteps, n_mels], tf.float32)
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    input_data = parsed_example['input']
    target_data = parsed_example['target']
    return input_data, target_data

def plot_spectrogram(data, output_path):
    plt.figure(figsize=(10, 4))
    plt.imshow(data.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

os.makedirs(output_directory, exist_ok=True)

dataset = tf.data.TFRecordDataset(tfrecord_file).map(parse_example).take(num_examples_to_display)

for idx, (input_data, target_data) in enumerate(dataset):
    input_data = input_data.numpy()
    target_data = target_data.numpy()

    plot_spectrogram(input_data, os.path.join(output_directory, f"input_{idx}.jpg"))
    plot_spectrogram(target_data, os.path.join(output_directory, f"target_{idx}.jpg"))
