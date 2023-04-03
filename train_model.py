import os
import tensorflow as tf
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.models import Sequential

# Parameters
read_local = False
model_directory = "model"
timesteps = 50
n_mels = 128
epochs = 10
batch_size = 2048
use_tpu = True

if read_local:
    tfrecord_file = "audio_data.tfrecord"
else:
    gcs_bucket = "gs://knots-audio-processing"
    tfrecord_file = os.path.join(gcs_bucket, "audio_data.tfrecord")


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


print("Building and training the model...")
with strategy.scope():
    model = Sequential()
    model.add(LSTM(2048, input_shape=(timesteps, n_mels), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(2048, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(2048))
    model.add(Dense(n_mels))
    model.add(Activation("linear"))

    model.compile(optimizer="adam", loss="mse")
    
    # Load the dataset from the TFRecord file
    def parse_example(example_proto):
        feature_description = {
            'input': tf.io.FixedLenFeature([timesteps, n_mels], tf.float32),
            'target': tf.io.FixedLenFeature([n_mels], tf.float32)
        }
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        return parsed_example['input'], parsed_example['target']

    dataset = tf.data.TFRecordDataset(tfrecord_file).map(parse_example)
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).repeat()

    # Train the model
    model.fit(dataset, epochs=epochs, steps_per_epoch=100)

print("Saving the model...")
if use_tpu and not read_local:
    model_directory = os.path.join(gcs_bucket, model_directory)

os.makedirs(model_directory, exist_ok=True)
model.save(model_directory)