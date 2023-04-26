
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import adam_v2

print("TensorFlow version: ", tf.__version__)
print("Connecting to TPU...")
resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu='node-8',zone='us-central1-f')
strategy = tf.distribute.TPUStrategy(resolver)
print("Done!")
print("Number of accelerators: ", strategy.num_replicas_in_sync)

import os
from transformers import GPT2Tokenizer
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

saved_transformers_folder = 'gs://saved_transformers'
os.makedirs(saved_transformers_folder, exist_ok=True)
saved_transformer_path = f'{saved_transformers_folder}/v1'

loss_history = []

with strategy.scope():
    # Hyperparameters
    # The transformer model currently has TK parameters.
    num_layers = 2
    d_model = 192
    num_heads = 8
    dff = 1024
    input_vocab_size = tokenizer.vocab_size
    target_vocab_size = tokenizer.vocab_size
    max_position_encoding = 1000
    dropout_rate = 0.1
    learning_rate = 1e-3
    batch_size = 64
    epochs = 3

    # Create the transformer model
    # Load the weights if the saved model exists
    if os.path.exists(saved_transformer_path):
        print('Loading the saved model')
        model = tf.keras.models.load_model(saved_transformer_path, custom_objects={'Transformer': Transformer})
        if model is None:
            raise Exception('Failed to load the saved model')
        transformer: Transformer = model
        print('Done')
    else:
        print('Loaded model not found. Creating a new model')
        transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_position_encoding, dropout_rate)
        print('Done')


    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    optimizer = adam_v2.Adam(learning_rate)

bucket_path = 'gs://dataset_w/'
input_tfrecord_files = [f'{bucket_path}wikitrain_{i:04d}.tfrecord' for i in range(79)]

# Function to parse a single example from the TFRecord files
def create_windows(sequence, window_size, step=1):
    num_windows = (tf.shape(sequence)[0] - window_size) // step + 1
    windows = tf.TensorArray(dtype=tf.int64, size=num_windows, dynamic_size=True)

    for i in range(num_windows):
        windows = windows.write(i, sequence[i * step:i * step + window_size])

    return windows.stack()

@tf.function
def _parse_function(example_proto):
    feature_description = {
        'token_ids': tf.io.VarLenFeature(tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    input_sequence = tf.sparse.to_dense(parsed_features['token_ids'])
    input_sequences = create_windows(input_sequence, max_position_encoding)

    def process_sequences(seq):
        inp = seq[:-1]
        tar = seq[1:]  # Get the last element as the target
        return inp, tar

    input_sequences_with_start, target_sequences = tf.map_fn(process_sequences, input_sequences, dtype=(tf.int64, tf.int64))

    return input_sequences_with_start, target_sequences


# Load and preprocess the dataset from the TFRecord files
def load_dataset(input_files):
    input_ds = tf.data.TFRecordDataset(input_files)
    input_ds = input_ds.map(_parse_function)
    return input_ds

def print_sequences_as_words(inp, tar):
    inp_tokens = tokenizer.batch_decode(inp.numpy(), skip_special_tokens=True)
    tar_tokens = tokenizer.batch_decode([tar.numpy()], skip_special_tokens=True)

    print("Input:")
    for seq in inp_tokens:
        print(seq)

    print("\nTarget:")
    for seq in tar_tokens:
        print(seq)

print('Processing dataset...')
input_dataset = load_dataset(input_tfrecord_files)
input_dataset = input_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

# def print_dataset(input_dataset, num_examples=1):
#     for i, (inp, tar) in enumerate(input_dataset.take(num_examples)):
#         print(f"Example {i + 1}:")
#         print("Input: ", inp.numpy())
#         print("Target: ", tar.numpy())
#         print_sequences_as_words(inp, tar)
#         print("\n")
# print_dataset(input_dataset)
dataset = input_dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size, drop_remainder=True)

dataset = strategy.experimental_distribute_dataset(dataset)
print('Done!')

import os
import matplotlib.pyplot as plt
import datetime
from IPython.display import clear_output

def plot_loss(loss_history):
    clear_output(wait=True)  # Clear the output before plotting a new graph
    plt.plot(loss_history)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    
    # Get the current timestamp
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.title(f"Loss History\nLast Updated at: {now}")
    
    plt.show()

@tf.function
def train_step(inp, tar):
    with tf.GradientTape() as tape:
        predictions = transformer([inp, tar], training=True)
        loss = loss_function(tar, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    return loss

try:
    for epoch in range(epochs):
        total_loss = tf.constant(0.0, dtype=tf.float32)
        for (batch, (inp, tar)) in enumerate(dataset):
            per_replica_losses = strategy.run(train_step, args=(inp, tar))
            loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            total_loss += loss
            loss = loss.numpy()
            
            if loss < 0.64:
                tf.saved_model.save(transformer, saved_transformer_path + "_good", options=tf.saved_model.SaveOptions(experimental_io_device='/job:localhost'))
                break

            loss_history.append(loss)  # Save the loss of the current batch
            plot_loss(loss_history)  # Update the loss graph
        break # TODO: remove this line

        tf.saved_model.save(transformer, saved_transformer_path, options=tf.saved_model.SaveOptions(experimental_io_device='/job:localhost'))
        avg_loss = total_loss / (batch + 1)
        print(f'Epoch {epoch + 1}, Average loss: {avg_loss:.4f}')

except Exception as e:
    print(e)
    print("Saving transformer...")
    tf.saved_model.save(transformer, saved_transformer_path, options=tf.saved_model.SaveOptions(experimental_io_device='/job:localhost'))
except KeyboardInterrupt:
    print("Saving transformer...")
    tf.saved_model.save(transformer, saved_transformer_path, options=tf.saved_model.SaveOptions(experimental_io_device='/job:localhost'))

min(loss_history)


