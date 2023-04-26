import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Embedding
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import adam_v2

print("TensorFlow version: ", tf.__version__)
print("Connecting to TPU...")
resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu='node-6',zone='us-central1-f')
strategy = tf.distribute.TPUStrategy(resolver)
print("Done!")
print("Number of accelerators: ", strategy.num_replicas_in_sync)

def positional_encoding(position, d_model):
    angle_rads = tf.range(position, dtype=tf.float32)[:, tf.newaxis] * 1 / tf.pow(10000, (2 * tf.range(0, d_model, dtype=tf.float32)) / d_model)
    angle_rads_even = tf.math.sin(angle_rads[:, 0::2])
    angle_rads_odd = tf.math.cos(angle_rads[:, 1::2])
    angle_rads = tf.stack([angle_rads_even, angle_rads_odd], axis=-1)
    angle_rads = tf.reshape(angle_rads, (-1, d_model))
    pos_encoding = angle_rads[tf.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# Transformer layer (Encoder or Decoder)
class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

# Transformer model
class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_position_encoding, rate=0.1):
        super(Transformer, self).__init__()

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_position_encoding, d_model)
        self.transformer_layers = [TransformerLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.final_layer = Dense(target_vocab_size)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding.output_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        for layer in self.transformer_layers:
            x = layer(x, training, mask)

        logits = self.final_layer(x)
        return logits

with strategy.scope():
    # Hyperparameters
    num_layers = 2
    d_model = 128
    num_heads = 8
    dff = 512
    input_vocab_size = 10000
    target_vocab_size = 10000
    max_position_encoding = 1000
    dropout_rate = 0.1
    learning_rate = 1e-4
    batch_size = 64
    epochs = 10

    # Create the transformer model
    transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_position_encoding, dropout_rate)

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

    input_sequences_with_start = []
    target_sequences = []
    for seq in input_sequences:
        inp_with_start = seq
        target = tf.roll(seq, shift=-1, axis=-1)
        input_sequences_with_start.append(inp_with_start)
        target_sequences.append(target)

    return input_sequences_with_start, target_sequences


# Load and preprocess the dataset from the TFRecord files
def load_dataset(input_files):
    input_ds = tf.data.TFRecordDataset(input_files)
    input_ds = input_ds.map(_parse_function)
    return input_ds


print('Processing dataset...')
input_dataset = load_dataset(input_tfrecord_files)
input_dataset = input_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
dataset = input_dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size, drop_remainder=True)


def print_dataset(input_dataset, num_examples=5):
    for i, (inp, tar) in enumerate(input_dataset.take(num_examples)):
        print(f"Example {i + 1}:")
        print("Input: ", inp.numpy())
        print("Target: ", tar.numpy())
        print("\n")
print_dataset(dataset)

dataset = strategy.experimental_distribute_dataset(dataset)
print('Done!')

@tf.function
def train_step(inp, tar):
    with tf.GradientTape() as tape:
        predictions = transformer(inp, training=True)
        loss = loss_function(tar, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    return loss


for epoch in range(epochs):
    total_loss = tf.constant(0.0, dtype=tf.float32)  # Initialize total_loss as a scalar tensor
    for (batch, (inp, tar)) in enumerate(dataset):
        # Call the train_step function using strategy.run
        per_replica_losses = strategy.run(train_step, args=(inp, tar))
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        total_loss += loss

        # Print batch number and current batch loss
        print(f'Epoch {epoch + 1}, Batch {batch + 1}, Batch Loss: {loss:.4f}')

    print(f'Epoch {epoch + 1}, Loss: {total_loss / (batch + 1):.4f}')
