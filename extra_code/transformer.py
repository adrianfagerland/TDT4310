# import tensorflow as tf
# from tensorflow.python.keras.layers import Dense, Dropout, Embedding, Layer
# from tensorflow.python.keras.models import Model

# def positional_encoding(position, d_model):
#     angle_rads = tf.range(position, dtype=tf.float32)[:, tf.newaxis] * 1 / tf.pow(10000, (2 * tf.range(0, d_model, dtype=tf.float32)) / d_model)
#     angle_rads_even = tf.math.sin(angle_rads[:, 0::2])
#     angle_rads_odd = tf.math.cos(angle_rads[:, 1::2])
#     angle_rads = tf.stack([angle_rads_even, angle_rads_odd], axis=-1)
#     angle_rads = tf.reshape(angle_rads, (-1, d_model))
#     pos_encoding = angle_rads[tf.newaxis, ...]
#     return tf.cast(pos_encoding, dtype=tf.float32)

# # Encoder layer
# class EncoderLayer(Layer):
#     def __init__(self, d_model, num_heads, dff, rate=0.1):
#         super(EncoderLayer, self).__init__()

#         self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
#         # When using MultiHeadAttention inside a custom layer, the custom layer must implement its own build() method and call MultiHeadAttention's _build_from_signature() there. This enables weights to be restored correctly when the model is loaded.
#         self.ffn = tf.keras.Sequential([
#             Dense(dff, activation='relu'),
#             Dense(d_model)
#         ])

#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

#         self.dropout1 = Dropout(rate)
#         self.dropout2 = Dropout(rate)

#     def call(self, x, training, mask):
#         attn_output = self.mha(x, x, x, attention_mask=mask)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(x + attn_output)

#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         out2 = self.layernorm2(out1 + ffn_output)

#         return out2
    

# # Decoder layer
# class DecoderLayer(Layer):
#     def __init__(self, d_model, num_heads, dff, rate=0.1):
#         super(DecoderLayer, self).__init__()

#         self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
#         self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

#         self.ffn = tf.keras.Sequential([
#             Dense(dff, activation='relu'),
#             Dense(d_model)
#         ])

#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

#         self.dropout1 = Dropout(rate)
#         self.dropout2 = Dropout(rate)
#         self.dropout3 = Dropout(rate)

#     def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
#         attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
#         attn1 = self.dropout1(attn1, training=training)
#         out1 = self.layernorm1(attn1 + x)

#         attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
#         attn2 = self.dropout2(attn2, training=training)
#         out2 = self.layernorm2(attn2 + out1)

#         ffn_output = self.ffn(out2)
#         ffn_output = self.dropout3(ffn_output, training=training)
#         out3 = self.layernorm3(ffn_output + out2)

#         return out3

# # Encoder
# class Encoder(Layer):
#     def __init__(self, num_layers, d_model, num_heads,
#         dff, input_vocab_size, maximum_position_encoding, rate=0.1):
#         super(Encoder, self).__init__()

#         self.d_model = d_model
#         self.num_layers = num_layers

#         self.embedding = Embedding(input_vocab_size, d_model)
#         self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

#         self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
#         self.dropout = Dropout(rate)

#     def call(self, x, training, mask):
#         seq_len = tf.shape(x)[1]

#         x = self.embedding(x)
#         x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#         x += self.pos_encoding[:, :seq_len, :]

#         x = self.dropout(x, training=training)

#         for i in range(self.num_layers):
#             x = self.enc_layers[i](x, training, mask)

#         return x

# # Decoder
# class Decoder(Layer):
#     def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
#         super(Decoder, self).__init__()

#         self.d_model = d_model
#         self.num_layers = num_layers

#         self.embedding = Embedding(target_vocab_size, d_model)
#         self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

#         self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
#         self.dropout = Dropout(rate)

#     def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
#         seq_len = tf.shape(x)[1]
#         attention_weights = {}
#         # TODO

#         x = self.embedding(x)
#         x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#         x += self.pos_encoding[:, :seq_len, :]

#         x = self.dropout(x, training=training)

#         for i in range(self.num_layers):
#             x = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

#         return x

# def create_padding_mask(seq):
#     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# def create_look_ahead_mask(size):
#     mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#     return mask  # (seq_len, seq_len)

# # Transformer model
# class Transformer(Model):
#     def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_position_encoding, rate=0.1):
#         super(Transformer, self).__init__()

#         self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_position_encoding, rate)
#         self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_position_encoding, rate)

#         self.final_layer = Dense(target_vocab_size)

#     def create_masks(self, inp):
#         enc_padding_mask = create_padding_mask(inp)
#         look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
#         dec_padding_mask = enc_padding_mask

#         return enc_padding_mask, look_ahead_mask, dec_padding_mask

#     def call(self, inputs, training):
#         inp, tar = inputs
#         enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp)

#         enc_output = self.encoder(inp, training, enc_padding_mask)
#         dec_output = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

#         final_output = self.final_layer(dec_output)

#         return final_output
# 1. Load the pre-trained GPT-2 model and tokenizer
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
config = GPT2Config.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")