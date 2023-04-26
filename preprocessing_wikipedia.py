from transformers import GPT2Tokenizer
import tensorflow as tf
import jsonlines
import concurrent.futures
from itertools import islice
import os

def tokenize_and_encode(text):
    global tokenizer
    tokens = tokenizer.tokenize(text + tokenizer.eos_token)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids

def read_jsonl_articles(file_path):
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            article_text = " \n ".join(section["text"] for section in obj["sections"])
            yield article_text

def save_token_ids_as_tfrecord(output_file_prefix, token_ids_list, shard_index):
    shard_file = f"{output_file_prefix}_{shard_index:04d}.tfrecord"
    writer = tf.io.TFRecordWriter(shard_file)

    for token_ids in token_ids_list:
        # Save token IDs as a record in the current shard
        feature = _int64_feature(token_ids)
        features = tf.train.Features(feature={'token_ids': feature})
        example = tf.train.Example(features=features)
        serialized_example = example.SerializeToString()

        writer.write(serialized_example)

    writer.close()

def process_jsonl_parallel(input_file, output_file_prefix, batch_size, minimum_mb_file_size):
    global tokenizer
    jsonl_reader = read_jsonl_articles(input_file)
    shard_index = 0
    accumulated_token_ids = []
    accumulated_size = 0
    min_file_size_bytes = minimum_mb_file_size * 1024 * 1024

    with concurrent.futures.ProcessPoolExecutor() as executor:
        while True:
            batch_texts = list(islice(jsonl_reader, batch_size))
            if not batch_texts:
                break

            # Tokenize and encode the batch of texts in parallel
            batch_token_ids = list(executor.map(tokenize_and_encode, batch_texts))

            # Accumulate token IDs and their sizes
            for token_ids in batch_token_ids:
                if accumulated_size < min_file_size_bytes:
                    accumulated_token_ids.append(token_ids)
                    accumulated_size += len(token_ids)
                else:
                    # Save accumulated token IDs if the article is already above the desired size
                    if accumulated_token_ids:
                        save_token_ids_as_tfrecord(output_file_prefix, accumulated_token_ids, shard_index)
                        shard_index += 1
                        accumulated_token_ids = []
                        accumulated_size = 0

                    # Save the article if it's already above the desired size
                    if len(token_ids) >= min_file_size_bytes:
                        save_token_ids_as_tfrecord(output_file_prefix, [token_ids], shard_index)
                        shard_index += 1
                    else:
                        # Start accumulating a new article
                        accumulated_token_ids.append(token_ids)
                        accumulated_size += len(token_ids)

    # Save any remaining accumulated token IDs
    if accumulated_token_ids:
        save_token_ids_as_tfrecord(output_file_prefix, accumulated_token_ids, shard_index)

# Initialize the tokenizer outside the main function
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.vocab_size

# Define the TFRecord feature
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# Define input and output file paths and batch size
input_file = "data/link_annotated_text.jsonl"
output_file_prefix = "gs://dataset_wiki/"
batch_size = 100
minimum_mb_file_size = 40

# Create the output directory if it doesn't exist
os.makedirs("p", exist_ok=True)

# Run the parallel processing
process_jsonl_parallel(input_file, output_file_prefix, batch_size, minimum_mb_file_size)
