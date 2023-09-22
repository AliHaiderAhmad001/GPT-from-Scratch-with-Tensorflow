# GenesisMind-Building-GPT-from-Scratch

GPT-1 "Generative Pre-trained Transformer" is the first version of the GPT series of models, revolutionized natural language processing with its autoregressive language modeling capabilities built on the Transformer architecture. This repository serves as a comprehensive guide to understanding and implementing the GPT model. I'm gonna walk through the essential components, techniques, and principles behind GPT.

**Note:** This is a mini-GPT model for text generation. The original model is very large and deals with massive data, so of course here we will not be using the same dataset or even the same size of model. However, you can control the parameters or even modify it easily if necessary to match the original model.

## The environment
We are creating this demo within Jupiter Notebook, so first of all don't forget to go to the working folder:
```
%cd '/content/drive/MyDrive/Colab Notebooks/projects/GenesisMind-Building-GPT-from-Scratch'
```

## Download requirements
We'll need the transformer pack just for the tokenizer.
```
!pip install transformers
```

## Download Dataset
To test the validity of our work, we will work with a relatively small dataset of 50,000 samples. IMDB popular movie dataset.

```
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```

## Prepare the dataset
All we need to do is create another folder in which we place a specific number of samples for validation. Of course you can change that. We will use 5000 samples for validation.

```
import os
import shutil
import random
import multiprocessing

# Define a function for moving files from one directory to another
def move_files(src_files, dest_dir):
    for file_to_move in src_files:
        destination_path = os.path.join(dest_dir, os.path.basename(file_to_move))
        try:
            shutil.move(file_to_move, destination_path)
        except Exception as e:
            print(f"Error moving file {file_to_move}: {str(e)}")

# Define your source and validation directories
test_dir = "aclImdb/test"
validation_dir = "aclImdb/valid"
num_files_to_move = 2500

# Create the validation directory if it doesn't exist
os.makedirs(validation_dir, exist_ok=True)

# Create separate 'pos' and 'neg' subdirectories within the validation directory
validation_pos_dir = os.path.join(validation_dir, 'pos')
validation_neg_dir = os.path.join(validation_dir, 'neg')
os.makedirs(validation_pos_dir, exist_ok=True)
os.makedirs(validation_neg_dir, exist_ok=True)

# Define the subdirectories
test_pos_dir = os.path.join(test_dir, 'pos')
test_neg_dir = os.path.join(test_dir, 'neg')

# List files in the test 'pos' and 'neg' directories
test_pos_files = [os.path.join(test_pos_dir, filename) for filename in os.listdir(test_pos_dir)]
test_neg_files = [os.path.join(test_neg_dir, filename) for filename in os.listdir(test_neg_dir)]

# Randomly shuffle the lists of files
random.seed(42)  # Set a random seed for reproducibility
random.shuffle(test_pos_files)
random.shuffle(test_neg_files)

# Split the files into chunks for parallel processing
chunk_size = num_files_to_move // multiprocessing.cpu_count()
test_pos_chunks = [test_pos_files[i:i + chunk_size] for i in range(0, num_files_to_move, chunk_size)]
test_neg_chunks = [test_neg_files[i:i + chunk_size] for i in range(0, num_files_to_move, chunk_size)]

# Create a pool of worker processes
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

# Move files in parallel
pool.starmap(move_files, [(chunk, validation_pos_dir) for chunk in test_pos_chunks])
pool.starmap(move_files, [(chunk, validation_neg_dir) for chunk in test_neg_chunks])

# Close the pool of worker processes
pool.close()
pool.join()
```

## Byte-Pair Encoding (BPE)

The tokenizer used in GPT-2 is the same as the one used in GPT-1. Both GPT-1 and GPT-2 use the Byte-Pair Encoding (BPE) tokenizer, which breaks down words into subword units based on their frequency in the training data. This allows the tokenizer to handle out-of-vocabulary words and create a more compact vocabulary.

The Hugging Face Transformers library provides a unified interface for various models, including GPT-1 and GPT-2. When you use the AutoTokenizer class from the library, it loads the appropriate tokenizer based on the model name you provide. Since GPT-1 and GPT-2 share the same tokenizer architecture, you can use the same tokenizer for both models.

We can build the tokenizer from scratch or we can use a ready-made implementation of it. I don't like the idea of rebuilding it at all, it's a tedious and complex process, and it seems off topic. Common practice is to reuse and adapt the same implementation on a new dataset or corpus.

It should be noted that tokenizer training continues until a certain number of iterations is reached, a certain number of vocabulary are reached, convergence is reached, or until the algorithm finds no new pairs to combine (in the latter case we get all the unique vocabulary in the corpus). That is, there are 4 different ways the training can be stopped. In this repo, we will use the most popular method, the second method (when training from scratch, I think the third method is the best, but it is more tiring than the other methods).

We already know that the GPT2 model has a vocab size of 50,257 I think, so I can stop the algorithm when I get to 50,357. I think that will suffice and more since we are dealing with a relatively small and familiar data set. Actually, we never need to retrain the tokenizer for this dataset, but it's nice to get our hands dirty sometimes. However, you can still skip this step and use the tokenizer without training it.

I will not explain the details of the algorithm's work, but [here](https://leimao.github.io/blog/Byte-Pair-Encoding/) there is a clear explanation.

```
import os
import argparse
import concurrent.futures
from transformers import AutoTokenizer
from abc import ABCMeta, abstractmethod

class BPETrainer():
    def __init__(self, model="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def train(self, data_dir, batch_size=1000, vocab_size=50357, save=True, save_fp='tokenizer/adapted-tokenizer'):
        """
        Retrains GPT tokenizer on a new corpus.

        Args:
            data_dir (str): Corpus directory path.
            batch_size (int): Batch size for reading files. Default is 1000.
            vocab_size (int): Target vocabulary size for adapted tokenizer. Default is 50000.
            save (bool): Whether to save the adapted tokenizer. Default is True.
            save_fp (str): File path to save the adapted tokenizer. Default is 'tokenizer/adapted-tokenizer'.
        """
        training_corpus = self.read_batch_of_files(data_dir, batch_size)
        self.tokenizer = self.tokenizer.train_new_from_iterator(training_corpus, vocab_size)
        if save:
            self.save(save_fp)

    def tokenize(self, sentence):
        return self.tokenizer.encode(sentence)

    def read_batch_of_files(self, data_dir, batch_size, num_workers=4):
        filenames = self.get_filenames(data_dir)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for start_idx in range(0, len(filenames), batch_size):
                batch_filenames = filenames[start_idx : start_idx + batch_size]
                batch_contents = []
                future_to_filename = {executor.submit(self.read_file, filename): filename for filename in batch_filenames}
                for future in concurrent.futures.as_completed(future_to_filename):
                    filename = future_to_filename[future]
                    content = future.result()
                    batch_contents.append(content)

                yield batch_contents

    def get_filenames(self, data_dir):
        filenames = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                filenames.append(os.path.join(root, file))
        return filenames

    def read_file(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    def save(self, fp):
        self.tokenizer.save_pretrained(fp)

def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a new corpus.")
    parser.add_argument("data_dir", type=str, help="Corpus directory path")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for reading files (default: 1000)")
    parser.add_argument("--vocab_size", type=int, default=50357, help="Target vocabulary size (default: 50357)")
    parser.add_argument("--save", action="store_true", help="Save the adapted tokenizer")
    parser.add_argument("--save_fp", type=str, default="tokenizer/adapted-tokenizer", help="File path to save the tokenizer (default: 'tokenizer/adapted-tokenizer')")

    args = parser.parse_args()

    bpe_trainer = BPETrainer()
    bpe_trainer.train(data_dir=args.data_dir, batch_size=args.batch_size, vocab_size=args.vocab_size, save=args.save, save_fp=args.save_fp)

if __name__ == "__main__":
    main()
```

## Tokenizer

The provided code defines two classes for tokenizing sentences using the Hugging Face Transformers library:

1. **`DataTokenizer` Abstract Base Class (ABC):**

    This abstract class serves as a base for all tokenization classes and defines the interface for tokenizing sentences. It includes an abstract method `tokenize` that must be implemented by any derived class. This method takes an input sentence and returns tokenized input IDs.

2. **`EnglishDataTokenizer` Class:**

    This class is an implementation of the `DataTokenizer` abstract class specifically tailored for tokenizing English sentences. It initializes a tokenizer from a given path or name of a pre-trained tokenizer using the Hugging Face `AutoTokenizer`. The `tokenize` method of this class tokenizes a given sentence, ensuring it's padded to a maximum length specified during initialization. The tokenized input IDs are returned.


**Note:** In GPT models, the text is usually generated without specific markers as the beginning of a sequence, and there may not be a dedicated token for unknown words or subwords because subword tokenization is often capable of handling most text. Keep in mind that the absence of certain special tokens doesn't limit the model's ability to generate coherent text. Instead, it focuses on the language generation aspect. If you have specific requirements for using additional special tokens, you might need to adapt or extend the tokenizer accordingly.

```
from transformers import AutoTokenizer, logging
from abc import ABCMeta, abstractmethod

# Set the logging level for transformers library to ERROR
logging.set_verbosity(logging.ERROR)

class DataTokenizer(metaclass=ABCMeta):
    """
    A wrapper class for tokenizing sentences using the Hugging Face transformers library.
    """

    @abstractmethod
    def tokenize(self, sentence):
        """
        Tokenizes a sentence and converts it to token IDs.

        Args:
            sentence (str): The input sentence to be tokenized.

        Returns:
            List[int]: List of token IDs.
        """

    @abstractmethod
    def detokenize(self, input_ids):
        """
        Converts token IDs back to a sentence.

        Args:
            input_ids (List[int]): List of token IDs.

        Returns:
            str: Detokenized sentence.
        """


class EnglishDataTokenizer(DataTokenizer):
    def __init__(self, tokenizer_path, sequence_length, training=True):
        """
        Initializes the EnglishDataTokenizer.

        Args:
            tokenizer_path (str): The name or path of the pre-trained tokenizer.
            sequence_length (int): The maximum length of tokenized sequences.
            training (bool): Whether the tokenizer is used for training or not.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, pad_token="[PAD]")
        self.sequence_length = sequence_length + 1 if training else sequence_length
        self.padding = 'max_length' if training else 'do_not_pad'
        self.truncation = True if training else False
        self.training = training

    def tokenize(self, sentence):
        if self.training:
            sentence += '<|endoftext|>'
        tokenized_sentence = self.tokenizer(sentence,
                                            padding=self.padding,
                                            max_length=self.sequence_length, 
                                            truncation=self.truncation,
                                            return_tensors="np")
        return tokenized_sentence['input_ids'][0]

    def detokenize(self, input_ids):
        return self.tokenizer.decode(input_ids)

    def convert_to_tokens(self, sentence):
        if self.training:
            sentence += '<|endoftext|>'
            tokenized_sentence = self.tokenizer.tokenize(sentence,
                                                         padding=self.padding,
                                                         max_length=self.sequence_length,
                                                         truncation=self.truncation)

        return tokenized_sentence

    def convert_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def tokenize2(self, sentence):
        tokens = self.convert_to_tokens(sentence)
        input_ids = self.convert_to_ids(tokens)

        return input_ids
```

Testing:

```
tokenizer_path = "tokenizer/adapted-tokenizer"
max_length = 5

# Create a Tokenizer instance
tokenizer = EnglishDataTokenizer(tokenizer_path, max_length)

# Test sentence
test_sentence = "This movie is"
tokenized_input_ids = tokenizer.tokenize(test_sentence)
sentence  = tokenizer.detokenize(tokenized_input_ids)
tokenized_input_ids2 = tokenizer.tokenize2(test_sentence)
sentence2  = tokenizer.detokenize(tokenized_input_ids2)

# Print the tokenized input IDs
print("Tokenized Input IDs:", tokenized_input_ids)
print("Sentence:", sentence)

print("Tokenized Input IDs2:", tokenized_input_ids2)
print("Sentence2:", sentence2)
```

## Data Stremer

**Note:** First of all you might be wondering why I didn't use `tf.data` here. The reason is that you can't combine it and the tokenizer from the Hugging Face library into a single pipeline. There is one solution that can be used but I will not go into it, because it is ineffective. That's why I'm building a pipeline from scratch that follows some useful practices found in `tf.data`. It should be noted that this code uses parallelization, which may have a subtle effect (and may even slow down) in the case of small corps like ours.

This code defines a data streaming framework for loading and iterating over tokenized data sequences in batches. It includes an abstract base class `DataStreamer` that outlines the required methods and their functionalities. It also provides a concrete implementation called `EnglishDataStreamer` tailored for English data.

The `DataStreamer` ABC includes the following methods:
- `__len__()`: Calculates the number of batches in the dataset.
- `_fetch_to_buffer()`: Fetches data from files into an internal buffer using parallelization.
- `__iter__()`: Returns the iterator object for iterating over batches of data.
- `_start_fetching()`: Starts the background fetching of data.
- `__next__()`: Returns the next batch of tokenized sequences.
- `_get_filenames(data_dir)`: Retrieves a list of file paths from subdirectories.
- `_reset()`: Resets the buffer index and fetches new data if necessary.

The `EnglishDataStreamer` class, which inherits from `DataStreamer`, implements the specifics of the abstract methods and adds functionality for handling English data. It takes various arguments like the root directory of data files, tokenizer information, batch size, maximum sequence length, and more.

Key features of `EnglishDataStreamer`:
- It maintains an internal buffer to efficiently load and manage tokenized sequences.
- Parallelization is used to fetch and process data from files into the buffer.
- The `__next__()` method prepares and returns batches of input sequences and their corresponding labels, suitable for training a language model.

```
import string
import os, re
import threading
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor

class DataStreamer(metaclass=ABCMeta):
    """
    A data streamer for loading and iterating over tokenized data sequences in batches.

    Methods:
        __len__(): Returns the number of batches in the dataset.
        _fetch_to_buffer(): Fetches data from files into the buffer.
        __iter__(): Returns the iterator object.
        __next__(): Returns the next batch of tokenized sequences.
        _get_filenames(data_dir): Retrieves a list of file paths from subdirectories.
        _reset(): Resets the buffer index and fetches new data if necessary.
    """

    @abstractmethod
    def __len__(self):
        """
        Calculates the number of batches in the dataset.

        Returns:
            int: Number of batches.
        """

    @abstractmethod
    def fetch_to_buffer(self):
        """
        Fetches data from files into the internal buffer using parallelization.
        """

    @abstractmethod
    def __iter__(self):
        """
        Returns the iterator object for iterating over batches of data.
        """

    @abstractmethod
    def __next__(self):
        """
        Retrieves the next batch of sentences from the buffer.

        Returns:
            list: Batch of sentences.
        Raises:
            StopIteration: If there is no more data to retrieve.
        """

    @abstractmethod
    def get_filenames(self):
        """
        Retrieves a list of file paths from subdirectories within the given root directory.

        Returns:
            list: List of file paths.
        """

    @abstractmethod
    def reset(self):
        """
        Resets the buffer index and fetches new data if necessary.
        """

class EnglishDataStreamer(DataStreamer):
    """
    A specialized implementation of DataStreamer for handling English language data.

    This class extends the functionality of the DataStreamer class to specifically
    handle English language data. It provides methods for loading and processing
    English text data for further use in natural language processing tasks.

    Args:
        config (object): Configuration object containing relevant parameters.

    Attributes:
        buffer_idx (int): Index pointing to the current position in the buffer.
        buffer (list): List to hold fetched data for efficient batching.
        ptr (int): Pointer for data processing within the buffer.
        flag (bool): Flag indicating the status of data fetching.
        dataset_type (str): Either "train" or "valid".
        data_dir (str): dataset diractory.
        buffer_size (int): Size of the buffer for holding fetched data.
        batch_size (int): Size of each batch of data to be processed.
        tokenizer_path (str): Path to the tokenizer used for text processing.
        sequence_length (int): Maximum length of sequences after tokenization.
        shuffle (bool): Flag indicating whether to shuffle the data.
        lower_case (bool): Flag indicating whether to convert text to lowercase.
        remove_punctuation (bool): to remove punctuation or not.
        random_state: Random state generator for reproducibility.
        filenames (list): List of file names containing the data.
        tokenizer: Instance of the EnglishDataTokenizer for text processing.
    """

    def __init__(self, config, dataset_type):
        assert config.buffer_size >= config.batch_size, "buffer_size should be equal or greater than batch_size"

        # Initialize attributes
        self.buffer_idx = 0
        self.buffer = []
        self.ptr = 0
        self.flag = False
        self.dataset_type = dataset_type
        self.data_dir = config.data_dir
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size
        self.tokenizer_path = config.tokenizer_path
        self.sequence_length = config.sequence_length
        self.shuffle = config.shuffle
        self.lower_case = config.lower_case
        self.remove_punctuation = config.remove_punctuation
        self.random_state = np.random.RandomState(config.seed)
        self.filenames = self.get_filenames()
        self.tokenizer = EnglishDataTokenizer(config.tokenizer_path, config.sequence_length)
        self.fetch_to_buffer()

    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def fetch_to_buffer(self):
        #print("Fetching...")
        self.buffer = self.buffer[self.ptr:]
        self.ptr = 0

        def custom_standardization(sentence):
            """ Remove html line-break tags and lowercasing """
            sentence = re.sub("<br />", " ", sentence).strip()
            if self.lower_case:
                sentence = sentence.lower()
            # remove punctuation
            if self.remove_punctuation:
                sentence = re.sub(f"[{re.escape(string.punctuation)}]", r" ", sentence)
            return sentence

        def process_data(filename):
            """ Read the data and perform the necessary processing and conversion operations """
            with open(filename, "r") as file:
                sentence = file.readline()
                sentence = custom_standardization(sentence)
                try:
                    tokenized_sentence = self.tokenizer.tokenize(sentence)
                except:
                    print(sentence)
                    raise ValueError("MyError")
                return tokenized_sentence

        with ThreadPoolExecutor() as executor:
            sentences = sentences = list(executor.map(process_data, self.filenames[self.buffer_idx:self.buffer_idx + self.buffer_size]))

        self.buffer.extend(sentences)
        self.buffer_idx += self.buffer_size

        if self.shuffle:
            self.random_state.shuffle(self.buffer)

        #print("Buffer size:", len(self.buffer))
        #print("Buffer elements:", self.buffer)
        #print("Fetching is complete")

    def __iter__(self):
        return self

    def __next__(self):
        """
        Retrieves the next batch of sentences from the buffer.

        Returns:
            list: Batch of sentences.
        Raises:
            StopIteration: If there is no more data to retrieve.
        """
        def prepare_lm_inputs_labels(batch):
            """
            Shift word sequences by 1 position so that the target for position (i) is
            word at position (i+1). The model will use all words up till position (i)
            to predict the next word.
            """
            batch = tf.convert_to_tensor(batch)
            x = batch[:, :-1]
            y = batch[:, 1:]
            return [x, y]

        if self.flag:
            self.flag = False
            self.reset()
            raise StopIteration

        if self.ptr + self.batch_size > self.buffer_size:
            self.fetch_to_buffer()

        batch = self.buffer[self.ptr:self.ptr + self.batch_size]
        self.ptr += self.batch_size

        if len(batch) < self.batch_size and self.buffer_idx >= len(self.filenames):
            self.flag = True

        return prepare_lm_inputs_labels(batch)



    def get_filenames(self):
        if self.dataset_type not in ["train", "valid"]:
            raise ValueError("Invalid dataset type. Choose 'train' or 'valid'.")

        base_dir = self.data_dir

        if self.dataset_type == "train":
            dataset_dirs = ["train", "test"]
        else:
            dataset_dirs = [self.dataset_type]

        all_files = []

        for dataset_dir in dataset_dirs:
            dataset_dir = os.path.join(base_dir, dataset_dir)
            pos_dir = os.path.join(dataset_dir, "pos")
            neg_dir = os.path.join(dataset_dir, "neg")

            pos_files = [os.path.join(pos_dir, filename) for filename in os.listdir(pos_dir)]
            neg_files = [os.path.join(neg_dir, filename) for filename in os.listdir(neg_dir)]

            # Combine positive and negative file lists
            all_files.extend(pos_files)
            all_files.extend(neg_files)

        return all_files

    def reset(self):
        self.buffer_idx = 0
        self.fetch_to_buffer()

```

Testing:

```

#  Here we create a dicactory that matches the dicactory structure we have and put only 7 files in it,
#  each with the letter 't' alongside the corresponding file number. Ex: t1 for the first one,
#  t2 for the second one, and so on. Then we run tests to make sure the streamer works as expected.


class ConfigDr:
    def __init__(self):
        self.tokenizer_path = "tokenizer/adapted-tokenizer"
        self.data_dir = 'dummy_data'
        self.checkpoint_filepath = 'tmp/checkpoint'
        self.shuffle = True
        self.lower_case = True
        self.sequence_length = 3
        self.buffer_size = 4
        self.batch_size = 2
        self.seed = 2024
        self.remove_punctuation = True

config_dr = ConfigDr()
data_streamer = EnglishDataStreamer(config_dr, 'train')

for _ in range(2):
    for batch_num, batch_data in enumerate(data_streamer):
        print(f"Batch {batch_num + 1}:")
        #print(batch_data[0].shape)
        #print((batch_data))
        #break
        print("\n----------------------\n")

    print("\nEnd of epoch \n")"""
```

## GPT Building Blocks
Most of the following components in this section were discussed earlier in [Neural-Machine-Translator](https://github.com/AliHaiderAhmad001/Neural-Machine-Translator/blob/main/README.md) repo (in the demo folder). Almost the only difference is the removal of the encoder and the modification of the decoder a bit. So we won't talk about the next layers anymore.

### Embeddings
```
import tensorflow as tf

class PositionalEmbeddings(tf.keras.layers.Layer):
    """
    PositionalEmbeddings layer.

    This layer generates positional embeddings based on input IDs.
    It uses an Embedding layer to map position IDs to position embeddings.

    Args:
        config (object): Configuration object containing parameters.
    """

    def __init__(self, config, name = None, **kwargs):
        super(PositionalEmbeddings, self).__init__(name=name, **kwargs)
        self.positional_embeddings = tf.keras.layers.Embedding(
            input_dim=config.sequence_length, output_dim=config.hidden_size
        )

    def call(self, input_ids):
        """
        Generate positional embeddings.

        Args:
            input_ids (tf.Tensor): Input tensor containing token IDs.

        Returns:
            tf.Tensor: Positional embeddings tensor of shape (batch_size, seq_length, hidden_size).
        """
        seq_length = tf.shape(input_ids)[1]
        position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        position_embeddings = self.positional_embeddings(position_ids)
        return position_embeddings

    def get_config(self):
        """
        Get the layer configuration.

        Returns:
            dict: Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "positional_embeddings": self.positional_embeddings,
        })
        return config


class Embeddings(tf.keras.layers.Layer):
    """
    Embeddings layer.

    This layer combines token embeddings with positional embeddings to create the final embeddings.

    Args:
        config (object): Configuration object containing parameters.

    Attributes:
        token_embeddings (tf.keras.layers.Embedding): Token embedding layer.
        dropout (tf.keras.layers.Dropout): Dropout layer for regularization.
        norm (tf.keras.layers.LayerNormalization): Layer normalization for normalization.
    """

    def __init__(self, config, name = None,  **kwargs):
        super(Embeddings, self).__init__(name=name, **kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim= config.vocab_size, output_dim=config.hidden_size
        )
        self.PositionalInfo = PositionalEmbeddings(config)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, input_ids, training=False):
        """
        Generate embeddings for input IDs.

        Args:
            input_ids (tf.Tensor): Input tensor containing token IDs.
            training (bool, optional): Whether the model is in training mode. Defaults to False.

        Returns:
            tf.Tensor: Embeddings tensor of shape (batch_size, seq_length, hidden_size).
        """
        positional_info = self.PositionalInfo(input_ids)
        x = self.token_embeddings(input_ids)
        x += positional_info
        x = self.norm(x)
        x = self.dropout(x, training=training)
        return x

    def compute_mask(self, inputs, mask=None):
        """
        Computes the mask for the inputs.

        Args:
            inputs (tf.Tensor): Input tensor.
            mask (tf.Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            tf.Tensor: Computed mask tensor.
        """
        return tf.math.not_equal(inputs, 50357)

    def get_config(self):
        """
        Get the layer configuration.

        Returns:
            dict: Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "token_embeddings": self.token_embeddings,
            "PositionalInfo": self.PositionalInfo,
            "dropout": self.dropout,
            "norm": self.norm,
        })
        return config
```

### Autoregressive Self-Attention

```
class AttentionHead(tf.keras.layers.Layer):
    """
    Attention head implementation.

    Args:
        head_dim: Dimensionality of the attention head.

    Attributes:
        head_dim: Dimensionality of the attention head.
        query_weights: Dense layer for query projection.
        key_weights: Dense layer for key projection.
        value_weights: Dense layer for value projection.
    """

    def __init__(self, head_dim, name = None, **kwargs):
        super(AttentionHead, self).__init__(name=name, **kwargs)
        self.supports_masking = True  # Enable masking support
        self.head_dim = head_dim
        self.query_weights = tf.keras.layers.Dense(head_dim)
        self.key_weights = tf.keras.layers.Dense(head_dim)
        self.value_weights = tf.keras.layers.Dense(head_dim)

    def call(self, query, key, value, mask=None):
        """
        Applies attention mechanism to the input query, key, and value tensors.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            mask: Optional mask tensor.

        Returns:
            Updated value embeddings after applying attention mechanism.
        """
        query = self.query_weights(query)
        key = self.key_weights(key)
        value = self.value_weights(value)

        att_scores = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1])) / tf.math.sqrt(tf.cast(tf.shape(query)[-1], tf.float32))

        if mask is not None:
            mask = tf.cast(mask, dtype=tf.bool)
            att_scores = tf.where(mask, att_scores, tf.constant(-1e9, dtype=att_scores.dtype))

        att_weights = tf.nn.softmax(att_scores, axis=-1)
        n_value = tf.matmul(att_weights, value)

        return n_value

    def get_config(self):
        """
        Returns the configuration of the attention head layer.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "head_dim": self.head_dim,
            "query_weights": self.query_weights,
            "key_weights": self.key_weights,
            "value_weights": self.value_weights,
        })
        return config


class MultiHead_Attention(tf.keras.layers.Layer):
    """
    Multi-head attention layer implementation.

    Args:
        config: Configuration object containing hyperparameters.

    Attributes:
        supports_masking: Boolean indicating if the layer supports masking.
        hidden_size: Dimensionality of the hidden state.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        attention_heads: List of AttentionHead layers.
        fc: Fully connected layer for final projection.
    """

    def __init__(self, config, name=None, **kwargs):
        super(MultiHead_Attention, self).__init__(name=name, **kwargs)
        self.supports_masking = True
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.attention_heads = [AttentionHead(self.head_dim) for _ in range(self.num_heads)]
        self.fc = tf.keras.layers.Dense(config.hidden_size)

    def call(self, query, key, value, mask=None):
        """
        Applies multi-head attention mechanism to the input query, key, and value tensors.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            mask: Optional mask tensor.

        Returns:
            Updated hidden state after applying multi-head attention mechanism.
        """
        attention_outputs = [attention_head(query, key, value, mask=mask) for attention_head in self.attention_heads]
        hidden_state = tf.concat(attention_outputs, axis=-1)
        hidden_state = self.fc(hidden_state)
        return hidden_state

    def get_config(self):
        """
        Returns the configuration of the multi-head attention layer.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "attention_heads": self.attention_heads,
            "fc": self.fc,
        })
        return config
```

### The Feed-Forward Layer

```
class FeedForward(tf.keras.layers.Layer):
    """
    Feed-forward layer implementation.

    Args:
        config: Configuration object containing hyperparameters.

    Attributes:
        supports_masking: Boolean indicating if the layer supports masking.
        fc1: First dense layer.
        fc2: Second dense layer.
        dropout: Dropout layer.
    """

    def __init__(self, config, name=None, **kwargs):
        super(FeedForward, self).__init__(name=name, **kwargs)
        self.supports_masking = True
        self.fc1 = tf.keras.layers.Dense(config.intermediate_fc_size, activation=tf.keras.activations.gelu)
        self.fc2 = tf.keras.layers.Dense(config.hidden_size)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_state, training=False):
        """
        Applies feed-forward transformation to the input hidden state.

        Args:
            hidden_state: Hidden state tensor (batch_size, sequence_length, hidden_size).
            training: Boolean indicating whether the model is in training mode or inference mode.

        Returns:
            Updated hidden state after applying feed-forward transformation.
        """
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.dropout(hidden_state, training=training)
        hidden_state = self.fc2(hidden_state)
        return hidden_state

    def get_config(self):
        """
        Returns the configuration of the feed-forward layer.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "fc1": self.fc1,
            "fc2": self.fc2,
            "dropout": self.dropout,
        })
        return config
```

### Decoder
The main difference in the decoder in GPT from the decoder in transformer model is that there is no longer an encoder, so there are no more inputs to the decoder from the encoder, and thus there is no need for a second attention layer. That's it.

```
class Decoder(tf.keras.layers.Layer):
    """
    Decoder layer implementation.

    Args:
        config: Configuration object containing hyperparameters.

    Attributes:
        masked_multihead_attention: Masked multi-head attention layer.
        norm1: Layer normalization layer.
        norm2: Layer normalization layer.
        feed_forward: Feed-forward layer.
        dropout: Dropout layer.
    """

    def __init__(self, config, name=None, **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.masked_multihead_attention = MultiHead_Attention(config)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.feed_forward = FeedForward(config)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_state, mask=None, training=False):
        """
        Applies the decoder layer to the input hidden state.

        Args:
            hidden_state: Hidden state tensor.
            mask: mask tensor.
            training: Boolean indicating if the model is in training mode.

        Returns:
            Updated hidden state after applying the decoder layer.
        """
        causal_mask = self.get_causal_attention_mask(hidden_state)

        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output = self.masked_multihead_attention(hidden_state, hidden_state, hidden_state, mask=causal_mask)
        attention_output = self.dropout(attention_output, training=training)
        hidden_state = self.norm1(attention_output + hidden_state)

        feed_forward_output = self.feed_forward(hidden_state)
        feed_forward_output = self.dropout(feed_forward_output, training=training)
        hidden_state = self.norm2(feed_forward_output + hidden_state)

        return hidden_state

    def get_causal_attention_mask(self, inputs):
        """
        Generates the causal attention mask.

        Args:
            inputs: Input tensor.

        Returns:
            Causal attention mask tensor.
        """
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype=tf.int32)
        mask = tf.reshape(mask, (1, sequence_length, sequence_length))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def get_config(self):
        """
        Returns the configuration of the decoder layer.

        Returns:
            Configuration dictionary.
        """
        config = super(Decoder, self).get_config()
        config.update({
            "masked_multihead_attention": self.masked_multihead_attention,
            "norm1": self.norm1,
            "norm2": self.norm2,
            "feed_forward": self.feed_forward,
            "dropout": self.dropout,
        })
        return config
```

### GPT Model

The `GPT` class is defined as a subclass of `tf.keras.Model`, which means it's designed to work with TensorFlow's Keras API for building and training neural networks. This class represents the core architecture of the GPT model and includes methods for creating the model's layers and performing the forward pass.

- The `__init__` method initializes the GPT model. It takes a `config` argument, which is a configuration object containing hyperparameters for the model. Within this method:
  - An embedding layer named `embed_layer` is created using the `Embeddings` class. This layer is used to convert input tokens into continuous vector representations.
  - The `decoder` attribute is a list of `Decoder` layers. The number of decoder layers is determined by `config.num_blocks`.
  - A dropout layer named `dropout` is included for regularization. The dropout rate is specified by `config.final_dropout_prob`.
  - An output dense layer named `output_layer` is created to predict the probabilities of the next token in the sequence.

- The `call` method defines the forward pass of the GPT model. Given input tokens (`inputs`), it:
  - Passes the tokens through the embedding layer to obtain continuous embeddings.
  - Sequentially feeds the embeddings through each decoder layer in the `decoder` list.
  - Applies dropout to the decoder output for regularization.
  - Passes the result through the output dense layer to obtain logits, which represent the predicted probabilities for each token in the vocabulary.
  - Removes the mask from the logits since it's not needed in the loss function.

- The `get_config` method returns a dictionary containing the configuration of the GPT model. It includes references to the `embed_layer`, `decoder` layers, `dropout` layer, and `output_layer`. This method is used for saving and loading model configurations.

```
import tensorflow as tf

class GPT(tf.keras.Model):
    """
    GPT model implementation.

    Args:
        config: Configuration object containing model hyperparameters.

    Attributes:
        embed_layer: Embeddings layer for the decoder inputs.
        decoder: List of decoder layers.
        dropout: Dropout layer for regularization.
        output_layer: Dense layer for output prediction.

    Methods:
        call: Forward pass of the GPT model.
        get_config: Returns the configuration dictionary of the GPT model.
    """

    def __init__(self, config, name=None, **kwargs):
        super(GPT, self).__init__(name=name, **kwargs)
        self.embed_layer = Embeddings(config, name="embeddings")
        self.decoder = [Decoder(config) for _ in range(config.num_blocks)]
        self.dropout = tf.keras.layers.Dropout(config.final_dropout_prob)
        self.output_layer = tf.keras.layers.Dense(config.vocab_size, name="output_layer")

    def call(self, inputs, training=False, mask=None):
        """
        Forward pass of the GPT model.

        Args:
            inputs: Input data.
            training: Boolean flag indicating whether the model is in training mode or not.
            mask: Optional mask for inputs.

        Returns:
            Output logits of the GPT model.
        """
        x_dec = self.embed_layer(inputs)

        for decoder_layer in self.decoder:
            x_dec = decoder_layer(x_dec, training=training)

        x_dec = self.dropout(x_dec, training=training)
        x_logits = self.output_layer(x_dec)
        # Remove the mask from the logits as it's not needed in the loss function
        x_logits._keras_mask = None

        return x_logits

    def get_config(self):
        """
        Returns the configuration dictionary of the GPT model.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "embed_layer": self.embed_layer,
            "decoder": self.decoder,
            "dropout": self.dropout,
            "output_layer": self.output_layer,
        })
        return config

```

## Configration

```
class Config:
    """
    Configuration class for GPT-based text generation model.

    Attributes:
        tokenizer_path (str): The path to the pre-trained tokenizer.
        data_dir (str): Directory containing the training data.
        checkpoint_directory (str): Directory to save model and optimizer checkpoints.
        model_weights_checkpoint_directory (str): Directory to save model wights checkpoints.
        shuffle (bool): Whether to shuffle the training data.
        lower_case (bool): Whether to convert text to lowercase during preprocessing.
        remove_punctuation (bool): Whether to remove punctuation during preprocessing.
        sequence_length (int): The maximum sequence length for input data.
        buffer_size (int): Size of the data buffer for shuffling.
        batch_size (int): Batch size for training.
        seed (int): Random seed for reproducibility.
        vocab_size (int): Vocabulary size, including the PAD token.
        hidden_size (int): Size of the hidden layers in the model.
        intermediate_fc_size (int): Size of intermediate fully connected layers.
        warmup_steps (int): Number of warm-up steps for learning rate scheduling.
        max_learning_rate (float): Maximum learning rate for training.
        hidden_dropout_prob (float): Dropout probability for hidden layers.
        num_heads (int): Number of attention heads in the model.
        final_dropout_prob (float): Dropout probability for the final layer.
        num_blocks (int): Number of transformer blocks in the model.
        num_epochs (int): Number of training epochs.
        patience (int): Number of epochs with no improvement to trigger early stopping.
        end_token (int): The identifier for the end-of-sentence symbol. When you train the tokenizer this identifier changes, so be careful.
    """

    def __init__(self):
        self.tokenizer_path = "tokenizer/adapted-tokenizer"
        self.data_dir = 'aclImdb'
        self.checkpoint_directory = 'tmp/checkpoint'
        self.model_weights_checkpoint_directory = 'tmp/weights_checkpoint'
        self.shuffle = True
        self.lower_case = True
        self.remove_punctuation = True
        self.sequence_length = 128
        self.buffer_size = 42500
        self.batch_size = 64
        self.seed = 2023
        self.vocab_size = 50357+1  # Because we have added the PAD token.
        self.hidden_size = 256
        self.intermediate_fc_size = self.hidden_size * 4
        self.warmup_steps = 4000
        self.max_learning_rate = 2.5e-4
        self.total_number_of_training_samples = 42500
        self.hidden_dropout_prob = 0.1
        self.num_heads = 4
        self.final_dropout_prob = 0.5
        self.num_blocks = 2
        self.num_epochs = 20
        self.patience = 4
        self.end_token = 0
```

## End-to-end GPT

### LrSchedule

According to the paper:

>We used the Adam optimization with a max learning rate of 2.5e-4. The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule.

```
import tensorflow as tf

class LrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule for training a model.

    This class implements a learning rate schedule that combines linear warmup
    followed by a cosine annealing schedule. It is designed to be used as the
    learning rate schedule for the optimizer during training.

    Args:
        config: Configuration object containing schedule hyperparameters.

    Attributes:
        warmup_steps: Number of warmup steps during which the learning rate increases linearly.
        max_learning_rate: Maximum learning rate reached after warmup.
        total_steps: Total number of training steps.
        learning_rate_schedule: Learning rate schedule for warmup phase.
        cosine_schedule: Learning rate schedule for cosine annealing phase.

    Methods:
        __call__: Returns the learning rate for a given training step.
        get_config: Returns the configuration dictionary of the learning rate schedule.
    """

    def __init__(self, config):
        super(LrSchedule, self).__init__()
        self.warmup_steps = config.warmup_steps
        self.max_learning_rate = config.max_learning_rate
        self.total_steps = config.num_epochs * (config.total_number_of_training_samples // config.batch_size)
        self.learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.0,
            decay_steps=self.warmup_steps,
            end_learning_rate=self.max_learning_rate,
            power=1.0  # Linear warmup
        )
        self.cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.max_learning_rate,
            decay_steps=self.total_steps - self.warmup_steps
        )

    def __call__(self, step):
        """
        Returns the learning rate for a given training step.

        Args:
            step: Training step.

        Returns:
            Learning rate for the given step.
        """
        def learning_rate_fn(step):
            if step < self.warmup_steps:
                return self.learning_rate_schedule(step)
            return self.cosine_schedule(step - self.warmup_steps)

        return tf.cond(step < self.warmup_steps,
                       lambda: self.learning_rate_schedule(step),
                       lambda: self.cosine_schedule(step - self.warmup_steps))


    def get_config(self):
        """
        Returns the configuration dictionary of the learning rate schedule.

        Returns:
            Configuration dictionary.
        """
        learning_rate_schedule_config = tf.keras.optimizers.schedules.serialize(self.learning_rate_schedule)
        cosine_schedule_config = tf.keras.optimizers.schedules.serialize(self.cosine_schedule)

        return {
            "warmup_steps": self.warmup_steps,
            "max_learning_rate": self.max_learning_rate,
            "total_steps": self.total_steps,
            "learning_rate_schedule_config": learning_rate_schedule_config,
            "cosine_schedule_config": cosine_schedule_config,
        }

```

Plot:

```
"""import numpy as np
import matplotlib.pyplot as plt

class ConfigLR:
    def __init__(self):
        self.num_epochs = 100
        self.batch_size = 64
        self.warmup_steps = 2000
        self.max_learning_rate = 2.5e-4

config = ConfigLR()

lr = LrSchedule(config, 100000)
optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Evaluate the learning rate tensor element-wise and collect the values in a numpy array
learning_rates = np.array([lr(step).numpy() for step in range(40000)], dtype=np.float32)

plt.plot(learning_rates)
plt.ylabel('Learning Rate')
plt.xlabel('Train Step')
plt.show()"""
```

### Loss Function
Note that the ID of the filler code I use is 50375, and this may change depending on the tokenizer used. This also changes if you train tokenize.

```
import tensorflow as tf

def loss_fn(label, pred):
    """
    Computes the masked Sparse Categorical Cross Entropy (SCCE) loss between the predicted and target labels.

    Args:
        label: Target label tensor.
        pred: Predicted logit tensor.

    Returns:
        Masked loss value.
    """
    # Create a mask to ignore padded tokens
    mask = label != 50357

    # Use Sparse Categorical Cross Entropy loss with no reduction
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    # Compute the loss without reducing, which will return a loss value for each token
    loss = loss_object(label, pred)

    # Apply the mask to ignore padded tokens in the loss calculation
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    # Compute the average loss over non-padded tokens
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss
```

### Metrices

Perplexity is a metric commonly used to evaluate the performance of language models, including generative models like GPT. Intuitively, perplexity means to be surprised. We measure how much the model is surprised by seeing new data. The lower the perplexity, the better the training is.

Mathematically, perplexity is defined as:

$$
\text{Perplexity} = 2^{-\frac{1}{N}\sum_{i=1}^{N} \log_2 P(w_i|w_1, w_2, ..., w_{i-1})},
$$

where:
- \(N\) is the total number of words or tokens in the sequence.
- \(w_i\) represents the \(i\)th word or token in the sequence.
- \(P(w_i|w_1, w_2, ..., w_{i-1})\) is the predicted probability assigned by the language model to the \(i\)th word given the previous words.

In essence, perplexity calculates the average negative log-likelihood of the true next word according to the model's predicted distribution. A lower perplexity indicates that the model assigns higher probabilities to the actual next words, meaning that the model's predictions align well with the true data distribution.

Perplexity is usually used only to determine how well a model has learned the training set. Other metrics like BLEU, ROUGE etc., are used on the test set to measure test performance.

**Here are the main cons**:

1. **Measuring Confidence, Not Accuracy**: Perplexity measures the model's confidence in its predictions but doesn't directly assess the accuracy of those predictions. A model can have low perplexity while still producing incorrect or nonsensical outputs if it consistently assigns high probabilities to incorrect words. It doesn't guarantee real-world performance or understanding.

2. **Apples-to-Apples Comparisons**: Comparing perplexity values across different datasets or models can be challenging due to various factors such as context lengths, vocabulary sizes, and model architectures. As you mentioned, a lower perplexity value on one dataset or with a specific configuration doesn't guarantee better overall performance or quality. Directly comparing perplexity values between different scenarios might not provide a clear indication of which model is truly better for a given task.

>We should note that the metric applies specifically to classical language models (sometimes called autoregressive or causal language models) and is not well defined for masked language models like BERT (see summary of the models).

```
import tensorflow as tf

class Perplexity(tf.keras.metrics.Metric):
    """
    Custom metric for calculating perplexity.

    Perplexity is a measure of how well a probability distribution or probability model predicts a sample.
    It is commonly used in natural language processing tasks to evaluate the quality of language models.

    Attributes:
        name (str): Name of the metric.
    """

    def __init__(self, name='perplexity', **kwargs):
        """
        Initializes the Perplexity metric.

        Args:
            name (str, optional): Name of the metric. Defaults to 'perplexity'.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super(Perplexity, self).__init__(name=name, **kwargs)
        self.loss_sum = self.add_weight(name='loss_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the metric state based on true and predicted values.

        Args:
            y_true (tensor): True target values.
            y_pred (tensor): Predicted values.
            sample_weight (tensor, optional): Optional weighting of samples. Not used.

        Returns:
            None
        """
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        mask = y_true != 50357  # Assumes that 50357 is the PAD token ID.
        # Apply the mask to ignore padded tokens in the loss calculation
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        self.loss_sum.assign_add(tf.reduce_sum(loss))
        self.count.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))

    def result(self):
        """
        Computes the final perplexity metric.

        Returns:
            tensor: The computed perplexity value.
        """
        return tf.pow(2.0, self.loss_sum / self.count)

```

### Monitor

```
import time
import tensorflow as tf
import random as python_random


def create_model_optimizer(config):
    model = GPT(config)

    # Create the learning rate schedule
    lr = LrSchedule(config)

    optimizer = tf.keras.optimizers.Adam(
        lr,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    return model, optimizer

def save_model_and_optimizer(model, optimizer, checkpoint_manager, epoch):
    # Save the model and optimizer state
    checkpoint_name = checkpoint_manager.save()
    print("Saved checkpoint for epoch {}: {}".format(epoch, checkpoint_name))

def save_model_weights_only(model, model_weights_manager, epoch):
    # Save model weights only
    model_weights_checkpoint_name = model_weights_manager.save()
    print("Saved model weights for epoch {}: {}".format(epoch, model_weights_checkpoint_name))

def train_one_epoch(model, optimizer, train_gr, loss_fn, train_ppe_metric):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_ppe_metric.update_state(y, logits)
        return loss_value

    for step, (x_batch_train, y_batch_train) in enumerate(train_gr):
        tr_loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, tr_loss_value)
            )
            print("Seen so far: %d samples" % ((step + 1) * config.batch_size))

def evaluate_one_epoch(model, val_gr, loss_fn, val_ppe_metric):
    @tf.function
    def test_step(x, y):
        logits = model(x, training=False)
        val_ppe_metric.update_state(y, logits)
        loss_value = loss_fn(y, logits)
        return loss_value

    val_loss_value = 0.0
    for x_batch_val, y_batch_val in val_gr:
        val_loss_value += test_step(x_batch_val, y_batch_val)
    return val_loss_value

def main_training_loop(config, resume_training=False):
    python_random.seed(config.seed)
    tf.random.set_seed(config.seed)

    train_gr = EnglishDataStreamer(config, 'train')
    val_gr = EnglishDataStreamer(config, 'valid')
    # Prepare the metrics.
    train_ppe_metric = Perplexity()
    val_ppe_metric = Perplexity()

    # Create the model and optimizer
    if resume_training:
        model, optimizer = load_model_and_optimizer(config)
    else:
        model, optimizer = create_model_optimizer(config)

    # Create a checkpoint for both the model and optimizer
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_directory = config.checkpoint_directory
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=1)

    # Create a separate checkpoint for model weights only
    model_weights_checkpoint = tf.train.Checkpoint(model=model)
    model_weights_checkpoint_directory = config.model_weights_checkpoint_directory
    model_weights_manager = tf.train.CheckpointManager(model_weights_checkpoint, model_weights_checkpoint_directory, max_to_keep=1)

    # Early stopping parameters
    best_val_loss = float("inf")
    patience = config.patience
    wait = 0

    for epoch in range(1, config.num_epochs+1):
        print("\n##### Start of epoch %d #####" % (epoch,))
        start_time = time.time()

        # Training
        train_one_epoch(model, optimizer, train_gr, loss_fn, train_ppe_metric)

        # Display metrics at the end of each epoch.
        train_ppe = train_ppe_metric.result()
        print("Training perplexity over epoch: %.4f" % (float(train_ppe),))

        # Reset training metrics at the end of each epoch
        train_ppe_metric.reset_states()

        # Evaluation
        val_loss_value = evaluate_one_epoch(model, val_gr, loss_fn, val_ppe_metric)
        print("Training loss over epoch: %.4f" % (val_loss_value,))

        val_ppe = val_ppe_metric.result()
        val_ppe_metric.reset_states()

        print("Validation perplexity: %.4f" % (val_ppe,))
        print("Time taken: %.2fs" % (time.time() - start_time))

        # Early Stopping Check
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            wait = 0
            # Save the model and optimizer
            save_model_and_optimizer(model, optimizer, manager, epoch)

            # Save model weights only
            save_model_weights_only(model, model_weights_manager, epoch)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered. No improvement in validation loss.")
                break

if __name__ == "__main__":
    config = Config()
    main_training_loop(config, resume_training=False)
```

### Load the model

```
import tensorflow as tf

config = Config()

custom_objects = {
    "LrSchedule": LrSchedule,
    "PositionalEmbeddings": PositionalEmbeddings,
    "Embeddings": Embeddings,
    "AttentionHead": AttentionHead,
    "MultiHead_Attention": MultiHead_Attention,
    "FeedForward": FeedForward,
    "Decoder": Decoder,
    "GPT": GPT,
    "loss_fn": loss_fn,
    "Perplexity": Perplexity
}

# I did not use `load_model_and_optimizer` function because I have worked with a small dataset,
# but I wrote it in case you wanted to train the model on a large data set,
# and then you wanted to save the entire model or you wanted to resume training at a later time.
# Because if you want to resume training at a later time, you must completely reload the Optimizer state and Model.
# This function is well tested, 
# but there were warnings which I filtered out 
# because I don't think they affect the model, but I suggest you check further.
# However, if you decide to use it to resume model training,
# you will have to modify the training loop or create another loop to accommodate it.

def load_model_and_optimizer(config):
    # Create the model and optimizer from checkpoint
    model, optimizer = create_model_optimizer(config)

    with tf.keras.utils.custom_object_scope(custom_objects):
          # Restore the model and optimizer state
          latest_checkpoint = tf.train.latest_checkpoint(config.checkpoint_directory)
          if latest_checkpoint:
              checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
              checkpoint_status = checkpoint.restore(latest_checkpoint)
              checkpoint_status.expect_partial()  # Suppress warnings about incomplete restores
              print("Restored model and optimizer from checkpoint successfully: {}".format(latest_checkpoint))
          else:
              print("Checkpoint not found. Initializing from scratch.")

    return model, optimizer

# We will use this function in the inference process
def load_model_weights_only(config):
    # Create the model
    model = GPT(config)

    # Restore only the model weights
    latest_checkpoint = tf.train.latest_checkpoint(config.model_weights_checkpoint_directory)
    if latest_checkpoint:
        model_weights_checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_status = model_weights_checkpoint.restore(latest_checkpoint)
        checkpoint_status.expect_partial()  # Suppress warnings about incomplete restores
        print("Restored model weights from checkpoint successfully: {}".format(latest_checkpoint))
    else:
        print("Model weights checkpoint not found. Initializing model from scratch.")

    return model
```

## Inference

Transforming the model's probabilistic predictions into textual form involves the use of a decoding process, which presents several distinct challenges specific to text generation:

• Decoding is an iterative process, demanding considerably more computational resources than the single forward pass typically used for input processing.

• The excellence and variety of the generated text are contingent upon the selection of the decoding technique and its corresponding hyperparameters.

You may be wondering how the process of generating text using the model we trained will work. The main idea is: We start with a prompt like "this movie is" and use the model to predict the next token. Once we have determined the next token, we append it to the prompt and then use the new input sequence to generate another token. We do this until we have reached to a predefined condition. How to choose the next token is the core of the decoding process. We now discuss the most important decoding algorithms.

**Note:**This type of text generation is often called conditional text generation, because it depends on the input prompt.

### **Greedy Search Decoding: A Deterministic Text Generation Approach**


Greedy search decoding is a simple and commonly used technique for text generation with autoregressive language models like GPT-2. In greedy search decoding, you always choose the token with the highest probability as the next token in the sequence. This approach is straightforward and fast but might result in less diverse and sometimes repetitive generated text.

The mathematical formula for greedy search decoding in text generation can be expressed as follows:

Let:
- \(S\) be the generated sequence of tokens.
- \(P\) be the probability distribution over vocabulary for the next token given the current sequence \(S\).
- \(t\) represent the time step in the generation process, starting from \(t = 0\) for the initial seed text.
- \(T\) be the maximum sequence length allowed.

The process of greedy search decoding involves iteratively selecting the token with the highest probability at each time step \(t\) and appending it to the sequence \(S\), subject to the constraint that the length of \(S\) does not exceed \(T\). The selection of the next token can be defined as:

$$
S(t) = argmax_{i} P(i|S)
$$

Where:
- \(S(t)\) is the token selected at time step \(t\).
- \(argmax\) denotes selecting the token index that maximizes the probability.
- \(P(i|S)\) is the conditional probability of token \(i\) given the current sequence \(S\).

The process continues until the maximum sequence length \(T\) is reached. We also can depend on an end-of-sequence token (e.g., \<eos\>) is predicted, indicating the completion of the generated text.

```
from abc import ABCMeta, abstractmethod

class Sampler(metaclass=ABCMeta):
    """
    Abstract base class for text generation samplers.
    """

    @abstractmethod
    def decode(self, sentence):
        """
        Decode a sentence based on the implemented sampling strategy.

        Args:
            sentence (str): The input sentence or prompt.

        Returns:
            str: The generated output sentence.
        """


class GreedySampler(Sampler):
    """
    Greedy sampling strategy for text generation using a transformer-based language model.

    This sampler uses a greedy approach to generate text based on the predictions of a transformer-based
    language model. It tokenizes an input prompt, iteratively generates tokens, and selects the token
    with the highest probability at each step until the maximum sequence length is reached or an
    end token (if provided) is predicted.

    Args:
        model: The transformer-based language model.
        tokenizer_path (str): The name or path of the pre-trained tokenizer.
        sequence_length (int): The maximum length of tokenized sequences.
        end_token (int or None, optional): An optional token ID that signifies the end of decoding. If
            provided, decoding will stop when this token is predicted. Defaults to None.
    """

    def __init__(self, model, tokenizer_path, sequence_length, end_token=0):
        """
        Initialize the GreedySampler.

        Args:
            model: The transformer-based language model.
            tokenizer_path (str): The name or path of the pre-trained tokenizer.
            sequence_length (int): The maximum length of tokenized sequences.
            end_token (int or None, optional): An optional token ID that signifies the end of decoding.
                If provided, decoding will stop when this token is predicted. Defaults to None.
        """
        self.model = model
        self.tokenizer = EnglishDataTokenizer(tokenizer_path, sequence_length, training=False)
        self.sequence_length = sequence_length
        self.end_token = end_token

    def decode(self, input_prompt):
        """
        Decode a sentence based on a greedy sampling strategy.

        Args:
            input_prompt (str): The input sentence or prompt.

        Returns:
            str: The generated output sentence.
        """
        input_ids = np.reshape(self.tokenizer.tokenize(input_prompt), (1, -1))
        for _ in range(self.sequence_length):
            predictions = self.model(input_ids, training=False)
            predicted_id = tf.argmax(predictions, axis=-1)
            input_ids = tf.concat([input_ids, predicted_id], axis=-1)
            if self.end_token is not None and predicted_id == self.end_token:
                break

        generated_text = self.tokenizer.detokenize(input_ids.numpy()[0])

        return generated_text
```

### **Beam Search Decoding: Enhancing Text Generation Precision and Diversity**

Beam search decoding is a popular technique for generating text using autoregressive language models like GPT-2. It's a variation of greedy decoding that explores multiple possible continuations of a sequence to improve the quality and diversity of generated text. Instead of selecting the most likely next token at each step, beam search keeps track of the top `k` candidates (where `k` is called the "beam width") and selects from them.

As you can notice; The primary goal is to find the most likely sequence of words or tokens in a probabilistic model, such as a language model. In contrast to greedy search, which selects the most likely word at each step, beam search keeps multiple possibilities, including those that may have lower probabilities at intermediate steps but lead to higher probabilities overall.


Here's how beam search decoding works for text generation:

1. **Initialization**:
   - Start with an initial input or prompt, typically a few words or a sentence.
   - Set a parameter `num_beams`, which determines how many alternative sequences (beams) to consider at each decoding step. A higher `num_beams` value explores more possibilities but increases computational complexity.

2. **Tokenization**:
   - Tokenize the initial input to obtain the initial token IDs.

3. **Decoding Loop**:
   - Initialize `num_beams` sequences, each starting with the same initial tokens.
   - At each decoding step:
     - For each of the `num_beams` sequences:
       - Generate the next token by sampling from the probability distribution over the vocabulary based on the current sequence's context.
       - Extend each sequence with the generated token.
       - Calculate the score (log probability) for each extended sequence.
     - Select the top `num_beams` sequences with the highest scores.
   - Repeat the decoding step until a stopping condition is met:
     - The stopping condition may include reaching a maximum sequence length, generating a predefined end token (e.g., a period for sentence generation), or producing a certain number of sequences.

4. **Output Selection**:
   - After decoding is complete, you have `num_beams` candidate sequences.
   - Choose the sequence with the highest cumulative score (log probability) as the final generated text.

Here are some key points to understand about beam search decoding:

- **Diverse Outputs**: Beam search allows for diversity in the generated text by exploring multiple hypotheses (beams) simultaneously. Different beams may produce variations of the same text, offering options for diverse outputs.

- **Trade-Off**: The choice of `num_beams` is a trade-off between diversity and computational cost. Larger `num_beams` values increase diversity but also increase computation time.

- **Prominent Use Cases**: Beam search is commonly used in machine translation, text summarization, and text generation tasks where generating coherent and contextually relevant text is crucial.

- **Refinement**: To further improve text generation, techniques like length normalization and nucleus sampling can be combined with beam search.

- **Beam Search Variants**: There are variants of beam search, such as diverse beam search, which aim to produce more diverse outputs by encouraging dissimilarity between beams.

```
class BeamSearchSampler(Sampler):
    def __init__(self, model, tokenizer, sequence_length, beam_width=2, end_token=0):
        """
        Initialize the BeamSearchSampler.

        Args:
            model: The language model used for text generation.
            tokenizer: The tokenizer used for tokenizing text.
            sequence_length (int): The maximum sequence length for generated text.
            beam_width (int): Width of the beam search (number of beams). Default is 2.
            end_token (int or None): The token indicating the end of a sequence, if applicable. Default is None.
        """
        self.model = model
        self.tokenizer = EnglishDataTokenizer(tokenizer_path, sequence_length, training=False)
        self.sequence_length = sequence_length
        self.beam_width = beam_width
        self.end_token = end_token

    def decode(self, input_prompt):
        """
        Decode a text sequence based on beam search.

        Args:
            input_prompt (str): The input text or prompt.

        Returns:
            str: The generated output text.
        """
        input_ids = np.reshape(self.tokenizer.tokenize(input_prompt), (1, -1))
        iterations = self.sequence_length - input_ids.shape[-1]

        # Create an initial beam of size 1
        beams = [(input_ids, tf.constant(0.0, dtype=tf.float32))]

        for k in range(iterations):
            all_candidates = []
            # Expand each beam
            for beam_input_ids, beam_score in beams:
                logits = self.model(beam_input_ids)
                last_token_logits = logits[:, -1, :]
                # Use top-k sampling to get the most likely next tokens
                top_k = tf.math.top_k(last_token_logits, self.beam_width)
                next_token_ids = top_k.indices
                log_probs = top_k.values

                for i in range(self.beam_width):
                    if self.end_token is not None and next_token_ids[0, i] == self.end_token:
                        # End of sequence
                        all_candidates.append((beam_input_ids, beam_score))
                        continue

                    new_beam_input_ids = tf.concat([beam_input_ids, tf.reshape(next_token_ids[0, i], (1, 1))], axis=-1)
                    new_beam_score = beam_score - log_probs[0, i]
                    all_candidates.append((new_beam_input_ids, new_beam_score))

            # Select the top-k candidates from all expanded beams
            all_candidates = sorted(all_candidates, key=lambda x: x[1])
            beams = all_candidates[:self.beam_width]

        # Return the best beam at the end
        best_beam = min(beams, key=lambda x: x[1])
        #print(best_beam[0])
        return self.tokenizer.detokenize(best_beam[0][0])
```

```
def generate_response(input_prompt, sampler_type, beam_width):
    """
    Generate text using a trained GPT model.

    Args:
        input_prompt (str): The input text prompt for text generation.
        sampler_type (str): Sampling strategy ('greedy' or 'beam'). Default is beam.
        beam_width (int): Beam width for beam search sampling. Default is 5.
    """
    model = load_model_and_optimizer(config.model_weights_checkpoint_directory)

    if sampler_type == 'beam':
        sampler = BeamSearchSampler(model, config.tokenizer_path, config.sequence_length, beam_width=beam_width, end_token=config.end_token)
    else:
        sampler = GreedySampler(model, config.tokenizer_path, config.sequence_length, end_token=config.end_token)

    generated_text = sampler.decode(input_prompt)

    # Print the generated text
    print("Generated Text:", generated_text)
    return generated_text
```

```
config = Config()
dummy_model = GPT(config)

# Generate text using beam search decoding
input_prompt = "This movie is"

generate_response(input_prompt, sampler_type='beam', beam_width=5)
```

**Greedy Approach vs. Beam Search Algorithm: A Comparative Analysis**

The greedy approach and the beam search algorithm are two distinct methods for decoding and generating text in natural language processing tasks. Here, we provide a comparative analysis of these approaches, highlighting their differences and relative advantages:

**1. Search Strategy:**

   - **Greedy Approach:** The greedy approach makes decisions at each step based solely on the highest probability option. It selects the token that appears to be the most likely continuation of the sequence at each step.

   - **Beam Search Algorithm:** Beam search, on the other hand, maintains multiple candidate sequences, or "beams," and explores a set number of top-ranked possibilities at each step. It retains a fixed number of promising sequences throughout decoding.

**2. Exploration of Possibilities:**

   - **Greedy Approach:** Greedy decoding is deterministic and may quickly converge to a locally optimal solution. It tends to produce more deterministic and less diverse output.

   - **Beam Search Algorithm:** Beam search allows for a broader exploration of possibilities. It considers multiple candidate sequences, promoting diversity in generated output.

**3. Sequence Length:**

   - **Greedy Approach:** Greedy decoding does not consider the global context of the sequence, focusing only on the current step. It may lead to sequences that lack coherence or context.

   - **Beam Search Algorithm:** Beam search considers longer-range dependencies by retaining multiple sequences, which can result in more coherent and contextually relevant output.

**4. Trade-off:**

   - **Greedy Approach:** Greedy decoding is computationally efficient as it only considers one option at each step. However, it may sacrifice output quality for speed.

   - **Beam Search Algorithm:** Beam search is more computationally intensive as it maintains multiple beams, potentially slowing down the decoding process. However, it often leads to higher-quality and more diverse output.

**5. Output Quality:**

   - **Greedy Approach:** The greedy approach can produce decent output but may get stuck in local optima, resulting in repetitive or less creative text.

   - **Beam Search Algorithm:** Beam search generally produces higher-quality and more diverse output, making it suitable for tasks where output quality and diversity are crucial.

**6. Hyperparameter Dependency:**

   - **Greedy Approach:** Greedy decoding typically does not involve many hyperparameters and is relatively straightforward to implement.

   - **Beam Search Algorithm:** Beam search requires tuning hyperparameters like beam width, which can impact its performance.


### **Sampling Methods: Other ideas for generating text**



Top-k and nucleus (or top-p) sampling are two techniques used in text generation to control the diversity and quality of generated text. Both top-k and nucleus sampling strategies help control the randomness and quality of text generation. Top-k is a fixed-size strategy, while nucleus sampling is adaptive based on a probability threshold.They both offer ways to make text generation more controllable and coherent. Here's an explanation of each:

1. **Top-k Sampling**:
   - **Idea**: In top-k sampling, instead of considering all possible tokens in the vocabulary, you limit the selection to the top-k most likely tokens at each step of generation.
   - **How It Works**:
     - At each generation step, you calculate the probabilities for all tokens in the vocabulary.
     - You then select the top-k tokens with the highest probabilities.
     - Randomly sample from this reduced set of k tokens according to their probabilities to choose the next token.
   - **Purpose**:
     - Top-k sampling helps in producing more focused and deterministic text because it limits the choices to a small set of high-probability tokens.
     - It reduces the risk of generating nonsensical or erratic text.

2. **Nucleus (Top-p) Sampling**:
   - **Idea**: Nucleus sampling, sometimes called top-p sampling, dynamically selects the number of tokens to consider based on a threshold probability p.
   - **How It Works**:
     - At each generation step, you calculate the probabilities for all tokens in the vocabulary.
     - You sort the tokens by probability and keep adding tokens with the highest probabilities until the cumulative probability surpasses the threshold p.
     - You then randomly sample from this set of tokens according to their probabilities to choose the next token.
   - **Purpose**:
     - Nucleus sampling offers a balance between randomness and determinism.
     - It allows for dynamic adjustment of the token set based on their probabilities, ensuring that you can generate both focused and diverse text depending on the value of p.
     - By setting p closer to 1, you get more diverse text, while setting it closer to 0 makes the generation more deterministic.


Both top-k and nucleus sampling strategies help control the randomness and quality of text generation. Top-k is a fixed-size strategy, while nucleus sampling is adaptive based on a probability threshold.


## References

* [Attention is All You Need - Transformer Model for Machine Translation](https://github.com/AliHaiderAhmad001/Neural-Machine-Translator/blob/main/README.md).
* [Natural Language Processing with Transformers](https://www.amazon.com/Natural-Language-Processing-Transformers-Revised/dp/1098136799).
* [Improving Language Understanding by Generative Pre-Training](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)
* [Language Models are Unsupervised Multitask Learners](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe).
* [Text generation with a miniature GPT](https://keras.io/examples/generative/text_generation_with_miniature_gpt/).
