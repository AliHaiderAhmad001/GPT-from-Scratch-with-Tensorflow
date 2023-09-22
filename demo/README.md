# GenesisMind-Building-GPT1-from-Scratch

GPT-1 "Generative Pre-trained Transformer" is the first version of the GPT series of models, revolutionized natural language processing with its autoregressive language modeling capabilities built on the Transformer architecture. This repository serves as a comprehensive guide to understanding and implementing the GPT model. I'm gonna walk through the essential components, techniques, and principles behind GPT.

**Note:** This is a mini-GPT model for text generation. The original model is very large and deals with massive data, so of course here we will not be using the same dataset or even the same size of model. However, you can control the parameters or even modify it easily if necessary to match the original model.

## The environment
We are creating this demo within Jupiter Notebook, so first of all don't forget to go to the working folder:
```
%cd '/content/drive/MyDrive/Colab Notebooks/projects/GenesisMind-Building-GPT1-from-Scratch'
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
```


