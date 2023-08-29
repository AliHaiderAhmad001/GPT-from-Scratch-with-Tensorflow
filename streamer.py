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
        _start_fetching(): Starts the background fetching of data.
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
    def start_fetching(self):
        """
        Starts the background fetching of data.
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
    def get_filenames(self, data_dir):
        """
        Retrieves a list of file paths from subdirectories within the given root directory.

        Args:
            data_dir (str): The root directory containing subdirectories.

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
    An implementation of the DataStreamer for English data.

    Args:
        data_dir (str): The root directory containing subdirectories with data files.
        tokenizer_path (str): Path to the tokenizer model or name of a pre-trained tokenizer.
        max_length (int): Maximum sequence length for tokenization.
        buffer_size (int, optional): The size of the internal buffer for loading data. Default is 1024.
        batch_size (int, optional): The number of tokenized sequences in each batch. Default is 64.
        shuffle (bool, optional): Whether to shuffle the data within the buffer. Default is True.
        lower_case (bool, optional): Whether to convert text to lowercase. Default is False.
        seed (int, optional): Seed for random operations. Default is 0.
    """

    def __init__(self, data_dir, tokenizer_path, max_length,
                 buffer_size=1024, batch_size=64, shuffle=True,
                 lower_case=False, seed=0):
        assert buffer_size >= batch_size, "buffer_size should be equal or greater than batch_size"

        # Initialize attributes
        self.buffer_idx = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.shuffle = shuffle
        self.lower_case = lower_case
        self.random_state = np.random.RandomState(seed)
        self.buffer = []  
        self.ptr = 0
        self.flag = False
        self.filenames = self.get_filenames(data_dir)
        self.tokenizer = EnglishDataTokenizer(tokenizer_path, max_length)
        self.fetch_to_buffer()
        self.fetching_thread = None
        self.fetching_executor = ThreadPoolExecutor(max_workers=1)
        self.start_fetching()

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
            return sentence

        def process_data(filename):
            """ Read the data and perform the necessary processing and conversion operations """
            with open(filename, "r") as file:
                sentence = file.readline()
                sentence = custom_standardization(sentence)
                return self.tokenizer.tokenize(sentence)

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

    def start_fetching(self):
        if not self.fetching_thread or not self.fetching_thread.is_alive():
            self.fetching_thread = self.fetching_executor.submit(self.fetch_to_buffer)

    def __next__(self):
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
            raise StopIteration

        if self.ptr + self.batch_size > len(self.buffer):
            # Wait for the fetching_thread to complete and update the buffer
            self.fetching_thread.result()
            self.fetching_thread = None

            # Load the next batch if available
            if self.buffer_idx < len(self.filenames):
                self._start_fetching()

        batch = self.buffer[self.ptr:self.ptr + self.batch_size]
        self.ptr += self.batch_size

        if len(batch) < self.batch_size and self.buffer_idx >= len(self.filenames):
            self.flag = True

        return prepare_lm_inputs_labels(batch)

    def get_filenames(self, data_dir):
        filenames = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                filenames.append(os.path.join(root, file))
        return filenames

    def reset(self):
        self.buffer_idx = 0
        self.fetch_to_buffer()

