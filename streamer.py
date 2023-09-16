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
