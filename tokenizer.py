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
