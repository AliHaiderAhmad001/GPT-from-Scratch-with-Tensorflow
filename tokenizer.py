from transformers import AutoTokenizer
from abc import ABCMeta, abstractmethod

class DataTokenizer(metaclass=ABCMeta):
    """
    A wrapper class for tokenizing sentences using the Hugging Face transformers library.

    Methods:
        tokenize(sentence): Tokenizes a sentence and returns the tokenized input IDs.
    """

    @abstractmethod
    def tokenize(self, sentence):
        """
        Tokenizes a sentence and returns the tokenized input IDs.

        Args:
            sentence (str): The input sentence to be tokenized.

        Returns:
            tf.Tensor: Tokenized input IDs.
        """

class EnglishDataTokenizer(DataTokenizer):
    """
    An implementation of the DataTokenizer for English data.

    Args:
        tokenizer_path (str): The name or path of the pre-trained tokenizer.
        sequence_length (int): The maximum length of tokenized sequences.

    Example:
        tokenizer_path = "tokenizer/adapted-tokenizer"
        sequence_length = 10

        # Create a Tokenizer instance
        tokenizer = EnglishDataTokenizer(tokenizer_path, sequence_length)

        # Test sentence
        test_sentence = "This is a test sentence for tokenization."

        # Tokenize the test sentence
        tokenized_input_ids = tokenizer.tokenize(test_sentence)

        # Print the tokenized input IDs
        print("Tokenized Input IDs:", tokenized_input_ids)
    """

    def __init__(self, tokenizer_path, sequence_length):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, pad_token="[PAD]", pad_token_id=0)
        self.sequence_length = sequence_length

    def tokenize(self, sentence):
        tokenized_sentence = self.tokenizer(sentence,
                                            padding='max_length',
                                            max_length=self.sequence_length + 1, # For shifting
                                            truncation=True,
                                            return_tensors="np")
        return tokenized_sentence['input_ids'][0]
