from transformers import AutoTokenizer
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
        pass

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

    Example:
        # Create a GreedySampler instance
        sampler = GreedySampler(model, tokenizer_path, sequence_length, end_token=5)

        # Generate text based on an input prompt
        input_prompt = "Once upon a time,"
        generated_text = sampler.decode(input_prompt)

        # Print the generated text
        print("Generated Text:", generated_text)
    """

    def __init__(self, model, tokenizer_path, sequence_length, end_token=None):
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
        self.tokenizer = EnglishDataTokenizer(tokenizer_path, sequence_length)
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

        generated_text = self.tokenizer.decode(input_ids.numpy()[0])

        return generated_text
