import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from tokenizer import EnglishDataTokenizer

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
