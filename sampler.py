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


class BeamSearchSampler(Sampler):
    """
    Beam search sampling strategy for text generation using a transformer-based language model.

    Args:
        model: The transformer-based language model.
        tokenizer_path (str): The name or path of the pre-trained tokenizer.
        sequence_length (int): The maximum length of tokenized sequences.
        beam_width (int): The beam width for beam search.
        end_token (int or None, optional): An optional token ID that signifies the end of decoding. If
            provided, decoding will stop when this token is predicted. Defaults to None.

    Example:
        # Create a BeamSearchSampler instance
        sampler = BeamSearchSampler(model, tokenizer_path, sequence_length, beam_width=5)

        # Generate text based on an input prompt
        input_prompt = "Once upon a time,"
        generated_text = sampler.decode(input_prompt)

        # Print the generated text
        print("Generated Text:", generated_text)
    """

    def __init__(self, model, tokenizer_path, sequence_length, beam_width, end_token=None):
        """
        Initialize the BeamSearchSampler.

        Args:
            model: The transformer-based language model.
            tokenizer_path (str): The name or path of the pre-trained tokenizer.
            sequence_length (int): The maximum length of tokenized sequences.
            beam_width (int): The beam width for beam search.
            end_token (int or None, optional): An optional token ID that signifies the end of decoding.
                If provided, decoding will stop when this token is predicted. Defaults to None.
        """
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.sequence_length = sequence_length
        self.beam_width = beam_width
        self.end_token = end_token

    def decode(self, input_prompt):
        """
        Decode a sentence based on beam search sampling strategy.

        Args:
            input_prompt (str): The input sentence or prompt.

        Returns:
            str: The generated output sentence.
        """
        input_ids = self.tokenizer.encode(input_prompt, return_tensors="tf")
        input_ids = tf.repeat(input_ids, self.beam_width, axis=0)
        input_ids = tf.pad(input_ids, [[0, 0], [0, self.sequence_length - input_ids.shape[1]]])

        beam_scores = tf.zeros((self.beam_width,), dtype=tf.float32)
        beam_hypotheses = [tf.constant([], dtype=tf.int32) for _ in range(self.beam_width)]
        done_beams = []

        for step in range(self.sequence_length):
            # Expand beams
            if step > 0:
                input_ids = tf.repeat(input_ids, self.beam_width, axis=0)

            # Get predictions
            predictions = self.model(input_ids, training=False)
            vocab_size = predictions.shape[-1]

            # Calculate scores for all possible tokens
            if step == 0:
                scores = tf.math.log(predictions[0, 0, :])
            else:
                scores = tf.reshape(beam_scores, (-1, 1)) + tf.math.log(predictions)

            # Get the top-k token indices for each beam
            top_k_scores, top_k_indices = tf.math.top_k(scores, k=self.beam_width, sorted=False)

            new_beam_scores = []
            new_beam_hypotheses = []

            # Update beams
            for i in range(self.beam_width):
                token_idx = top_k_indices[i]
                score = top_k_scores[i]

                # Check for end token
                if self.end_token is not None and token_idx == self.end_token:
                    done_beams.append((beam_scores[i], beam_hypotheses[i]))
                else:
                    new_beam_scores.append(score)
                    new_beam_hypotheses.append(tf.concat([beam_hypotheses[i], [token_idx]], axis=0))

            beam_scores = tf.stack(new_beam_scores)
            beam_hypotheses = new_beam_hypotheses

            # Check if all beams are done
            if len(done_beams) == self.beam_width:
                break

        # Sort the completed beams by score and get the best one
        done_beams.sort(key=lambda x: x[0], reverse=True)
        best_beam = done_beams[0][1]

        generated_text = self.tokenizer.decode(best_beam.numpy())
        return generated_text

