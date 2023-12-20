import tensorflow as tf
from positional_embeddings import PositionalEmbeddings

class Embeddings(tf.keras.layers.Layer):
    """
    Embeddings layer.

    This layer combines token embeddings with positional embeddings to create the final embeddings.

    Args:
        config (object): Configuration object containing parameters.

    Attributes:
        token_embeddings (tf.keras.layers.Embedding): Token embedding layer.
        dropout (tf.keras.layers.Dropout): Dropout layer for regularization.
    """

    def __init__(self, config, name = None,  **kwargs):
        super(Embeddings, self).__init__(name=name, **kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim= config.vocab_size, output_dim=config.hidden_size
        )
        self.PositionalInfo = PositionalEmbeddings(config)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

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
        })
        return config
