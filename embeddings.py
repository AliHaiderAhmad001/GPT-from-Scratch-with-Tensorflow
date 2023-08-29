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
        seq_length = input_ids.shape[1]
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
        vocab_size: Vocabulary size.

    Attributes:
        token_embeddings (tf.keras.layers.Embedding): Token embedding layer.
        dropout (tf.keras.layers.Dropout): Dropout layer for regularization.
        norm (tf.keras.layers.LayerNormalization): Layer normalization for normalization.
    """

    def __init__(self, config, vocab_size, name = None,  **kwargs):
        super(Embeddings, self).__init__(name=name, **kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim= vocab_size, output_dim=config.hidden_size
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
        return tf.math.not_equal(inputs, 0)

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
