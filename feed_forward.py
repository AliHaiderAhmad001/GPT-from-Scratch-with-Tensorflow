class FeedForward(tf.keras.layers.Layer):
    """
    Feed-forward layer implementation.

    Args:
        config: Configuration object containing hyperparameters.

    Attributes:
        supports_masking: Boolean indicating if the layer supports masking.
        fc1: First dense layer.
        fc2: Second dense layer.
        dropout: Dropout layer.
    """

    def __init__(self, config, name=None, **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.supports_masking = True
        self.fc1 = tf.keras.layers.Dense(config.intermediate_fc_size, activation=tf.keras.activations.gelu)
        self.fc2 = tf.keras.layers.Dense(config.hidden_size)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_state, training=False):
        """
        Applies feed-forward transformation to the input hidden state.

        Args:
            hidden_state: Hidden state tensor (batch_size, sequence_length, hidden_size).
            training: Boolean indicating whether the model is in training mode or inference mode.

        Returns:
            Updated hidden state after applying feed-forward transformation.
        """
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.dropout(hidden_state, training=training)
        hidden_state = self.fc2(hidden_state)
        return hidden_state

    def get_config(self):
        """
        Returns the configuration of the feed-forward layer.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "fc1": self.fc1,
            "fc2": self.fc2,
            "dropout": self.dropout,
        })
        return config
