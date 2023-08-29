class Decoder(tf.keras.layers.Layer):
    """
    Decoder layer implementation.

    Args:
        config: Configuration object containing hyperparameters.

    Attributes:
        masked_multihead_attention: Masked multi-head attention layer.
        norm1: Layer normalization layer.
        norm2: Layer normalization layer.
        feed_forward: Feed-forward layer.
        dropout: Dropout layer.
    """

    def __init__(self, config, name=None, **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.masked_multihead_attention = MultiHead_Attention(config)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon)
        self.feed_forward = FeedForward(config)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_state, mask=None, training=False):
        """
        Applies the decoder layer to the input hidden state.

        Args:
            hidden_state: Hidden state tensor.
            mask: mask tensor.
            training: Boolean indicating if the model is in training mode.

        Returns:
            Updated hidden state after applying the decoder layer.
        """
        causal_mask = self.get_causal_attention_mask(hidden_state)

        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output = self.masked_multihead_attention(hidden_state, hidden_state, hidden_state, mask=causal_mask)
        attention_output = self.dropout(attention_output, training=training)
        hidden_state = self.norm1(attention_output + hidden_state)

        feed_forward_output = self.feed_forward(hidden_state)
        feed_forward_output = self.dropout(feed_forward_output, training=training)
        hidden_state = self.norm2(feed_forward_output + hidden_state)

        return hidden_state

    def get_causal_attention_mask(self, inputs):
        """
        Generates the causal attention mask.

        Args:
            inputs: Input tensor.

        Returns:
            Causal attention mask tensor.
        """
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype=tf.int32)
        mask = tf.reshape(mask, (1, sequence_length, sequence_length))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def get_config(self):
        """
        Returns the configuration of the decoder layer.

        Returns:
            Configuration dictionary.
        """
        config = super(Decoder, self).get_config()
        config.update({
            "masked_multihead_attention": self.masked_multihead_attention,
            "norm1": self.norm1,
            "norm2": self.norm2,
            "feed_forward": self.feed_forward,
            "dropout": self.dropout,
        })
        return config
