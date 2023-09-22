import tensorflow as tf

class AttentionHead(tf.keras.layers.Layer):
    """
    Attention head implementation.

    Args:
        head_dim: Dimensionality of the attention head.

    Attributes:
        head_dim: Dimensionality of the attention head.
        query_weights: Dense layer for query projection.
        key_weights: Dense layer for key projection.
        value_weights: Dense layer for value projection.
    """

    def __init__(self, head_dim, name = None, **kwargs):
        super(AttentionHead, self).__init__(name=name, **kwargs)
        self.supports_masking = True  # Enable masking support
        self.head_dim = head_dim
        self.query_weights = tf.keras.layers.Dense(head_dim)
        self.key_weights = tf.keras.layers.Dense(head_dim)
        self.value_weights = tf.keras.layers.Dense(head_dim)

    def call(self, query, key, value, mask=None):
        """
        Applies attention mechanism to the input query, key, and value tensors.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            mask: Optional mask tensor.

        Returns:
            Updated value embeddings after applying attention mechanism.
        """
        query = self.query_weights(query)
        key = self.key_weights(key)
        value = self.value_weights(value)

        att_scores = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1])) / tf.math.sqrt(tf.cast(tf.shape(query)[-1], tf.float32))

        if mask is not None:
            mask = tf.cast(mask, dtype=tf.bool)
            att_scores = tf.where(mask, att_scores, tf.constant(-1e9, dtype=att_scores.dtype))

        att_weights = tf.nn.softmax(att_scores, axis=-1)
        n_value = tf.matmul(att_weights, value)

        return n_value

    def get_config(self):
        """
        Returns the configuration of the attention head layer.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "head_dim": self.head_dim,
            "query_weights": self.query_weights,
            "key_weights": self.key_weights,
            "value_weights": self.value_weights,
        })
        return config


class MultiHead_Attention(tf.keras.layers.Layer):
    """
    Multi-head attention layer implementation.

    Args:
        config: Configuration object containing hyperparameters.

    Attributes:
        supports_masking: Boolean indicating if the layer supports masking.
        hidden_size: Dimensionality of the hidden state.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        attention_heads: List of AttentionHead layers.
        fc: Fully connected layer for final projection.
    """

    def __init__(self, config, name=None, **kwargs):
        super(MultiHead_Attention, self).__init__(name=name, **kwargs)
        self.supports_masking = True
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.attention_heads = [AttentionHead(self.head_dim) for _ in range(self.num_heads)]
        self.fc = tf.keras.layers.Dense(config.hidden_size)

    def call(self, query, key, value, mask=None):
        """
        Applies multi-head attention mechanism to the input query, key, and value tensors.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            mask: Optional mask tensor.

        Returns:
            Updated hidden state after applying multi-head attention mechanism.
        """
        attention_outputs = [attention_head(query, key, value, mask=mask) for attention_head in self.attention_heads]
        hidden_state = tf.concat(attention_outputs, axis=-1)
        hidden_state = self.fc(hidden_state)
        return hidden_state

    def get_config(self):
        """
        Returns the configuration of the multi-head attention layer.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "attention_heads": self.attention_heads,
            "fc": self.fc,
        })
        return config
