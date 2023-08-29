import tensorflow as tf

class GPT(tf.keras.Model):
    """
    GPT model implementation.

    Args:
        config: Configuration object containing model hyperparameters.

    Attributes:
        embed_layer: Embeddings layer for the decoder inputs.
        decoder: List of decoder layers.
        dropout: Dropout layer for regularization.
        output_layer: Dense layer for output prediction.

    Methods:
        call: Forward pass of the GPT model.
        get_config: Returns the configuration dictionary of the GPT model.
    """

    def __init__(self, config, name=None, **kwargs):
        super(GPT, self).__init__(name=name, **kwargs)
        self.embed_layer = Embeddings(config, config.vocab_size, name="embeddings")
        self.decoder = [Decoder(config) for _ in range(config.num_blocks)]
        self.dropout = tf.keras.layers.Dropout(config.final_dropout_prob)
        self.output_layer = tf.keras.layers.Dense(config.vocab_size, name="output_layer")

    def call(self, inputs, training=False, mask=None):
        """
        Forward pass of the GPT model.

        Args:
            inputs: Input data.
            training: Boolean flag indicating whether the model is in training mode or not.
            mask: Optional mask for inputs.

        Returns:
            Output logits of the GPT model.
        """
        x_dec = self.embed_layer(inputs)

        for decoder_layer in self.decoder:
            x_dec = decoder_layer(x_dec, training=training)

        x_dec = self.dropout(x_dec, training=training)
        x_logits = self.output_layer(x_dec)
        # Remove the mask from the logits as it's not needed in the loss function
        x_logits._keras_mask = None

        return x_logits

    def get_config(self):
        """
        Returns the configuration dictionary of the GPT model.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "embed_layer": self.embed_layer,
            "decoder": self.decoder,
            "dropout": self.dropout,
            "output_layer": self.output_layer,
        })
        return config

