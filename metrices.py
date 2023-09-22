# Assumes that 50357 is the PAD token ID.

import tensorflow as tf

class Perplexity(tf.keras.metrics.Metric):
    """
    Custom metric for calculating perplexity.

    Perplexity is a measure of how well a probability distribution or probability model predicts a sample.
    It is commonly used in natural language processing tasks to evaluate the quality of language models.

    Attributes:
        name (str): Name of the metric.
    """

    def __init__(self, name='perplexity', **kwargs):
        """
        Initializes the Perplexity metric.

        Args:
            name (str, optional): Name of the metric. Defaults to 'perplexity'.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super(Perplexity, self).__init__(name=name, **kwargs)
        self.loss_sum = self.add_weight(name='loss_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the metric state based on true and predicted values.

        Args:
            y_true (tensor): True target values.
            y_pred (tensor): Predicted values.
            sample_weight (tensor, optional): Optional weighting of samples. Not used.

        Returns:
            None
        """
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        mask = y_true != 50357
        # Apply the mask to ignore padded tokens in the loss calculation
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        self.loss_sum.assign_add(tf.reduce_sum(loss))
        self.count.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))

    def result(self):
        """
        Computes the final perplexity metric.

        Returns:
            tensor: The computed perplexity value.
        """
        return tf.pow(2.0, self.loss_sum / self.count)
