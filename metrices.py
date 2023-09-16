import tensorflow as tf

class Perplexity(tf.keras.metrics.Metric):
    def __init__(self, name='perplexity', **kwargs):
        super(Perplexity, self).__init__(name=name, **kwargs)
        self.loss_sum = self.add_weight(name='loss_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        mask = y_true != 50357
        # Apply the mask to ignore padded tokens in the loss calculation
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        self.loss_sum.assign_add(tf.reduce_sum(loss))
        self.count.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))

    def result(self):
        return tf.pow(2.0, self.loss_sum / self.count)


class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='masked_accuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.total_matches = self.add_weight(name='total_matches', initializer='zeros')
        self.total_tokens = self.add_weight(name='total_tokens', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_labels = tf.argmax(y_pred, axis=2)
        y_true = tf.cast(y_true, pred_labels.dtype)
        match = y_true == pred_labels
        mask = y_true != 50357
        match = match & mask
        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        self.total_matches.assign_add(tf.reduce_sum(match))
        self.total_tokens.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total_matches / self.total_tokens
