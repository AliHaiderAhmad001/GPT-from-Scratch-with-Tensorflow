import tensorflow as tf
"""
Note that the ID of the filler code I use is 50375, and this may change depending on the tokenizer used. This also changes if you train tokenize.
"""
def loss_fn(label, pred):
    """
    Computes the masked Sparse Categorical Cross Entropy (SCCE) loss between the predicted and target labels.

    Args:
        label: Target label tensor.
        pred: Predicted logit tensor.

    Returns:
        Masked loss value.
    """
    # Create a mask to ignore padded tokens
    mask = label != 50357

    # Use Sparse Categorical Cross Entropy loss with no reduction
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    # Compute the loss without reducing, which will return a loss value for each token
    loss = loss_object(label, pred)

    # Apply the mask to ignore padded tokens in the loss calculation
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    # Compute the average loss over non-padded tokens
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss
