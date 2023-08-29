import tensorflow as tf

def scce_masked_loss(label, pred):
    """
    Computes the masked Sparse Categorical Cross Entropy (SCCE) loss between the predicted and target labels.

    Args:
        label: Target label tensor.
        pred: Predicted logit tensor.

    Returns:
        Masked loss value.
    """
    # Create a mask to ignore padded tokens
    mask = label != 0

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


def cce_loss(label, pred):
    """
    Computes the Categorical Cross Entropy (CCE) loss with optional label smoothing.

    Args:
        label: Target label tensor.
        pred: Predicted logit tensor.

    Returns:
        Computed CCE loss value.
    """
    # Create a mask to ignore padded tokens
    mask = label != 0

    # Use Categorical Cross Entropy with optional label smoothing
    scc_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1, reduction='none')

    # Convert label to one-hot encoding
    label = tf.one_hot(tf.cast(label, tf.int32), config.target_vocab_size)

    # Compute the loss with the label smoothing
    loss = scc_loss(label, pred)

    # Apply the mask to ignore padded tokens in the loss calculation
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    # Compute the average loss over non-padded tokens
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss
