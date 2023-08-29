import tensorflow as tf

def masked_accuracy(label, pred):
    """
    Computes the masked accuracy between the predicted and target labels.

    Args:
        label: Target label tensor.
        pred: Predicted label tensor.

    Returns:
        Masked accuracy value.
    """
    # Get the predicted labels by taking the argmax along the last dimension
    pred_labels = tf.argmax(pred, axis=2)

    # Convert the target labels to the same data type as the predicted labels
    label = tf.cast(label, pred_labels.dtype)

    # Compute a binary tensor for matching predicted and target labels
    match = label == pred_labels

    # Create a mask to ignore padded tokens
    mask = label != 0

    # Apply the mask to the matching tensor
    match = match & mask

    # Convert the binary tensor to floating-point values
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    # Compute the accuracy over non-padded tokens
    return tf.reduce_sum(match) / tf.reduce_sum(mask)

def perplexity(label, pred):
    """
    Computes the perplexity metric based on the Categorical Cross Entropy (CCE) loss.

    Args:
        label: Target label tensor.
        pred: Predicted logit tensor.

    Returns:
        Computed perplexity value.
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

    # Compute the sum of losses over non-padded tokens
    total_loss = tf.reduce_sum(loss)

    # Compute the total number of non-padded tokens
    total_tokens = tf.reduce_sum(mask)

    # Compute the average loss over non-padded tokens
    avg_loss = total_loss/total_tokens

    # Compute perplexity as 2 raised to the power of the average loss
    perplexity = 2 ** avg_loss

    return perplexity
