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
