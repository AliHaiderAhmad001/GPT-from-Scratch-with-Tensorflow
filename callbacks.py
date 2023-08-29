import tensorflow as tf

class GPTCallbacks(tf.keras.callbacks.Callback):
    """
    Custom callback to monitor the validation loss during training and save the best model.

    Args:
        config: Configuration object containing hyperparameters.

    Attributes:
        checkpoint_filepath: Filepath to save the best model.
        patience: Number of epochs to wait for improvement in validation loss.
        best_loss: Best validation loss observed during training.

    Methods:
        on_epoch_end: Called at the end of each epoch to monitor the validation loss.

    """

    def __init__(self, config):
        super(GPTCallbacks, self).__init__()
        self.checkpoint_filepath = config.checkpoint_filepath
        self.patience = config.patience
        self.best_loss = float('inf')  # Initialize with a very large value for the first comparison

    def on_epoch_end(self, epoch, logs={}):
        """
        Callback function called at the end of each epoch to monitor the validation loss.

        Args:
            epoch: The current epoch number.
            logs: Dictionary containing training and validation metrics.

        """
        # Access the validation loss from the logs dictionary
        val_loss = logs.get('val_loss')

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.model.save(self.checkpoint_filepath)
            print('The best model has been saved at epoch #{}'.format(epoch))
        elif self.patience:
            self.patience -= 1
            if self.patience == 0:
                self.model.stop_training = True
                print('Training stopped. No improvement after {} epochs.'.format(epoch))
