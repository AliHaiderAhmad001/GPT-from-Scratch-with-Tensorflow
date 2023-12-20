class Config:
    """
    Configuration class for GPT-based text generation model.

    Attributes:
        tokenizer_path (str): The path to the pre-trained tokenizer.
        data_dir (str): Directory containing the training data.
        checkpoint_directory (str): Directory to save model and optimizer checkpoints.
        model_weights_checkpoint_directory (str): Directory to save model wights checkpoints.
        shuffle (bool): Whether to shuffle the training data.
        lower_case (bool): Whether to convert text to lowercase during preprocessing.
        remove_punctuation (bool): Whether to remove punctuation during preprocessing.
        sequence_length (int): The maximum sequence length for input data.
        buffer_size (int): Size of the data buffer for shuffling.
        batch_size (int): Batch size for training.
        seed (int): Random seed for reproducibility.
        vocab_size (int): Vocabulary size, including the PAD token.
        hidden_size (int): Size of the hidden layers in the model.
        intermediate_fc_size (int): Size of intermediate fully connected layers.
        warmup_steps (int): Number of warm-up steps for learning rate scheduling.
        max_learning_rate (float): Maximum learning rate for training.
        hidden_dropout_prob (float): Dropout probability for hidden layers.
        num_heads (int): Number of attention heads in the model.
        final_dropout_prob (float): Dropout probability for the final layer.
        num_blocks (int): Number of transformer blocks in the model.
        num_epochs (int): Number of training epochs.
        patience (int): Number of epochs with no improvement to trigger early stopping.
    """

    def __init__(self):
        self.tokenizer_path = "tokenizer/adapted-tokenizer"
        self.data_dir = 'aclImdb'
        self.checkpoint_directory = 'tmp/checkpoint'
        self.model_weights_checkpoint_directory = 'tmp/weights_checkpoint'
        self.shuffle = True
        self.lower_case = True
        self.remove_punctuation = True
        self.sequence_length = 128
        self.buffer_size = 42500
        self.batch_size = 64
        self.seed = 2023
        self.vocab_size = 50357+1  # Because we have added the PAD token.
        self.hidden_size = 256
        self.intermediate_fc_size = self.hidden_size * 4
        self.warmup_steps = 4000
        self.max_learning_rate = 2.5e-4
        self.total_number_of_training_samples = 42500
        self.hidden_dropout_prob = 0.1
        self.num_heads = 4
        self.final_dropout_prob = 0.5
        self.num_blocks = 2
        self.num_epochs = 20
        self.patience = 4
        self.layer_norm_epsilon = 1e-6
