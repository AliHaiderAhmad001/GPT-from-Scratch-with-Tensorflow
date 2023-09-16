import tensorflow as tf

class LrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule for training a model.

    This class implements a learning rate schedule that combines linear warmup
    followed by a cosine annealing schedule. It is designed to be used as the
    learning rate schedule for the optimizer during training.

    Args:
        config: Configuration object containing schedule hyperparameters.

    Attributes:
        warmup_steps: Number of warmup steps during which the learning rate increases linearly.
        max_learning_rate: Maximum learning rate reached after warmup.
        total_steps: Total number of training steps.
        learning_rate_schedule: Learning rate schedule for warmup phase.
        cosine_schedule: Learning rate schedule for cosine annealing phase.

    Methods:
        __call__: Returns the learning rate for a given training step.
        get_config: Returns the configuration dictionary of the learning rate schedule.
    """

    def __init__(self, config, total_number_of_training_samples):
        super(LrSchedule, self).__init__()
        self.warmup_steps = config.warmup_steps
        self.max_learning_rate = config.max_learning_rate
        self.total_steps = config.num_epochs * (total_number_of_training_samples // config.batch_size)
        self.learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.0,
            decay_steps=self.warmup_steps,
            end_learning_rate=self.max_learning_rate,
            power=1.0  # Linear warmup
        )
        self.cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.max_learning_rate,
            decay_steps=self.total_steps - self.warmup_steps
        )

    def __call__(self, step):
        """
        Returns the learning rate for a given training step.

        Args:
            step: Training step.

        Returns:
            Learning rate for the given step.
        """
        def learning_rate_fn(step):
            if step < self.warmup_steps:
                return self.learning_rate_schedule(step)
            return self.cosine_schedule(step - self.warmup_steps)

        return tf.cond(step < self.warmup_steps,
                       lambda: self.learning_rate_schedule(step),
                       lambda: self.cosine_schedule(step - self.warmup_steps))


    def get_config(self):
        """
        Returns the configuration dictionary of the learning rate schedule.

        Returns:
            Configuration dictionary.
        """
        return {
            "warmup_steps": self.warmup_steps,
            "max_learning_rate": self.max_learning_rate,
            "total_steps": self.total_steps,
            "learning_rate_schedule": self.learning_rate_schedule,
            "cosine_schedule": self.cosine_schedule,
        }
