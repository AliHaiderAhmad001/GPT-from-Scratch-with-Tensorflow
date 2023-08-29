import tensorflow as tf

class LrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, config):
        super(LrSchedule, self).__init__()
        self.warmup_steps = config.warmup_steps
        self.max_learning_rate = config.max_learning_rate
        self.total_steps = config.num_epochs * (config.total_number_of_training_samples // config.batch_size)
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
        if step < self.warmup_steps:
            return self.learning_rate_schedule(step)
        return self.cosine_schedule(step - self.warmup_steps)

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "max_learning_rate": self.max_learning_rate,
            "total_steps": self.total_steps,
            "learning_rate_schedule": self.learning_rate_schedule,
            "cosine_schedule": self.cosine_schedule,
        }

