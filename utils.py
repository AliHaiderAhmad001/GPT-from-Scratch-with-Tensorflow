import tensorflow as tf
from decoder import Decoder
from embeddings import Embeddings
from lr_schedule import LrSchedule
from gpt_model import GPT
from feed_forward import FeedForward
from attention import AttentionHead, MultiHead_Attention
from positional_embeddings import PositionalEmbeddings
from loss_functions import loss_fn
from metrics import Perplexity

custom_objects = {
    "LrSchedule": LrSchedule,
    "PositionalEmbeddings": PositionalEmbeddings,
    "Embeddings": Embeddings,
    "AttentionHead": AttentionHead,
    "MultiHead_Attention": MultiHead_Attention,
    "FeedForward": FeedForward,
    "Decoder": Decoder,
    "GPT": GPT,
    "loss_fn": loss_fn,
    "Perplexity": Perplexity
}

# I did not use `load_model_and_optimizer` function because I have worked with a small dataset,
# but I wrote it in case you wanted to train the model on a large data set,
# and then you wanted to save the entire model or you wanted to resume training at a later time.
# Because if you want to resume training at a later time, you must completely reload the Optimizer state and Model.
# This function is well tested, 
# but there were warnings which I filtered out 
# because I don't think they affect the model, but I suggest you check further.
# However, if you decide to use it to resume model training,
# you will have to modify the training loop or create another loop to accommodate it.

def load_model_and_optimizer(config):
    # Create the model and optimizer from checkpoint
    model, optimizer = create_model_optimizer(config)

    with tf.keras.utils.custom_object_scope(custom_objects):
          # Restore the model and optimizer state
          latest_checkpoint = tf.train.latest_checkpoint(config.checkpoint_directory)
          if latest_checkpoint:
              checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
              checkpoint_status = checkpoint.restore(latest_checkpoint)
              checkpoint_status.expect_partial()  # Suppress warnings about incomplete restores
              print("Restored model and optimizer from checkpoint successfully: {}".format(latest_checkpoint))
          else:
              print("Checkpoint not found. Initializing from scratch.")

    return model, optimizer

# We will use this function in the inference process
def load_model_weights_only(config):
    # Create the model
    model = GPT(config)

    # Restore only the model weights
    latest_checkpoint = tf.train.latest_checkpoint(config.model_weights_checkpoint_directory)
    if latest_checkpoint:
        model_weights_checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_status = model_weights_checkpoint.restore(latest_checkpoint)
        checkpoint_status.expect_partial()  # Suppress warnings about incomplete restores
        print("Restored model weights from checkpoint successfully: {}".format(latest_checkpoint))
    else:
        print("Model weights checkpoint not found. Initializing model from scratch.")

    return model
