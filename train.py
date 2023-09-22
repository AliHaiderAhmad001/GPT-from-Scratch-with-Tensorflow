import argparse
import time
import tensorflow as tf
import random as python_random
from config import Config
from gpt_model import GPT
from lr_schedule import LrSchedule
from loss_functions import loss_fn
from metrics import Perplexity

def parse_args():
    parser = argparse.ArgumentParser(description="GPT Training Script")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    return parser.parse_args()


def create_model_optimizer(config):
    model = GPT(config)

    # Create the learning rate schedule
    lr = LrSchedule(config)

    optimizer = tf.keras.optimizers.Adam(
        lr,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    return model, optimizer

def save_model_and_optimizer(model, optimizer, checkpoint_manager, epoch):
    # Save the model and optimizer state
    checkpoint_name = checkpoint_manager.save()
    print("Saved checkpoint for epoch {}: {}".format(epoch, checkpoint_name))

def save_model_weights_only(model, model_weights_manager, epoch):
    # Save model weights only
    model_weights_checkpoint_name = model_weights_manager.save()
    print("Saved model weights for epoch {}: {}".format(epoch, model_weights_checkpoint_name))

def train_one_epoch(model, optimizer, train_gr, loss_fn, train_ppe_metric):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_ppe_metric.update_state(y, logits)
        return loss_value

    for step, (x_batch_train, y_batch_train) in enumerate(train_gr):
        tr_loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, tr_loss_value)
            )
            print("Seen so far: %d samples" % ((step + 1) * config.batch_size))

def evaluate_one_epoch(model, val_gr, loss_fn, val_ppe_metric):
    @tf.function
    def test_step(x, y):
        logits = model(x, training=False)
        val_ppe_metric.update_state(y, logits)
        loss_value = loss_fn(y, logits)
        return loss_value

    val_loss_value = 0.0
    for x_batch_val, y_batch_val in val_gr:
        val_loss_value += test_step(x_batch_val, y_batch_val)
    return val_loss_value

def main_training_loop(config, resume_training=False):
    python_random.seed(config.seed)
    tf.random.set_seed(config.seed)

    train_gr = EnglishDataStreamer(config, 'train')
    val_gr = EnglishDataStreamer(config, 'valid')
    # Prepare the metrics.
    train_ppe_metric = Perplexity()
    val_ppe_metric = Perplexity()

    # Create the model and optimizer
    if resume_training:
        model, optimizer = load_model_and_optimizer(config)
    else:
        model, optimizer = create_model_optimizer(config)

    # Create a checkpoint for both the model and optimizer
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_directory = config.checkpoint_directory
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=1)

    # Create a separate checkpoint for model weights only
    model_weights_checkpoint = tf.train.Checkpoint(model=model)
    model_weights_checkpoint_directory = config.model_weights_checkpoint_directory
    model_weights_manager = tf.train.CheckpointManager(model_weights_checkpoint, model_weights_checkpoint_directory, max_to_keep=1)

    # Early stopping parameters
    best_val_loss = float("inf")
    patience = config.patience
    wait = 0

    for epoch in range(1, config.num_epochs+1):
        print("\n##### Start of epoch %d #####" % (epoch,))
        start_time = time.time()

        # Training
        train_one_epoch(model, optimizer, train_gr, loss_fn, train_ppe_metric)

        # Display metrics at the end of each epoch.
        train_ppe = train_ppe_metric.result()
        print("Training perplexity over epoch: %.4f" % (float(train_ppe),))

        # Reset training metrics at the end of each epoch
        train_ppe_metric.reset_states()

        # Evaluation
        val_loss_value = evaluate_one_epoch(model, val_gr, loss_fn, val_ppe_metric)
        print("Training loss over epoch: %.4f" % (val_loss_value,))

        val_ppe = val_ppe_metric.result()
        val_ppe_metric.reset_states()

        print("Validation perplexity: %.4f" % (val_ppe,))
        print("Time taken: %.2fs" % (time.time() - start_time))

        # Early Stopping Check
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            wait = 0
            # Save the model and optimizer
            save_model_and_optimizer(model, optimizer, manager, epoch)

            # Save model weights only
            save_model_weights_only(model, model_weights_manager, epoch)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered. No improvement in validation loss.")
                break

if __name__ == "__main__":
    args = parse_args()
    config = Config() 
    main_training_loop(config, resume_training=args.resume)


