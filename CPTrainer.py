import tensorflow as tf
import Transformer
from DataLoader import DataLoader
from tensorflow.python.keras.metrics import Mean
from CPTModel import CPTModel
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
import time
from transformers import BertTokenizer
import datetime


class CPTrainer:
    def __init__(self, model, loss_fn, optimizer, ckpt_manager):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loss = Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.val_loss = Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        self.ckpt_manager = ckpt_manager
        self.best_val_loss = float('inf')

    def evaluate(self, inp, tar_u, tar_g, training, enc_padding_mask, look_ahead_mask_u, dec_padding_mask_u,
                 look_ahead_mask_g, dec_padding_mask_g):
        prediction_u, _ = self.model(inp, tar_u, training, enc_padding_mask, look_ahead_mask_u, dec_padding_mask_u,
                                     task='understanding')
        prediction_g, _ = self.model(inp, tar_g, training, enc_padding_mask, look_ahead_mask_g, dec_padding_mask_g,
                                     task='generation')
        loss_u = self.loss_fn(tar_u, prediction_u)
        loss_g = self.loss_fn(tar_g, prediction_g)
        loss = loss_u + loss_g
        return loss, prediction_u, prediction_g

    @tf.function
    def train_step_u(self, inp, tar_u, training, enc_padding_mask, look_ahead_mask_u, dec_padding_mask_u):
        with tf.GradientTape() as tape:
            prediction_u, _ = self.model(inp, tar_u, training, enc_padding_mask, look_ahead_mask_u, dec_padding_mask_u,
                                         task='understanding')
            loss_u = self.loss_fn(tar_u, prediction_u)
        gradients = tape.gradient(loss_u, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss_u)
        self.train_accuracy(tar_u, prediction_u)

    @tf.function
    def train_step_g(self, inp, tar_g, training, enc_padding_mask, look_ahead_mask_g, dec_padding_mask_g):
        with tf.GradientTape() as tape:
            prediction_g, _ = self.model(inp, tar_g, training, enc_padding_mask, look_ahead_mask_g, dec_padding_mask_g,
                                         task='generation')
            loss_g = self.loss_fn(tar_g, prediction_g)
        gradients = tape.gradient(loss_g, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss_g)
        self.train_accuracy(tar_g, prediction_g)

    @tf.function
    def val_step_u(self, inp, tar_u, training, enc_padding_mask, look_ahead_mask_u, dec_padding_mask_u):
        prediction_u, _ = self.model(inp, tar_u, training, enc_padding_mask, look_ahead_mask_u, dec_padding_mask_u,
                                     task='understanding')
        loss_u = self.loss_fn(tar_u, prediction_u)

        self.val_loss(loss_u)
        self.val_accuracy(tar_u, prediction_u)

    @tf.function
    def val_step_g(self, inp, tar_g, training, enc_padding_mask, look_ahead_mask_g, dec_padding_mask_g):
        prediction_g, _ = self.model(inp, tar_g, training, enc_padding_mask, look_ahead_mask_g, dec_padding_mask_g,
                                     task='generation')
        loss_g = self.loss_fn(tar_g, prediction_g)

        self.val_loss(loss_g)
        self.val_accuracy(tar_g, prediction_g)

    def train(self, train_u_dataset, train_g_dataset, val_u_dataset, val_g_dataset, num_epochs):
        for epoch in range(num_epochs):
            start = time.time()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()
            # Get one batch of data
            inp, tar = next(iter(train_u_dataset))

            # Create dummy masks
            enc_padding_mask, combined_mask, dec_padding_mask = Transformer.create_masks(inp, tar)

            # Call model on dummy data
            _ = model(inp, tar, training=False, enc_padding_mask=enc_padding_mask, look_ahead_mask=combined_mask,
                      dec_padding_mask=dec_padding_mask, task='understanding')

            # Training for understanding task
            for (batch, (inp, tar)) in enumerate(train_u_dataset):
                enc_padding_mask, combined_mask, dec_padding_mask = Transformer.create_masks(inp, tar)
                self.train_step_u(inp, tar, True, enc_padding_mask, combined_mask, dec_padding_mask)
            tf.summary.scalar('u_loss', self.train_loss.result(), step=epoch)
            tf.summary.scalar('u_accuracy', self.train_accuracy.result(), step=epoch)

            # Training for generation task
            for (batch, (inp, tar)) in enumerate(train_g_dataset):
                enc_padding_mask, combined_mask, dec_padding_mask = Transformer.create_masks(inp, tar)
                self.train_step_g(inp, tar, True, enc_padding_mask, combined_mask. dec_padding_mask)
            tf.summary.scalar('g_loss', self.train_loss.result(), step=epoch)
            tf.summary.scalar('g_accuracy', self.train_accuracy.result(), step=epoch)

            # Validation for understanding task
            for (batch, (inp, tar)) in enumerate(val_u_dataset):
                enc_padding_mask, combined_mask, dec_padding_mask = Transformer.create_masks(inp, tar)
                self.val_step_u(inp, tar, False, enc_padding_mask, combined_mask, dec_padding_mask)
            tf.summary.scalar('u_val_loss', self.val_loss.result(), step=epoch)
            tf.summary.scalar('u_val_accuracy', self.val_accuracy.result(), step=epoch)

            # Validation for generation task
            for (batch, (inp, tar)) in enumerate(val_g_dataset):
                enc_padding_mask, combined_mask, dec_padding_mask = Transformer.create_masks(inp, tar)
                self.val_step_g(inp, tar, False, enc_padding_mask, combined_mask, dec_padding_mask)
            tf.summary.scalar('g_val_loss', self.val_loss.result(), step=epoch)
            tf.summary.scalar('g_val_accuracy', self.val_accuracy.result(), step=epoch)

            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')

            ckpt_save_path = self.ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


# Static parameters
d_model = 768
warmup_step = 10000
max_seq_len = 512
batch_size = 256
train_epochs = 10
tokenizer = BertTokenizer(vocab_file="./dataset/bert_vocab.txt", do_lower_case=False)
input_vocab_size = output_vocab_size = tokenizer.vocab_size
checkpoint_path = "./checkpoints/train"
train_dataset_path = 'F:/datasets/train'
validation_dataset_path = 'F:/datasets/validation'
usage_file_path = './dataset/usage_file.json'

# initialize model, loss function and optimizer
model = CPTModel(num_layer_enc=10, num_layer_dec=2, d_model=d_model, num_head=12, dff=3072,
                 input_vocab_size=input_vocab_size, output_vocab_size=output_vocab_size,
                 pe_input=max_seq_len, pe_output=max_seq_len, rate=0.1)

loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.AdamW(learning_rate=Transformer.CustomSchedule(d_model, warmup_step),
                                      beta_1=0.9, beta_2=0.98, epsilon=1e-6, weight_decay=0.01)

# initialize train checkpoint and restore model's trainable parameters
ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# record training metrics
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()

# fetch train&validation dataset
loader = DataLoader(train_dataset_path, validation_dataset_path, usage_file_path)
understanding_dataset = loader.load(mode='train', batch_size=batch_size, tokenizer=tokenizer)
generation_dataset = loader.load(mode='train', batch_size=batch_size, tokenizer=tokenizer, shuffling=True)
val_dataset = loader.load(mode='validation', batch_size=batch_size, tokenizer=tokenizer)
generation_val_dataset = loader.load(mode='validation', batch_size=batch_size, tokenizer=tokenizer, shuffling=True)
# initialize trainer class
trainer = CPTrainer(model, loss_fn, optimizer, ckpt_manager)
# start training
trainer.train(understanding_dataset, generation_dataset, val_dataset, generation_val_dataset, num_epochs=train_epochs)
