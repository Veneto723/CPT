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
    def train_step(self, inp, tar_u, tar_g, training, enc_padding_mask, look_ahead_mask_u, dec_padding_mask_u,
                   look_ahead_mask_g, dec_padding_mask_g):
        with tf.GradientTape as tape:
            loss, prediction_u, prediction_g = self.evaluate(inp, tar_u, tar_g, training, enc_padding_mask,
                                                             look_ahead_mask_u, dec_padding_mask_u,
                                                             look_ahead_mask_g, dec_padding_mask_g)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_u, prediction_u)
        self.train_accuracy(tar_g, prediction_g)

    @tf.function
    def val_step(self, inp, tar_u, tar_g, training, enc_padding_mask, look_ahead_mask_u, dec_padding_mask_u,
                 look_ahead_mask_g, dec_padding_mask_g):
        loss, prediction_u, prediction_g = self.evaluate(inp, tar_u, tar_g, training, enc_padding_mask,
                                                         look_ahead_mask_u, dec_padding_mask_u,
                                                         look_ahead_mask_g, dec_padding_mask_g)
        self.val_loss(loss)
        self.val_accuracy(tar_u, prediction_u)
        self.val_accuracy(tar_g, prediction_g)

    def train(self, train_dataset, val_dataset, num_epochs=10):
        for epoch in range(num_epochs):
            start = time.time()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            understanding_batches, generation_batches = train_dataset
            val_understanding_batches, val_generation_batches = val_dataset

            train_num_batch_u = tf.cast(tf.math.ceil(understanding_batches / 50), tf.int64)
            train_num_batch_g = tf.cast(tf.math.ceil(generation_batches / 50), tf.int64)
            val_num_batch_u = tf.cast(tf.math.ceil(val_understanding_batches / 50), tf.int64)
            val_num_batch_g = tf.cast(tf.math.ceil(val_generation_batches / 50), tf.int64)
            # Training for understanding task
            for (batch, (inp, tar)) in enumerate(understanding_batches):
                self.train_step(inp, tar, 'understanding', self.optimizer, self.train_loss, self.train_accuracy)
                if batch % 50 == 0 and batch != 0:
                    tf.summary.scalar('u_loss', self.train_loss.result(), step=(epoch * train_num_batch_u + batch))
                    tf.summary.scalar('u_accuracy', self.train_accuracy.result(),
                                      step=(epoch * train_num_batch_u + batch))

            # Training for generation task
            for (batch, (inp, tar)) in enumerate(generation_batches):
                self.train_step(inp, tar, 'generation', self.optimizer, self.train_loss, self.train_accuracy)
                if batch % 50 == 0 and batch != 0:
                    tf.summary.scalar('g_loss', self.train_loss.result(), step=(epoch * train_num_batch_g + batch))
                    tf.summary.scalar('g_accuracy', self.train_accuracy.result(),
                                      step=(epoch * train_num_batch_g + batch))

            # Validation for understanding task
            for (batch, (inp, tar)) in enumerate(val_understanding_batches):
                self.val_step(inp, tar, 'understanding', self.val_loss, self.val_accuracy)
                if batch % 50 == 0 and batch != 0:
                    tf.summary.scalar('u_val_loss', self.train_loss.result(), step=(epoch * val_num_batch_u + batch))
                    tf.summary.scalar('u_val_accuracy', self.train_accuracy.result(),
                                      step=(epoch * val_num_batch_u + batch))

            # Validation for generation task
            for (batch, (inp, tar)) in enumerate(val_generation_batches):
                self.val_step(inp, tar, 'generation', self.val_loss, self.val_accuracy)
                if batch % 50 == 0 and batch != 0:
                    tf.summary.scalar('g_val_loss', self.train_loss.result(), step=(epoch * val_num_batch_g + batch))
                    tf.summary.scalar('g_val_accuracy', self.train_accuracy.result(),
                                      step=(epoch * val_num_batch_g + batch))

            if epoch % 2 == 0:
                print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f} '
                      f'Accuracy {self.train_accuracy.result():.4f}')

            ckpt_save_path = self.ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


# TODO GLUE SQuAD to test our model

# Static parameters
d_model = 768
warmup_step = 10000
max_seq_len = 256  # BERT uses 512 as max_seq_len
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
train_dataset = loader.load(mode='train', batch_size=batch_size)
val_dataset = loader.load(mode='validation', batch_size=batch_size)
# initialize trainer class
trainer = CPTrainer(model, loss_fn, optimizer, ckpt_manager)
# start training
trainer.train(train_dataset=train_dataset, val_dataset=val_dataset, num_epochs=train_epochs)

"""
    TODO 在BERT的论文中，"we pre-train the model with sequence length of 128 for 90% of the steps. Then we train the rest
    10% of the steps of sequence of 512 to learn the positional encoding." 所以理论上我们是可以“自由”调整max_seq_len的，但是
    要考虑到模型的计算复杂性和服务器算力
"""
