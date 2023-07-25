import os
import random
import json
import tensorflow as tf
import pandas as pd
import ast


def text_infilling(sequence, tokenizer, rate=0.15):
    # Convert token ids to words
    words = tokenizer.convert_ids_to_tokens(sequence)

    # Determine special tokens
    is_special = [True if word in tokenizer.all_special_tokens else False for word in words]

    word_list = []
    temp_word = []
    for word, special in zip(words, is_special):
        if not special:
            if word.startswith("##"):
                temp_word.append(word)
            else:
                if temp_word:
                    word_list.append(temp_word)
                temp_word = [word]
        else:
            if temp_word:
                word_list.append(temp_word)
            temp_word = []
            word_list.append([word])
    if temp_word:
        word_list.append(temp_word)

    # Randomly mask some words
    num_to_mask = int(len(word_list) * rate)
    mask_indices = random.sample(range(len(word_list)), num_to_mask)
    for i in mask_indices:
        for j in range(len(word_list[i])):
            word_list[i][j] = tokenizer.mask_token

    # Convert words back to ids
    masked_words = [word for sublist in word_list for word in sublist]
    masked_ids = tokenizer.convert_tokens_to_ids(masked_words)

    # Convert list to a TensorFlow tensor for downstream processing
    return tf.constant(masked_ids, dtype=tf.int32)


class DataLoader:
    """
    DataLoader class for our model, it will load dataset using epsilon-greedy strategy to ensure the data exposure and
    introduce a certain degree of randomness in data sampling. After selecting the file, we will convert it to
    tf.data.Dataset and return it.

    Epsilon-Greedy: With probability epsilon, we explore the whole data pool (choose a file randomly); With
    probability 1 - epsilon, we exploit what we have learned. We choose a file that has been selected the least
    amount of times so far.
    """

    def __init__(self, train_dataset_path, validation_dataset_path, usage_file_path):
        # load dataset and randomly get data from it
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = validation_dataset_path
        self.usage_file_path = usage_file_path
        if os.path.exists(usage_file_path):
            with open(usage_file_path, 'r') as usage_file:
                data = json.load(usage_file)
                self.file_usage_counts = data.get('usage_counts', {})
                self.steps = data.get('steps', 0)

        else:
            self.file_usage_counts = {}
            self.steps = 0
            self.save_usage_counts()

    def save_usage_counts(self):
        data = {'usage_counts': self.file_usage_counts, 'steps': self.steps}
        with open(self.usage_file_path, 'w') as usage_file:
            json.dump(data, usage_file)

    def select_next_files(self, mode='train'):
        if mode == 'train':
            dataset_path = self.train_dataset_path
        else:
            dataset_path = self.val_dataset_path
        all_files = [file for file in os.listdir(dataset_path) if file.endswith(".csv.bz2")]
        epsilon = 0.9 * tf.math.exp(-0.001 * self.steps) + 0.1
        # epsilon-greedy strategy
        if random.random() < epsilon or not self.file_usage_counts:
            # explore
            chosen_file = random.choice(all_files)
        else:
            # exploit
            file_usage = {file: self.file_usage_counts.get(file, 0) for file in all_files}
            chosen_file = min(file_usage, key=file_usage.get)
        # increment file usage count
        self.file_usage_counts[chosen_file] = self.file_usage_counts.get(chosen_file, 0) + 1
        self.steps += 1
        return chosen_file

    def load(self, tokenizer, mode='train', batch_size=64, mask_rate=0.15):
        chosen_file = self.select_next_files(mode)
        file_path = os.path.join(self.train_dataset_path if mode == 'train' else self.val_dataset_path, chosen_file)

        # Read the CSV file as a pandas DataFrame, and convert the string of token IDs into list of integers
        frame = pd.read_csv(file_path, names=['text'])
        frame['text'] = frame['text'].apply(lambda x: ast.literal_eval(x))

        # Convert the DataFrame to a list of lists and then shuffle
        data_list = frame['text'].to_list()
        random.shuffle(data_list)

        # Use the map function to apply text_infilling to each sequence in the dataset
        infilled_data = data_list.map(lambda x: text_infilling(x, tokenizer, rate=mask_rate))

        # Zip the datasets together to form (masked, unmasked) pairs, batch, and cache the data
        dataset = tf.data.Dataset.zip((infilled_data, data_list)).batch(batch_size).cache()

        self.save_usage_counts()

        return dataset
