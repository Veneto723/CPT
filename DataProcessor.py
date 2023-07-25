from datasets import load_dataset
import pandas as pd
import jieba
from transformers import BertTokenizer
import time
import os
import pickle


def process_text(text, tokenizer, max_seq_len):
    """
    Process a piece of text by cleaning it, splitting it into sentences,
    tokenizing and encoding it, and padding it to the max sequence length.

    Args:
        text (str): The text to process.
        tokenizer: tokenizer to be used that tokenize and embed words to IDs
        max_seq_len (int): max_seq_len each sentence can hold
    Returns:
        list of int: The processed text, represented as a list of token IDs. [[...], [...]]
    """

    # Clean the text
    text = text.replace("\n", "")
    # Split the text into sentences if necessary
    sentences = split_sentence(text, max_seq_len) if len(text) > max_seq_len else [text]

    processed_sentences = []
    for sentence in sentences:
        # Skip empty sentences
        if not sentence:
            continue

        # Tokenize the sentence
        sentence = " ".join(jieba.lcut(sentence, cut_all=False))
        # Encode the sentence
        sentence = tokenizer.encode(sentence, add_special_tokens=True)

        # Pad the sentence
        if len(sentence) < max_seq_len:
            sentence += [0] * (max_seq_len - len(sentence))
        elif len(sentence) > max_seq_len:
            sentence = sentence[:max_seq_len]

        processed_sentences.append(sentence)

    return processed_sentences


class DataProcessor:
    """
    Class for processing large text datasets in chunks.

    Attributes:
        chunk_size (int): Number of data samples to process at a time.
        max_seq_len (int): Maximum sequence length for BERT encoding.
        dataset_path (str): Path to save the processed datasets.
        temp_path (str): Temporary path to save intermediate files.
        train_dataset_path (str): Path to save the processed training datasets.
        val_dataset_path (str): Path to save the processed validation datasets.
        file_path (str): Path to the dataset to process.
        progress (int): Indicates the progress of data processing.
    """

    def __init__(self, chunk_size, max_seq_len, dataset_path, file_path='', initialize=True, cache_dir=''):
        """
        Initializes the DataProcessor with the given parameters.

        Args:
            chunk_size (int): Number of data samples to process at a time.
            max_seq_len (int): Maximum sequence length for BERT encoding.
            dataset_path (str): Path to save the processed datasets.
            file_path (str, optional): Path to the dataset to process. Defaults to ''.
            initialize (bool, optional): Whether to initialize the processing. Defaults to True.
            cache_dir (str, optional): Path to download the dataset. Defaults to ''.
        """
        # Initialize class variables and create necessary directories.
        self.progress_file = "./progress.pickle"
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.dataset_path = dataset_path
        self.temp_path = os.path.join(self.dataset_path, "temp")
        self.train_dataset_path = os.path.join(self.dataset_path, "train")
        self.val_dataset_path = os.path.join(self.dataset_path, "validation")
        self.fine_tune_path = os.path.join(self.dataset_path, "fine_tune")
        self.tokenizer = BertTokenizer(vocab_file="./dataset/bert_vocab.txt", do_basic_tokenize=False)
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)
        os.makedirs(self.train_dataset_path, exist_ok=True)
        os.makedirs(self.val_dataset_path, exist_ok=True)
        os.makedirs(self.fine_tune_path, exist_ok=True)

        # If initialize is set to False, set file path. Otherwise, download and load the dataset.
        # Also try to load the progress, or set it to 0 if it does not exist.
        if not initialize:
            self.file_path = file_path
        else:
            self.dataset = load_dataset("shjwudp/chinese-c4", cache_dir=cache_dir)
            try:
                with open(self.progress_file, "rb") as file:
                    self.progress = pickle.load(file)
            except FileNotFoundError:
                self.progress = 0

    def extractor(self, output_file_name):
        chunk = pd.read_excel(self.file_path, names=['questions', 'answers'])
        chunk = chunk.applymap(lambda x: process_text(x, self.tokenizer, self.max_seq_len))
        # Flatten the list of processed sentences
        chunk = chunk.explode('questions')
        chunk = chunk.explode('answers')
        chunk.dropna(inplace=True)

        # Save the content to a csv file.
        chunk.to_csv(f"{os.path.join(self.fine_tune_path, output_file_name)}.csv.bz2", header=False, index=False,
                     compression='bz2')

    def initialize(self):
        """
        Initialize the processing task by downloading the dataset and processing it in chunks.
        """
        num_chunks = (len(self.dataset['train']) // self.chunk_size) + 1
        start_time = time.time()
        for i in range(self.progress, num_chunks):
            chunk_start_time = time.time()

            # Load data chunk and process it.
            data = self.dataset['train'][i * self.chunk_size: (i + 1) * self.chunk_size]
            chunk = pd.DataFrame(data, columns=['text'])
            chunk = chunk.applymap(lambda x: process_text(x, self.tokenizer, self.max_seq_len))

            # Flatten DataFrame and drop any rows with null values.
            chunk = chunk.explode('text')
            chunk.dropna(inplace=True)

            # Save the chunk to a csv file.
            chunk.to_csv(self.dataset_path + "sentences{:04}.csv.bz2".format(i), header=False,
                         index=False, compression='bz2')

            # Update the progress variable and save it.
            self.progress = i + 1
            with open("progress.pickle", "wb") as f:
                pickle.dump(self.progress, f)

            chunk_end_time = time.time()
            chunk_time = chunk_end_time - chunk_start_time
            total_elapsed_time = chunk_end_time - start_time
            avg_chunk_time = total_elapsed_time / self.progress
            remaining_chunks = num_chunks - self.progress
            estimated_remaining_time = remaining_chunks * avg_chunk_time

            print(f"Processed chunk {i + 1}/{num_chunks} in {chunk_time:.2f} seconds. "
                  f"Total elapsed time: {total_elapsed_time:.2f} seconds. "
                  f"Estimated remaining time: {estimated_remaining_time:.2f} seconds.")

        print("Data Process completion.")
        os.remove("./progress.pickle")


def split_sentence(text, max_len):
    """
    Split a sentence into multiple sentences, each with a maximum length.

    Args:
        text (str): The sentence to split.
        max_len (int): The maximum length of each split sentence.

    Returns:
        list of str: The split sentences.
    """
    if len(text) <= max_len:
        return [text]

    pivot = max_len // 2
    for i in range(pivot, max_len):
        if text[i] in [';', '!', '?', '。', '；', '！', '？']:
            return [text[:i + 1].strip()] + split_sentence(text[i + 1:].strip(), max_len)
    for i in range(pivot, 0, -1):
        if text[i] in [';', '!', '?', '。', '；', '！', '？']:
            return [text[:i + 1].strip()] + split_sentence(text[i + 1:].strip(), max_len)

    # if we reached here, there is no punctuation in the sentence, so we just ignore this sentence
    return []
