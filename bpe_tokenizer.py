import os
import argparse
import concurrent.futures
from transformers import AutoTokenizer
from abc import ABCMeta, abstractmethod

class DataTokenizer(metaclass=ABCMeta):
    """
    A wrapper class for tokenizing sentences using the Hugging Face transformers library.

    Methods:
        tokenize(sentence): Tokenizes a sentence and returns the tokenized input IDs.
    """

    @abstractmethod
    def tokenize(self, sentence):
        """
        Tokenizes a sentence and returns the tokenized input IDs.

        Args:
            sentence (str): The input sentence to be tokenized.

        Returns:
            List[int]: Tokenized input IDs.
        """
        pass

class BPETrainer(DataTokenizer):
    def __init__(self, model="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def train(self, data_dir, batch_size=1000, vocab_size=50357, save=True, save_fp='tokenizer/adapted-tokenizer'):
        """
        Trains the tokenizer on a new corpus using BPE.

        Args:
            data_dir (str): Corpus directory path.
            batch_size (int): Batch size for reading files. Default is 1000.
            vocab_size (int): Target vocabulary size for adapted tokenizer. Default is 50000.
            save (bool): Whether to save the adapted tokenizer. Default is True.
            save_fp (str): File path to save the adapted tokenizer. Default is 'tokenizer/adapted-tokenizer'.
        """
        training_corpus = self.read_batch_of_files(data_dir, batch_size)
        self.tokenizer = self.tokenizer.train_new_from_iterator(training_corpus, vocab_size)
        if save:
            self.save(save_fp)

    def tokenize(self, sentence):
        return self.tokenizer.encode(sentence)

    def read_batch_of_files(self, data_dir, batch_size, num_workers=4):
        filenames = self.get_filenames(data_dir)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for start_idx in range(0, len(filenames), batch_size):
                batch_filenames = filenames[start_idx : start_idx + batch_size]
                batch_contents = []
                future_to_filename = {executor.submit(self.read_file, filename): filename for filename in batch_filenames}
                for future in concurrent.futures.as_completed(future_to_filename):
                    filename = future_to_filename[future]
                    content = future.result()
                    batch_contents.append(content)

                yield batch_contents

    def get_filenames(self, data_dir):
        filenames = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                filenames.append(os.path.join(root, file))
        return filenames

    def read_file(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    def save(self, fp):
        self.tokenizer.save_pretrained(fp)

def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a new corpus.")
    parser.add_argument("data_dir", type=str, help="Corpus directory path")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for reading files (default: 1000)")
    parser.add_argument("--vocab_size", type=int, default=50357, help="Target vocabulary size (default: 50357)")
    parser.add_argument("--save", action="store_true", help="Save the adapted tokenizer")
    parser.add_argument("--save_fp", type=str, default="tokenizer/adapted-tokenizer", help="File path to save the tokenizer (default: 'tokenizer/adapted-tokenizer')")

    args = parser.parse_args()

    bpe_trainer = BPETrainer()
    bpe_trainer.train(data_dir=args.data_dir, batch_size=args.batch_size, vocab_size=args.vocab_size, save=args.save, save_fp=args.save_fp)

if __name__ == "__main__":
    main()
