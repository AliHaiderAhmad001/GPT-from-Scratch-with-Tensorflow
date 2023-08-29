import os
import argparse
import concurrent.futures
from transformers import AutoTokenizer

class BPE():
    def __init__(self, data_dir, model="gpt2"):
        """
        Initializes the BPE tokenizer adaptation.

        Args:
            data_dir: corpus diractory.
            model (str): Pretrained model name or path. Default is "gpt2".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.data_dir = data_dir

    def train(self, batch_size=1000, vocab_size=50300, save=True,
              save_fp='tokenizer/adapted-tokenizer'):
        """
        Trains the tokenizer on a new corpus using BPE.

        Args:
            batch_size (int): Batch size for reading files. Default is 1000.
            vocab_size (int): Target vocabulary size for adapted tokenizer. Default is 50000.
            save (bool): Whether to save the adapted tokenizer. Default is True.
            save_fp (str): File path to save the adapted tokenizer. Default is 'tokenizer/adapted-tokenizer'.

        """
        training_corpus = self.read_batch_of_files(batch_size)
        self.tokenizer = self.tokenizer.train_new_from_iterator(training_corpus, vocab_size)
        if save:
            self.save(save_fp)

    def read_batch_of_files(self, batch_size, num_workers=4):
        """
        Reads batches of file contents in parallel.

        Args:
            batch_size (int): Batch size for reading files.
            num_workers (int): Number of worker threads/processes. Default is 4.

        Yields:
            list: A list containing the content of each file in the batch.
        """
        filenames = self.get_filenames()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for start_idx in range(0, len(filenames), batch_size):
                batch_filenames = filenames[start_idx : start_idx + batch_size]
                batch_contents = []
                # Use executor.map to parallelize file reading
                future_to_filename = {executor.submit(self.read_file, filename): filename for filename in batch_filenames}
                for future in concurrent.futures.as_completed(future_to_filename):
                    filename = future_to_filename[future]
                    content = future.result()
                    batch_contents.append(content)

                yield batch_contents

    def get_filenames(self, data_dir):
        """
        Retrieves the filenames of the training corpus.

        Returns:
            list: A list of filenames.
        """
        filenames = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                filenames.append(os.path.join(root, file))
        return filenames

    def read_file(self, filename):
        """
        Reads the content of a file.

        Args:
            filename (str): File path to read.

        Returns:
            str: Content of the file.
        """
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    def save(self, fp):
        """
        Saves the adapted tokenizer.

        Args:
            fp (str): File path to save the tokenizer.
        """
        self.tokenizer.save_pretrained(fp)


def main(args):
    data_dir = args.data_dir
    bpe = BPE(data_dir)

    bpe.train(batch_size=args.batch_size, vocab_size=args.vocab_size, save=True, save_fp=args.save_fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPE Tokenizer Training")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to the corpus directory")
    parser.add_argument("--batch_size", type=int, default=50000, help="Batch size for reading files")
    parser.add_argument("--vocab_size", type=int, default=50357, help="Target vocabulary size")
    parser.add_argument("--save_fp", type=str, default="tokenizer/adapted-tokenizer", help="File path for saving the tokenizer")

    args = parser.parse_args()
    main(args)

"""
When you run the script, you can provide command-line arguments like this:

python train_bpe_tokenizer.py --data_dir /path/to/your/corpus --batch_size 10000 --vocab_size_target 50000 --save_fp tokenizer/my-tokenizer
"""

