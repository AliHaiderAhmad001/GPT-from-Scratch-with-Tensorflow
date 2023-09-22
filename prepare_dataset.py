import os
import shutil
import random
import multiprocessing
import argparse

# Define a function for moving files from one directory to another
def move_files(src_files, dest_dir):
    for file_to_move in src_files:
        destination_path = os.path.join(dest_dir, os.path.basename(file_to_move))
        try:
            shutil.move(file_to_move, destination_path)
        except Exception as e:
            print(f"Error moving file {file_to_move}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Move files from the test directory to the validation directory.")
    parser.add_argument("test_dir", type=str, help="Path to the test directory")
    parser.add_argument("validation_dir", type=str, help="Path to the validation directory")
    parser.add_argument("--num_files_to_move", type=int, default=2500, help="Number of files to move (default: 2500)")

    args = parser.parse_args()

    # Define your source and validation directories
    test_dir = args.test_dir
    validation_dir = args.validation_dir
    num_files_to_move = args.num_files_to_move

    # Create the validation directory if it doesn't exist
    os.makedirs(validation_dir, exist_ok=True)

    # Create separate 'pos' and 'neg' subdirectories within the validation directory
    validation_pos_dir = os.path.join(validation_dir, 'pos')
    validation_neg_dir = os.path.join(validation_dir, 'neg')
    os.makedirs(validation_pos_dir, exist_ok=True)
    os.makedirs(validation_neg_dir, exist_ok=True)

    # Define the subdirectories
    test_pos_dir = os.path.join(test_dir, 'pos')
    test_neg_dir = os.path.join(test_dir, 'neg')

    # List files in the test 'pos' and 'neg' directories
    test_pos_files = [os.path.join(test_pos_dir, filename) for filename in os.listdir(test_pos_dir)]
    test_neg_files = [os.path.join(test_neg_dir, filename) for filename in os.listdir(test_neg_dir)]

    # Randomly shuffle the lists of files
    random.seed(42)  # Set a random seed for reproducibility
    random.shuffle(test_pos_files)
    random.shuffle(test_neg_files)

    # Split the files into chunks for parallel processing
    chunk_size = num_files_to_move // multiprocessing.cpu_count()
    test_pos_chunks = [test_pos_files[i:i + chunk_size] for i in range(0, num_files_to_move, chunk_size)]
    test_neg_chunks = [test_neg_files[i:i + chunk_size] for i in range(0, num_files_to_move, chunk_size)]

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Move files in parallel
    pool.starmap(move_files, [(chunk, validation_pos_dir) for chunk in test_pos_chunks])
    pool.starmap(move_files, [(chunk, validation_neg_dir) for chunk in test_neg_chunks])

    # Close the pool of worker processes
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()

