# GenesisMind-Building-GPT-from-Scratch

GPT "Generative Pre-trained Transformer" is the first version of the GPT series of models, revolutionized natural language processing with its autoregressive language modeling capabilities built on the Transformer architecture.

---

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Ubuntu-red.svg)](https://www.ubuntu.com/)
[![Tensorflow](https://img.shields.io/badge/Tensorflow-2-orange.svg)](https://tensorflow.org/)
![Transformers](https://img.shields.io/badge/transformers-4.36-yellow.svg)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

## Overview

This project is an implementation of the GPT (Generative Pre-trained Transformer) model from scratch using TensorFlow. It includes all the components of the model, such as the positional embeddings, attention mechanism, feed-forward layers, and more. 

**Important Note:** The goal of this project is to provide a deep understanding of the GPT architecture and its inner workings. So, it's mainly for educational purposes. You can fully understand the structure and working mechanism of this model here, and use the components I have implemented in your projects. Generally, if you want to use the project to train your language model with big data, you may need to modify the dataset file to be able to process big data more efficiently. I designed the dataset file mainly to handle simple, not large, data, because I am not in this regard now.

There are several versions of the GPT. This implementation focuses mainly on the implementation of ["Improving Language Understanding by Generative Pre-Training"](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035).


## Project Structure

- `config.py`: Configuration file for model hyperparameters.
- `decoder.py`: GPT decoder.
- `positional_embeddings.py`: Positional embedding generation.
- `embeddings.py`: Token embeddings generation.
- `attention.py`: Self-attention mechanism.
- `feed_forward.py`: Feed-forward neural network.
- `lr_schedule.py`: Learning rate scheduling.
- `utils.py`: Utility functions for training and inference.
- `loss_functions.py`: Custom loss functions.
- `metrics.py`: Custom evaluation metrics.
- `streamer.py`: Data streamer for efficient training.
- `gpt_model.py`: Main GPT model implementation.
- `bpe_tokenizer.py`: Tokenizer for BPE (Byte Pair Encoding) tokenization.
- `tokenizer.py`: Pre-trained GPT tokenizer.
- `prepare_dataset.py`: A file through which we perform some operations on the dataset (creating a special folder for validation data).
- `inferance.py`: A file needed to generate sentences from the model based on the input prompt.
- `tmp/`: Directory for storing model checkpoints.
- `demo/`: Project documentation.
- `tokenizer/`: Directory for saving the retrained tokenizer.
- `dummy_data/`: Directory in which we put some data to test the streamer.

## Requirements

- Python 3.10.6
- TensorFlow 2.12.0
- Transformers 4.33.2 (Just for tokenizer)

## Documentation

Detailed project documentation can be found in the `demo/` directory. It includes explanations of the GPT architecture, training procedures, and how to use the model for various natural language processing tasks.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/AliHaiderAhmad001/GPT-from-Scratch-with-Tensorflow.git
   cd GPT-from-Scratch-with-Tensorflow
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install project dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Download and prepare Dataset: You can go and review the demo.You can work on the same dataset, change it or adjust your preferences. However, You can download the dataset directly from [here](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz). You can take a part of it for validation through the following code:
   ```bash
   python prepare_dataset.py aclImdb/test aclImdb/valid --num_files_to_move 2500
   ```
The data loader I use requires the validation set to be in a separate folder.
   
6. Optionally, you can re-train GPT tokinizer:
   ```bash
   python bpe_tokenizer.py aclImdb --batch_size 1000 --vocab_size 50357 --save --save_fp tokenizer/adapted-tokenizer
   ```
7. Train the GPT model (provide more specific instructions if needed):
   * To start training from scratch: `python train.py`
   * To resume training from a checkpoint: `python train.py --resume`
8. Generate Sentences. You can use the following command to generate text using your script:
   ```bash
   python inferance.py "input_text_prompt" --sampler "greedy"  # For greedy sampling
   ```
   Or:
   ```bash
   python inferance.py "input_text_prompt" --sampler "beam" --beam_width 5  # For beam search sampling with a beam width of 5
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/AliHaiderAhmad001/Neural-Machine-Translator/blob/main/LICENSE.txt) file for details.

## Acknowledgments

- [OpenAI GPT](https://openai.com/research/gpt).

---
