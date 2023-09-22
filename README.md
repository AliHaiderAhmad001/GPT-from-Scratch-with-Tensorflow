# GenesisMind-Building-GPT-from-Scratch

GPT "Generative Pre-trained Transformer" is the first version of the GPT series of models, revolutionized natural language processing with its autoregressive language modeling capabilities built on the Transformer architecture.

---

![Python](https://img.shields.io/badge/python-3.10.6-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.12.0-orange.svg)
![Transformers](https://img.shields.io/badge/transformers-required-green.svg)

## Overview

This project is an implementation of the GPT (Generative Pre-trained Transformer) model from scratch using TensorFlow. It includes all the components of the model, such as the positional embeddings, attention mechanism, feed-forward layers, and more. The goal of this project is to provide a deep understanding of the GPT architecture and its inner workings. 

There are several versions of the GPT. This implementation focuses mainly on the implementation of ["Improving Language Understanding by Generative Pre-Training"](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035). Generally all versions are somewhat similar, the main difference in the capacity of the model and the size of the dataset.

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
- `tmp/`: Directory for storing model checkpoints.
- `demo/`: Project documentation.
- `tokenizer/`: Directory for saving the retrained tokenizer.

## Requirements

- Python 3.10.6
- TensorFlow 2.12.0
- Transformers 4.33.2 (Just for tokenizer)

## Documentation

Detailed project documentation can be found in the `demo/` directory. It includes explanations of the GPT architecture, training procedures, and how to use the model for various natural language processing tasks.

## Usage

1. Clone the repository:

   ```bash
   git clone [https://github.com/your-username/your-gpt-project.git](https://github.com/AliHaiderAhmad001/GenesisMind-Building-GPT1-from-Scratch.git)
   cd GenesisMind-Building-GPT1-from-Scratch
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

- [OpenAI GPT](https://openai.com/research/gpt) for inspiring this project.
- Any other acknowledgments or credits.

---

Feel free to add more sections or details to this README as needed for your project. You should include instructions on how to use your GPT model for text generation and any other relevant information for users and contributors.
