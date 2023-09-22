import argparse
from config import Config
from utols import load_model_and_optimizer
from sampler import GreedySampler, BeamSearchSampler

config = Config()

def parse_generation_args():
    parser = argparse.ArgumentParser(description="Text Generation Script")
    parser.add_argument("input_prompt", type=str, help="The input text prompt for text generation")
    parser.add_argument("--sampler", type=str, choices=["greedy", "beam"], default="beam", help="Sampling strategy ('greedy' or 'beam')")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search sampling (if applicable)")
    return parser.parse_args()

def generate_response(input_prompt, sampler_type, beam_width):
    """
    Generate text using a trained GPT model.

    Args:
        input_prompt (str): The input text prompt for text generation.
        sampler_type (str): Sampling strategy ('greedy' or 'beam'). Default is beam.
        beam_width (int): Beam width for beam search sampling. Default is 5.
    """
    model = load_model_and_optimizer(config.model_weights_checkpoint_directory)

    if sampler_type == 'beam':
        sampler = BeamSearchSampler(model, config.tokenizer_path, config.sequence_length, beam_width=beam_width, end_token=config.end_token)
    else:
        sampler = GreedySampler(model, config.tokenizer_path, config.sequence_length, end_token=config.end_token)

    generated_text = sampler.decode(input_prompt)

    # Print the generated text
    print("Generated Text:", generated_text)

    return generated_text

if __name__ == "__main__":
    args = parse_generation_args()
    input_prompt = args.input_prompt
    sampler_type = args.sampler
    beam_width = args.beam_width

    generate_response(input_prompt, sampler_type, beam_width)

