import torch


def generate_data(num_samples, sequence_length, vocab_size):
    inputs = torch.randint(0, vocab_size, (num_samples, sequence_length))
    targets = torch.roll(inputs, -1, dims=1)
    return inputs, targets
