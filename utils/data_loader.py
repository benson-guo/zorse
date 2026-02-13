# -*- coding: utf-8 -*-
import numpy as np
import torch
from models.hub import is_vision_model


def generate_model_data(
    model,
    batch_size=1,
    vocab_size=49152,
    seq_length=512,
    num_channels=3,
    image_size=224,
):
    image_data = is_vision_model(model._model_name)
    if image_data:
        size = (batch_size, num_channels, image_size, image_size)
        return torch.randn(size, device="cuda")
    else:
        size = (batch_size, seq_length)
        return torch.randint(low=0, high=vocab_size, size=size, device="cuda")


def get_dataset(args):
    if is_vision_model(args.model_name):
        return get_image_dataset(args)
    else:
        return get_sequence_dataset(args)


def get_sequence_dataset(args):
    num_batches = args.warmup_iterations + args.iterations
    size = (num_batches * args.local_batch_size, args.seq_length)
    input_ids = torch.randint(low=0, high=args.vocab_size, size=size, device="cuda")
    targets = torch.randint(low=0, high=args.vocab_size, size=size, device="cuda")
    dataset = torch.utils.data.TensorDataset(input_ids, targets)
    return torch.utils.data.DataLoader(dataset, batch_size=args.local_batch_size)


# Dataset for ViT model
def get_image_dataset(args):
    num_channels = 3  # RGB
    image_size = args.image_size
    num_batches = args.warmup_iterations + args.iterations
    size = (num_batches * args.local_batch_size, num_channels, image_size, image_size)
    images = torch.randn(size, device="cuda")
    targets = torch.randint(
        low=0, high=2, size=(num_batches * args.local_batch_size,), device="cuda"
    )
    dataset = torch.utils.data.TensorDataset(images, targets)
    return torch.utils.data.DataLoader(dataset, batch_size=args.local_batch_size)


def split_microbatches(inputs, labels, microbatches):
    batch_size = len(inputs)
    min_microbatch_size = batch_size // microbatches
    assert min_microbatch_size > 0, "Microbatch size must be greater than 0"
    split_sizes = [
        min_microbatch_size + 1
        if i < batch_size % microbatches
        else min_microbatch_size
        for i in range(microbatches)
    ]
    return inputs.split(split_sizes), labels.split(split_sizes)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Datasets for deepspeed


class DatasetForLLM(torch.utils.data.Dataset):
    """
    Test dataset for deepspeedgpt ()
    """

    def __init__(self, vocab_size, seq_length, n_embd, size=1000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.n_embd = n_embd
        self.size = size

        self.inputs = np.random.randint(low=0, high=vocab_size, size=(size, seq_length))
        self.labels = np.random.rand(size, seq_length)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.inputs[idx], dtype=torch.long)
        label_seq = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_seq, label_seq


class DatasetForViTRegression(torch.utils.data.Dataset):
    """
    Test dataset for ViT for regression tasks.
    """

    def __init__(self, image_size, channels, size=1000, dtype=torch.float32):
        self.image_size = image_size
        self.channels = channels
        self.size = size
        self.dtype = dtype

        self.images = np.random.rand(size, channels, image_size, image_size)
        self.labels = np.random.uniform(0, 1, size=(size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=self.dtype)
        label = torch.tensor(self.labels[idx], dtype=self.dtype)
        return image, label


class DatasetForLLMLanguage(torch.utils.data.Dataset):
    """
    Test dataset for language modeling
    """

    def __init__(self, vocab_size, seq_length, size=1000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.size = size

        self.inputs = np.random.randint(low=0, high=vocab_size, size=(size, seq_length))
        # shift by one and add new token
        self.labels = np.hstack(
            (
                self.inputs[:, 1:],
                np.random.randint(low=0, high=vocab_size, size=(size, 1)),
            )
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.inputs[idx], dtype=torch.long)
        label_seq = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_seq, label_seq


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Datasets for deepspeed
