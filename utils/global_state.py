# -*- coding: utf-8 -*-

from models.hub import get_all_layers
import torch

# Global state used for torch run
SPLIT_STATE = {
    "model_partitions": [0.25, 0.75],
    "split_idx_map": {},
    "split_uneven": False,
    "proportional_split": False,
    "microbatches": 1,
    "forward_prefetch": {},
    "backward_prefetch": {},
    "streams": None,
    "optimizer_in_backwards": False,
    "gradient_accumulation": False,
    "unshard_in_compute": False,
    "unshard_events": {},
    # for flashflex pipeline ga
    "skip_reduce_scatter": False,
    "zero2_pipeline": False,
}


def init_split_state():
    """Initialize the SPLIT_STATE with default values."""
    global SPLIT_STATE
    SPLIT_STATE = {
        "model_partitions": [0.25, 0.75],
        "split_idx_map": {},
        "split_uneven": False,
        "proportional_split": False,
        "backward_count_map": {},
        "microbatches": 1,
        "forward_prefetch": {},
        "backward_prefetch": {},
        "streams": None,
        "optimizer_in_backwards": False,
        "gradient_accumulation": False,
        "unshard_in_compute": False,
        "unshard_events": {},
        "skip_reduce_scatter": False,
        "zero2_pipeline": False,
    }


def get_split_state():
    """Retrieve the current SPLIT_STATE."""
    return SPLIT_STATE


def get_compute_stream():
    return SPLIT_STATE["streams"][0]


def set_split_state(key, value):
    """Set a specific value in the SPLIT_STATE."""
    global SPLIT_STATE
    SPLIT_STATE[key] = value


def set_split_state_nested(key, subkey, value):
    """Set a specific in nested dict in SPLIT_STATE"""
    global SPLIT_STATE
    SPLIT_STATE[key][subkey] = value


def update_split_state(new_state):
    """Update the SPLIT_STATE with a new state dictionary."""
    global SPLIT_STATE
    SPLIT_STATE.update(new_state)


def configure_gradient_accumulation(model, microbatches):
    model_layers = get_all_layers(model)
    # initialize ga specific split_state
    num_layers = len(model_layers)
    set_split_state(key="microbatches", value=microbatches)
    # setup split state for FSDP logic
    for i in range(num_layers - 1):
        cur_state = model_layers[i]
        next_state = model_layers[i + 1]
        set_split_state_nested(
            "forward_prefetch", cur_state._handle, next_state._handle
        )
    for i in range(1, num_layers):
        cur_state = model_layers[i]
        next_state = model_layers[i - 1]
        set_split_state_nested(
            "backward_prefetch", cur_state._handle, next_state._handle
        )
    for i in range(num_layers):
        cur_state = model_layers[i]
        set_split_state_nested(
            key="unshard_events", subkey=cur_state._handle, value=torch.cuda.Event()
        )
