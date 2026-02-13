# -*- coding: utf-8 -*-
import argparse

def transformer_layer_param(model_name, d_model, d_ff):
    """
    Calculate the number of parameters in Transformer Encoder/Decoder in a single layer. 
    :param model_name: bert, gpt, llama, opt
    :param d_model: model dimension (hidden_size)
    :param d_ff: internal dimensionality of a feed-forward neural network (intermediate_size)
    """
    if model_name.lower().startswith("llama"):
        attention = 4 * (d_model ** 2)
        feed_forward = 3 * d_model * d_ff
        layer_norm = d_model
    else:    
        attention = 4 * (d_model ** 2 + d_model)
        feed_forward = 2 * d_model * d_ff + d_model + d_ff
        layer_norm = 2 * d_model
    
    return attention + feed_forward + 2 * layer_norm


def total_param_estimate(model_name, hidden_size, hidden_layers, num_attention_heads, vocab_size=49152, seq_length=2048):
    """
    Estimate the number of parameters based on model type and configurations
    """
    if hidden_size % num_attention_heads != 0:
        print("Warning: hidden size should be divisible by number of attention heads")
        return None
    
    if model_name.lower().startswith("llama"):
        if (hidden_size / num_attention_heads) % 2 != 0:
            print("Warning: hidden size / num_attention_heads should be divisible by 2 in Llama model")
            print(f"An alternative hidden_size is {hidden_size + num_attention_heads}")
            return None
    
    # Calculate the number of parameters in transfomer layers
    layer_param = transformer_layer_param(model_name, hidden_size, hidden_size * 4)
        
    print("layer_param: ", layer_param)
    total_layer_param = layer_param * hidden_layers
    
    # Calculate the number of other parameters
    if model_name.lower().startswith("llama"):
        other_param = 2 * hidden_size * vocab_size + hidden_size
    elif model_name.lower().startswith("opt"):
        word_embed_proj_dim = 768
        other_param = word_embed_proj_dim * vocab_size + 2 * word_embed_proj_dim * hidden_size + (seq_length + 2) * hidden_size + hidden_size * 2
    elif model_name.lower().startswith("bert"):
        other_param = hidden_size * vocab_size + hidden_size**2 + seq_length * hidden_size + 7 * hidden_size + vocab_size
    elif model_name.lower().startswith("gpt"):
        other_param = hidden_size * vocab_size + seq_length * hidden_size
    else:
        print("Warning: unsupported model type")
        return None
    
    return total_layer_param + other_param

# Usage: python param_estimate.py model_name hidden_size hidden_layers num_attention_heads [--vocab_size vocab_size] [--seq_length seq_length]
# Example: python param_estimate.py gpt 4096 24 32
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="model name")
    parser.add_argument("hidden_size", type=int, help="hidden size")
    parser.add_argument("hidden_layers", type=int, help="hidden layers")
    parser.add_argument("num_attention_heads", type=int, help="number of attention heads")
    parser.add_argument("--vocab_size", type=int, default=49152, help="vocab size")
    parser.add_argument("--seq_length", type=int, default=512, help="sequence length")
    
    args = parser.parse_args()
    
    print("model: ", args.model_name)
    print("hidden_size: ", args.hidden_size)
    print("hidden_layers: ", args.hidden_layers)
    print("num_attention_heads: ", args.num_attention_heads)
    print("------------------------------")
    print("Estimated number of parameters: ", total_param_estimate(args.model_name, args.hidden_size, args.hidden_layers, args.num_attention_heads, args.vocab_size, args.seq_length))
    