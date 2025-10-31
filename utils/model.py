import sys 
import torch
import torch.nn as nn
from dotenv import dotenv_values
from segformer_pytorch import Segformer
from functools import partial

# N                     : heads of each stage, heads
# E                     : feedforward expansion factor of each stage
# R                     : reduction ratio of each stage for efficient attention
# L / Depths            : num_layers, num layers of each stage
# C / Hidden sizes      : MiT Encoder dimensions of each stage
# Decoder hidden size   : MLP Decoder channel dimension
# number of  classes    : Foreground & Background

segformer_b0_config = {
    "dims" : (32, 64, 160, 256),      
    "heads" : (1, 2, 5, 8),
    "ff_expansion" : (8, 8, 4, 4),
    "reduction_ratio" : (8, 4, 2, 1),
    "num_layers" : (2, 2, 2, 2),
    "decoder_dim" : 256,
    "num_classes" : 1
}
    
segformer_b1_config = {
    "dims" : (64, 128, 320, 512),      
    "heads" : (1, 2, 5, 8),
    "ff_expansion" : (8, 8, 4, 4),
    "reduction_ratio" : (8, 4, 2, 1),
    "num_layers" : (2, 2, 2, 2),
    "decoder_dim" : 256,
    "num_classes" : 1
}

segformer_b2_config = {
    "dims" : (64, 128, 320, 512),      
    "heads" : (1, 2, 5, 8),
    "ff_expansion" : (8, 8, 4, 4),
    "reduction_ratio" : (8, 4, 2, 1),
    "num_layers" : (3, 3, 6, 3),
    "decoder_dim" : 768,
    "num_classes" : 1
}

segformer_b3_config = {
    "dims" : (64, 128, 320, 512),       
    "heads" : (1, 2, 5, 8),
    "ff_expansion" : (8, 8, 4, 4),
    "reduction_ratio" : (8, 4, 2, 1),
    "num_layers" : (3, 3, 18, 3),
    "decoder_dim" : 768,
    "num_classes" : 1
}

segformer_b4_config = {
    "dims" : (64, 128, 320, 512),       
    "heads" : (1, 2, 5, 8),
    "ff_expansion" : (8, 8, 4, 4),
    "reduction_ratio" : (8, 4, 2, 1),
    "num_layers" : (3, 3, 27, 3),
    "decoder_dim" : 768,
    "num_classes" : 1
}

segformer_b5_config = {
    "dims" : (64, 128, 320, 512),       
    "heads" : (1, 2, 5, 8),
    "ff_expansion" : (8, 8, 4, 4),
    "reduction_ratio" : (8, 4, 2, 1),
    "num_layers" : (3, 3, 40, 3),
    "decoder_dim" : 768,
    "num_classes" : 1
}

unet_config = {
    'in_channels': 3,
    'out_channels': 1,
    'init_features': 32,
    'pretrained': False,
    'force_reload': True
}

def select_model(model_name, model_config):
    """
    Unet & Polyp-PVT expects image and masks to be same. 
    Segformer expects mask_size=image_size/4. 
    Make sure to change this in parameters.
    """
    env_vars = dotenv_values(dotenv_path="./.env")
    model = None  
    if model_name == "unet":
        config = unet_config
        config["init_features"] = int(model_config)
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch',
            'unet',
            **config
            )
    else:
        segformer_configs = {
            "b0": segformer_b0_config,
            "b1": segformer_b1_config,
            "b2": segformer_b2_config,
            "b3": segformer_b3_config,
            "b4": segformer_b4_config,
            "b5": segformer_b5_config
            } 
        if model_config not in segformer_configs:
            raise ValueError(f"Invalid config: {model_config}. Choose from 'b1' to 'b5'.")
        
        config = segformer_configs[model_config]
        model = Segformer(**config)

    return model

    
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
