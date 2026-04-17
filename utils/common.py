import argparse
import random
import numpy as np
import os
import torch

def reset_seed(
    seed: int=0
):
    """
    Reset the random variables with the given seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def parse_arguments(
    return_default: bool=False,
) -> object:
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', type=int, default=0,
         help='Random seed'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU to use'
    )
    parser.add_argument(
        '--dataset', type=str, default='',
        help='The name of target dataset. Empty string means all dataset.'
    )
    parser.add_argument(
        '--base_model', type=str, default='Llama-3.1-8B',
        choices=['Llama-3.1-8B', 'Qwen2.5-7B', 'deepseek-llm-7b-base'],
        help='The name of base model'
    )
    parser.add_argument(
        '--selection_strategy', type=str, default='norm',
        help='The strategy of lora selection.'
    )
    parser.add_argument(
        '--n_selected_modules', type=int, default=20,
        help='The number of selected modules'
    )
    parser.add_argument(
        '--generation_early_stop', action='store_true',
        help='Stop generation based on stopping criteria'
    )
    parser.add_argument(
        '--data_dir', type=str, default=f'./data',
        help='The directory name to get data'
    )
    parser.add_argument(
        '--model_dir', type=str, default=f'./hf_models',
        help='The directory name to get models'
    )
    parser.add_argument(
        '--result_dir', type=str, default='results',
        help='The directory name to save results'
    )
    parser.add_argument(
        '--save_dir', type=str, default='save_dir',
        help='The directory to save interim files'
    )

    if return_default:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    return args
