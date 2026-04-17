import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set HF_HOME to a writable directory with sufficient disk space if not already set.
# Example: export HF_HOME=/path/to/large/storage/.cache/huggingface
if "HF_HOME" not in os.environ:
    print("[Warning] HF_HOME is not set. Models will be cached in the default ~/.cache/huggingface directory.")
    print("[Warning] If you face disk quota issues, set HF_HOME to a larger storage path before running.")

# To access gated models (e.g. Llama), log in via: huggingface-cli login
# Or set the HF_TOKEN environment variable: export HF_TOKEN=<your_token>

import time
from pathlib import Path
import numpy as np
import json

import torch

import utils.common as cm
import utils.dataset as ds
import utils.model as md

if __name__ == "__main__":
    args = cm.parse_arguments()
    cm.reset_seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    
    result_dir = Path(args.result_dir)
    model_dir = Path(args.model_dir)
    
    data_dir = Path(args.data_dir)
    if args.dataset == '':
        dataset_list = ds.DATASETS
    else:
        dataset_list = [args.dataset]
        
    print(f'{len(dataset_list)} Datasets')
        
    loading_start = time.time()
    print('Loading Base Model')
    base_model, config = md.get_base_model(
        base_model_name=args.base_model,
        model_dir=model_dir
    )
    tokenizer = md.get_tokenizer(
        base_model_name=args.base_model,
        model_dir=model_dir
    )
    
    all_loras = md.LORA_MODULE_NAMES[args.base_model]
    all_loras.sort()
    
    print(f'Loading {len(all_loras)} LoRA Adapters')
    peft_model, lora_name_dict = md.get_mixed_lora_model(
        base_model,
        all_loras,
        model_dir=model_dir
    )
    loading_end = time.time()
    
    print(f'Extracting LoRA Matrices')
    target_name, lora_matrix_dict = md.get_lora_matrices_for(peft_model, args.base_model)
    
    loading_time = loading_end - loading_start
        
    for dataset in dataset_list:
        inference_start = time.time()
        
        _, test_dataset_obj = ds.get_dataset(
            dataset, data_dir, base_model_name=args.base_model
        )
        
        test_dataloader = ds.create_dataloader(
            test_dataset_obj,
            batch_size=1
        )
        
        print(f'Dataset {dataset}: {len(test_dataset_obj)} sample')

        eval_results, predictions, selected_loras, weights, metric_dict = md.select_and_predict(
            peft_model,
            tokenizer,
            test_dataloader,
            target_name,
            lora_matrix_dict,
            dataset,
            n_selected_modules=args.n_selected_modules,
            selection_strategy=args.selection_strategy,
            lora_name_dict=lora_name_dict,
            base_model_name=args.base_model,
            model_dir=model_dir,
            generation_early_stop=args.generation_early_stop,
        )
        
        selected_lora_names = []
        for row in selected_loras:
            new_row = [lora_name_dict[r] for r in row]
            selected_lora_names.append(new_row)
        
        inference_end = time.time()
        
        inference_time = inference_end - inference_start
        time_dict = {
            'loading': loading_time,
            'inference': inference_time,
        }
        
        if not dataset.startswith('gem'):
            acc = np.array(eval_results).mean()
            print(f'Accuracy: {acc * 100:.2f}')
            
        target_str = f'{args.base_model}_selection_{args.selection_strategy}_n_{args.n_selected_modules}_seed_{args.seed}'
        target_dir = result_dir / target_str / dataset
        target_dir.mkdir(exist_ok=True, parents=True)
        
        print(target_dir)
        
        with open(target_dir / 'eval_results.json', 'w') as f:
            json.dump(eval_results, f)

        with open(target_dir / 'predictions.json', 'w') as f:
            json.dump(predictions, f)

        with open(target_dir / 'selected_loras.json', 'w') as f:
            json.dump(selected_lora_names, f)

        with open(target_dir / 'weights.json', 'w') as f:
            json.dump(weights, f)

        with open(target_dir / 'metric_dict.json', 'w') as f:
            json.dump(metric_dict, f)

        with open(target_dir / 'time_dict.json', 'w') as f:
            json.dump(time_dict, f)