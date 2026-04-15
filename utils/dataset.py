from .dataset_list import *

import torch
import pandas as pd
from datasets import Dataset
from pathlib import Path
import json

def get_dataset(dataset, data_dir=Path('./data'), base_model_name='Llama-3.1-8B'):
    dataset_words = dataset.split('.')
    main = dataset_words[0]
    sub = '.'.join(dataset_words[1:])

    input_keyword = 'context'
    output_keyword = 'completion'

    train_data_path = data_dir / main / sub / 'train.jsonl'
    test_data_path = data_dir / main / sub / 'test.jsonl'
        
    train_samples = []

    if train_data_path.exists():
        with open(train_data_path, 'r') as f:
            for line in f:
                train_samples.append(json.loads(line))

    test_samples = []
    if test_data_path.exists():
        with open(test_data_path, 'r') as f:
            for line in f:
                test_samples.append(json.loads(line))
            
    def generate_dataset_obj(samples):
        df = []
        for sample in samples:
            cur_input = sample[input_keyword]
            cur_output = sample[output_keyword]

            cur = {
                'input': cur_input,
                'output': cur_output
            }
            df.append(cur)

        dataset_obj = Dataset.from_pandas(pd.DataFrame(df))
        return dataset_obj

            
    train_dataset_obj = generate_dataset_obj(train_samples)
    test_dataset_obj = generate_dataset_obj(test_samples)
    
    return train_dataset_obj, test_dataset_obj


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_obj):
        self.input_list = dataset_obj["input"]
        self.label_list = dataset_obj["output"]
    
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        cur_label = self.label_list[idx]
        cur_input = self.input_list[idx]
        
        return cur_input, cur_label
    

def create_dataloader(
    dataset_obj,
    batch_size=1
):
    simple_dataset = SimpleDataset(dataset_obj)
    simple_dataloader = torch.utils.data.DataLoader(
        simple_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    
    return simple_dataloader