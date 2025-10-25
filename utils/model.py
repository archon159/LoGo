from .model_list import *

from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import PeftMixedModel
from peft.tuners.lora.layer import LoraLayer
from peft.utils.save_and_load import set_peft_model_state_dict
from peft.helpers import rescale_adapter_scale
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

METRICS = ['norm', 'entropy']

EVAL_METHOD_DICT = {
    'bbh': 'em',
    'wmt14': 'bleu',
    'wmt16': 'bleu',
    'gem': 'rouge',
    'ai2_arc': 'em',
    'nq': 'em_multi',
    'tqa': 'em_multi',
    'anli': 'em',
    'glue': 'em',
    'code_x_glue': 'bleu',
}

TARGET_BLOCK = {
    'Qwen2.5_7b': {
        'layer_idx': 27,
        'proj': 'q',
    },
    'Llama-3.1-8B': {
        'layer_idx': 31,
        'proj': 'q',
    },
    'deepseek-llm-7b-base': {
        'layer_idx': 29,
        'proj': 'q',
    },
}

def entropy(x: torch.Tensor, dim: int = -1, eps: float = 1e-8):
    probs = torch.softmax(x, dim=dim)
    entropy = - (probs * torch.log(probs + eps)).sum(dim=dim)
    return entropy

def get_base_model(
    base_model_name="Llama-3.1-8B",
    model_dir=Path('./hf_models'),
):
    if (model_dir / base_model_name).exists():
        model_path = model_dir / base_model_name
    else:
        model_path = HF_MODEL_PATH_DICT[base_model_name]
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path
    )
    
    config = GenerationConfig(
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    return base_model, config

def get_tokenizer(
    base_model_name="Llama-3.1-8B",
    model_dir=Path('./hf_models'),
):
    if (model_dir / base_model_name).exists():
        model_path = model_dir / base_model_name
    else:
        model_path = HF_MODEL_PATH_DICT[base_model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
        
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return tokenizer

def get_mixed_lora_model(
    base_model,
    lora_module_name_list,
    model_dir=Path('./hf_models'),
):
    if (model_dir / lora_module_name_list[0]).exists():
        peft_model = PeftMixedModel.from_pretrained(
            base_model,
            model_dir / lora_module_name_list[0],
            adapter_name="__bootstrap__"
        )
    else:
        peft_model = PeftMixedModel.from_pretrained(
            base_model,
            HF_ADAPTER_REPO,
            subfolder=lora_module_name_list[0],
            adapter_name="__bootstrap__"
        )
        
    peft_model = peft_model.cuda()
    
    lora_name_dict = {}
    for i, lora_module_name in enumerate(tqdm(lora_module_name_list)):
        if (model_dir / lora_module_name).exists():
            peft_model.load_adapter(
                model_dir / lora_module_name,
                adapter_name=f'lora_{i}'
            )
        else:
            peft_model.load_adapter(
                HF_ADAPTER_REPO,
                subfolder=lora_module_name,
                adapter_name=f'lora_{i}'
            )
        
        lora_name_dict[f'lora_{i}'] = lora_module_name
    
    peft_model.set_adapter(list(lora_name_dict.keys()))
    peft_model.delete_adapter("__bootstrap__")
    
    return peft_model, lora_name_dict


    
def get_lora_matrices_for(
    model,
    base_model_name,
):
    target_block_detail = TARGET_BLOCK[base_model_name]        
    layer_idx = target_block_detail['layer_idx']
    proj = target_block_detail['proj']

    target_name = f"base_model.model.model.layers.{layer_idx}.self_attn.{proj}_proj"
        
    module_dict = dict(model.named_modules())
    module = module_dict[target_name]
    if not isinstance(module, LoraLayer):
        raise TypeError(f"Not LoRA Adapter: {target_name} -> {type(module)}")

    lora_matrix_dict = {}
    for adapter in model.active_adapters:
        A = module.lora_A[adapter].weight.data
        B = module.lora_B[adapter].weight.data
        scaling = module.scaling[adapter]
    
        lora_matrix_dict[adapter] = (A, B, scaling)

    return target_name, lora_matrix_dict

@contextmanager
def rescale_adapter_scale_multi(model, multiplier_dict):
    original_scaling = {}
    for module in model.modules():
        if isinstance(module, LoraLayer):
            original_scaling[module] = module.scaling.copy()
            module.scaling = {k: v * multiplier_dict[k] if k in multiplier_dict else v for k, v in module.scaling.items()}

    if not original_scaling:
        raise ValueError("scaling is only supported for models with `LoraLayer`s")
    try:
        yield

    finally:
        # restore original scaling values after exiting the context
        for module, scaling in original_scaling.items():
            module.scaling = scaling

            
def eval_single(
    output,
    label,
    dataset_name,
):
    dataset_words = dataset_name.split('.')
    eval_method = EVAL_METHOD_DICT[dataset_words[0]]
    
    if eval_method == 'em':
        ret = output.strip().lower().replace(".", "") == label.strip().lower().replace(".", "")
        ret = int(ret) # Making boolean to integer
        
    elif eval_method == 'em_multi':
        output_norm = output.strip().lower().replace(".", "")
        matches = [output_norm == l.strip().lower().replace(".", "") for l in label]
        ret = int(any(matches))
        
    elif eval_method == 'bleu':
        candidate = output.split()
        reference = label.split()

        ret = sentence_bleu(
            [reference], 
            candidate,
            weights=(0.25, 0.25, 0.25, 0.25),  # BLEU-4
            smoothing_function=SmoothingFunction().method1
        )

    elif eval_method == 'rouge':
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(label, output)
        # Extract f-measure values
        ret = {
            "rouge-1": scores["rouge1"].fmeasure,
            "rouge-2": scores["rouge2"].fmeasure,
            "rouge-L": scores["rougeL"].fmeasure,
        }
        
    else:
        assert(0)
    
    return ret

class StopOnNewline(StoppingCriteria):
    def __init__(self, newline_token_id):
        self.newline_token_id = newline_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.newline_token_id


def select_and_predict(
    peft_model,
    tokenizer,
    dataloader,
    target_name,
    lora_matrix_dict,
    dataset_name,
    n_selected_modules=20,
    selection_strategy='norm',
    lora_name_dict=None,
    base_model_name='Llama-3.1-8B',
    model_dir=Path('./hf_models'),
    generation_early_stop=False,
):    
    _, _, _, _, layer_num, _, matrix_type = target_name.split('.')
    matrix_type = matrix_type.split('_')[0]
    layer_num = int(layer_num)
    
    peft_model.eval()
    
    metric_dict = defaultdict(list)
    total_eval_results = []
    total_outputs = []
    total_selected_loras = []
    total_weights = []
    
    if generation_early_stop:
        newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        stopping_criteria = StoppingCriteriaList([
            StopOnNewline(newline_id)
        ])
    else:
        stopping_criteria = None
    
    
    for index, (inputs, labels) in enumerate(tqdm(dataloader)):
        inputs = tokenizer(
            inputs,
            max_length=2048,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()

        with torch.no_grad():
            outputs = peft_model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            input_tensor = outputs.hidden_states[layer_num]

            lora_projections = []
            for lora_name, (lora_A, lora_B, scaling) in lora_matrix_dict.items():
                lora_projection = (input_tensor @ lora_A.T) @ lora_B.T * scaling
                lora_projections.append(lora_projection)
                
            lora_projections = torch.stack(lora_projections, dim=1) # (N, K, L, D)
            batch_size, n_loras, n_tokens, n_dim = lora_projections.shape
            assert(batch_size == 1)
            
            mask = attention_mask.unsqueeze(dim=1) # (N, 1, L)

            seq_lengths = mask.sum(dim=-1) # (N, 1)
            last_idx = (seq_lengths - 1).view(batch_size, 1, 1, 1).expand(-1, n_loras, 1, n_dim)
            
            last_token_proj = lora_projections.gather(dim=2, index=last_idx).squeeze(2) # (N, K, D)
                    
            if selection_strategy == 'norm':
                metric_result = torch.norm(last_token_proj, dim=-1) # (N, K)
            elif selection_strategy == 'entropy':
                metric_result = 1 / (entropy(last_token_proj, dim=-1) + 1e-6) # (N, K)
            else:
                assert(0)

            metric_dict[selection_strategy].append(metric_result)
            top_metric, top_indices = torch.topk(metric_result, k=n_selected_modules, dim=-1) # (N, S), (N, S)
            
            batch_outputs = []
            batch_selected_loras = []
            batch_weights = []
            for input_idx in range(input_ids.shape[0]):
                selected_loras = [f'lora_{j}' for j in top_indices[input_idx]]
                batch_selected_loras.append(selected_loras)
                
                peft_model.set_adapter(selected_loras)
                metric_weight = top_metric[input_idx] / top_metric[input_idx].sum()
                
                multiplier_dict = {
                    k: v.item()
                        for (k, v) in zip(selected_loras, metric_weight)
                }

                with rescale_adapter_scale_multi(peft_model, multiplier_dict):
                    output = peft_model.generate(
                        input_ids=input_ids[input_idx:input_idx+1],
                        max_new_tokens=256,
                        pad_token_id=tokenizer.pad_token_id,
                        stopping_criteria=stopping_criteria,
                    )
                        
                input_length = attention_mask[input_idx].sum().item()
                cur = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
                cur = cur.strip()
                cur = (cur.splitlines() or [cur])[0]

                decoded_output = [cur]
                
                batch_outputs += decoded_output
                batch_weights.append(metric_weight.cpu().tolist())
            
        batch_eval_results = []
        for output, label in zip(batch_outputs, labels):
            ret = eval_single(output, label, dataset_name)
            batch_eval_results.append(ret)
                
        total_outputs += batch_outputs
        total_eval_results += batch_eval_results
        total_selected_loras += batch_selected_loras
        total_weights += batch_weights
        
    for metric in metric_dict:
        metric_dict[metric] = torch.cat(metric_dict[metric], dim=0).cpu().tolist()
    
    return total_eval_results, total_outputs, total_selected_loras, total_weights, metric_dict