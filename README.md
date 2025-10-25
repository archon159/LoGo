# LoGo
Official repository of "LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging"

## Environment Setup

Python==3.12.10

CUDA==11.8

numpy==1.26.4

pandas==2.3.1

nltk==3.9.1

rouge_score==0.1.2

datasets==4.0.0

torch== 2.7.1

transformers==4.54.0

peft==0.16.0

## Quick Start
The base pretrained model and LoRA adpaters will be automatically downloaded.
```
python3 main.py --dataset <DATASET> --base_model <BASE_MODEL> --gpu <GPU>
```
