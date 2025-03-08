# Conceptual Combination with Property Type (CCPT)
[**📖 CCPT arXiv**](https://arxiv.org/abs/2502.06086/) | [**🤗 CCPT**](https://huggingface.co/datasets/seokwon99/CCPT_12.3K/)

This is the official repository of our NAACL 2025 main (Oral) paper: <a href="https://arxiv.org/abs/2502.06086"><b>Is Peeled Apple Still Red? Evaluating LLM for Conceptual Combination with Property Type</b></a>

## Installation

First, clone our GitHub repository.

```bash
git clone https://github.com/seokwon99/CCPT.git
```

Then navigate to the newly-created folder.
```bash
cd CCPT
```

Next, create a new Python 3.9+ environment using `conda`.

```bash
conda create --name ccpt python=3.9
```

Activate the newly-created environment.

```bash
conda activate ccpt
```

All external package requirements are listed in `requirements.txt`.
To install all packages, and run the following command.

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python -m spacy download en_core_web_sm
```

## Reimplementation

### Set Environment
```bash
export OPENAI_API_KEY="sk-..."    # if you want openai models
export ANTHROPIC_API_KEY="sk-..." # if you want anthropic models
```

### Get Model Response
```bash
python -m experiment.run.gen_property --property_type emergent    # property induction (emergent)
python -m experiment.run.gen_property --property_type canceled    # property induction (canceled)
python -m experiment.run.gen_combination --property_type emergent # noun phrase completion (emergent)
python -m experiment.run.cls_property_type                        # property type prediction
```

### Automatic Evaluation
```bash
python -m experiment.eval.eval_property --property_type emergent    # property induction (emergent)
python -m experiment.eval.eval_property --property_type canceled    # property induction (canceled)
python -m experiment.eval.eval_combination --property_type emergent # noun phrase completion (emergent)
python -m experiment.eval.eval_type                                 # property type prediction
```

## Fine-tuning
### Additional Installation
```bash
pip install torch==2.3.0 
pip install transformers trl peft bitsandbytes wandb
```

### Step1. Export Environment
```bash
# Define environment
export ACCELERATE_USE_FSDP=1
export TOKENIZERS_PARALLELISM=false

GPU_NUM=4
MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
TASK="task_name" # ["npc", "pi"]
DATA="YOUR_DATA"
```

### Step2. Training
```bash
# Run training
torchrun --nproc_per_node $GPU_NUM --nnodes 1 ./train.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --task_type $TASK \
    --output_dir model_params/qlora/$MODEL/$TASK \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 512 \
    --fp16 True \
    --gradient_checkpointing True \
    --use_reentrant False
```

## Have any questions?

Please contact at seokwon.song@vision.snu.ac.kr


## Citation
If you use CCPT in your research, please cite our work:

```bibtex  
@article{song2025peeled,
  title={Is a Peeled Apple Still Red? Evaluating LLMs' Ability for Conceptual Combination with Property Type},
  author={Song, Seokwon and Lee, Taehyun and Ahn, Jaewoo and Sung, Jae Hyuk and Kim, Gunhee},
  journal={arXiv preprint arXiv:2502.06086},
  year={2025}
}
'''
