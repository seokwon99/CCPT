#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist
import transformers
# from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import Trainer, BitsAndBytesConfig
import numpy as np
import pandas as pd
from typing import List
from peft import LoraConfig, get_peft_model

from results import task1, task2
from results import prompts

# import wandb
# wandb.init(project="Conceptual Combination Fine-Tuning")

task_prompts = {
    'npc': prompts.baseline_instruction_npc['naive'],
    'pi_emergent': prompts.baseline_instruction_pi_emergent['naive'],
    'pi_canceled': prompts.baseline_instruction_pi_canceled['naive']
}
fewshot = {
    'npc': task1.add_instruction['emergent'],
    'pi_emergent': task2.add_instruction['emergent'],
    'pi_canceled': task2.add_instruction['canceled']
}
input_format = {
    'npc': prompts.input_format_npc,
    'pi_emergent': prompts.input_format_pi,
    'pi_canceled': prompts.input_format_pi
}

output_format = {
    'npc': "{{\"noun_phrase\": \"{noun_phrase}\", \"modifier\": \"{modifier}\"}}",
    'pi_emergent': "{{\"property\": \"{property}\"}}",
    'pi_canceled': "{{\"property\": \"{property}\"}",
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    lora: Optional[bool] = field(default=False)
    quantization: Optional[bool] = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
                           "help": "Path to the training data."})
    task_type: List[str] = field(default=None, metadata={
                           "help": "Path to the training data."})
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
    
def print_trainable(model):
    print("Learnable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.size()}")


IGNORE_INDEX = -100
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
    
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tasks: List[str], data_path: str, tokenizer: transformers.PreTrainedTokenizer, split="0.0:1.0"):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        data = pd.read_csv(data_path)
        split = split.split(":")
        data = data[int(len(data) * float(split[0])):int(len(data) * float(split[1]))]
        sources = []
        targets = []
        
        logging.warning("Formatting inputs...")
        for task in tasks:
            prompt = task_prompts.get(task, None)
            if prompt is None:
                print(f"Not valid task: {task}")
                continue
            for _, row in data.iterrows():
                sources.append(tokenizer.apply_chat_template([
                        {"role": "system", "content": prompts.task_instructions},
                        {"role": "user", "content": prompt.format(add_instruction = fewshot[task], input_format = input_format[task].format(noun_phrase=row['noun_phrase'], head=row['root'], property=row['property']))}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                ))
                targets.append(output_format[task].format(**row))
        logging.warning("Tokenizing inputs... This may take some time...")
        
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, eval=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tasks=data_args.task_type, tokenizer=tokenizer, data_path=data_args.data_path, split="0.0:0.8")
    eval_dataset = SupervisedDataset(
        tasks=data_args.task_type, tokenizer=tokenizer, data_path=data_args.data_path, split="0.8:1.0") if eval else None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, transformers.TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    bit_config = None
    if model_args.quantization:
        print("Apply 4-bit Quantization")
        bit_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_storage=torch.float16
        )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bit_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if model_args.lora else torch.float32
    )
    # model = prepare_model_for_kbit_training(model)
    
    if model_args.lora:
        print("Apply LoRA layer")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            # modules_to_save = ["lm_head", "embed_tokens"]
        )
        model = get_peft_model(model, lora_config)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.model_max_length,
        padding_side="right",
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        eval=training_args.do_eval
    )
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    
    if dist.get_rank() == 0:
        if getattr(trainer.model, "print_trainable_parameters", None):
            trainer.model.print_trainable_parameters()
        bytes_value = trainer.model.get_memory_footprint()
        gb_value = bytes_value / (1024 ** 3)
        print(f"{gb_value:.2f} GB.")
        # print_trainable(trainer.model)

    if model_args.lora and getattr(trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy
        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
    
    trainer.model.config.use_cache = not training_args.gradient_checkpointing
    
    dist.barrier()

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

if __name__ == "__main__":
    train()