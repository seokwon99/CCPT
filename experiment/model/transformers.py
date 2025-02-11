import json
import re
import torch 
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator


class Transformers():
    def __init__(self, model_id=""):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate(self, text, batch_size=8, max_length=2048, temperature=0.7, num_return_sequences=1, system_prompt = None, **kwargs):
        model_inputs = self.tokenizer.apply_chat_template([
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": t}
                ] for t in text
            ],
            tokenize=False,
            add_generation_prompt=True
        ) if system_prompt else self.tokenizer.apply_chat_template([
                [
                    {"role": "user", "content": t}
                ] for t in text
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_outputs = []
        for i in tqdm(range(0, len(model_inputs), batch_size)):
            inputs = model_inputs[i:i+batch_size]
            inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs.to(self.model.device),
                    num_return_sequences=num_return_sequences,
                    max_length=max_length,
                    temperature=temperature + 0.01,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    **kwargs
                )
            output = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            output = [output[j:j+num_return_sequences] for j in range(0, len(output), num_return_sequences)]
            model_outputs.extend(output)
            del inputs
            del outputs
            torch.cuda.empty_cache()
        return {"responses": model_outputs}