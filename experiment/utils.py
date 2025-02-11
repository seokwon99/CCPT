import ast
import numpy as np
import pandas as pd
import spacy
import re
import argparse

from data.utils import available_concept_pos
from experiment.prompts import eval_prompt

EME_DATA_PATH = "ccpt_emergent.csv"
CAN_DATA_PATH = "ccpt_canceled.csv"
COM_DATA_PATH = "ccpt_component.csv"

def get_backend(lib, args):
    if lib == "openai":
        from experiment.model.openai import ParallelGPT as MODEL
    elif lib == "openai_o1":
        from experiment.model.openai_o1 import ParallelGPT as MODEL
    elif lib == "transformers":
        from experiment.model.transformers import Transformers as MODEL
    elif lib == 'anthropic':
        from experiment.model.anthropic import ParallelClaude as MODEL
    elif lib == 'ollama':
        from experiment.model.ollama import ParallelOllama as MODEL
        import litellm
        API_BASE = args.local_host
        litellm.api_base = API_BASE
    else:
        raise Exception("Not implemented")
    return MODEL

def get_relevance_score(df, model, col_name, model_name=None, method=None, seed=None, t=None):
    model_output = {i: 0 for i in range(df.shape[0])}
    while True:
        model_inputs = []
        not_solved = []
        for i, row in df.iterrows():
            if isinstance(model_output[i], int):
                model_inputs.append(eval_prompt.format(
                    concept=row[col_name].capitalize(),
                    property=row['property'] if model_name == None else row[f'{model_name}_{method}_{seed}_property'] if t == None else row[f'{model_name}_{method}_{seed}_{t}_property']
                ))
                not_solved.append(i)
        
        if len(not_solved) == 0:
            break
        output = model.generate(model_inputs, temperature=0.0, num_return_sequences=1)['responses']

        for i, o in zip(not_solved, output):
            try:
                for sample in o:
                    ast.literal_eval(sample.lower())['relevance']
                model_output[i] = o
            except:
                if model_output[i] > 2:
                    model_output[i] = ["{\"relevance\": 1:.1f}"]
                else:
                    model_output[i] += 1
                pass
        
    model_output = [value for key, value in sorted(model_output.items())]
            
    return [np.mean([int(ast.literal_eval(sample.lower())['relevance'])-1 for sample in o])/9 for o in model_output]



def custom_token_match(text):
    # Match hyphenated words
    return re.match(r'\b\w+(?:-\w+)+\b', text)

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer.token_match = custom_token_match

def remove_parentheses(text):
    return re.sub(r'\([^)]*\)', '', text).strip()

# input: noun phrase and head noun as a string
# output: dictionary containing noun phrase and modifier
def np_to_dict(noun_phrase, head_noun):
    noun_phrase = remove_parentheses(noun_phrase)
    if len(noun_phrase.split()) == 1:
        modifier = noun_phrase.replace(head_noun, "").strip().replace("-", "")
    else:
        noun_phrase_tokens = nlp(noun_phrase)
        modifier_tokens = []

        head_lemma = nlp(head_noun)[0].lemma_
        for token in noun_phrase_tokens:
            if token.lemma_ == head_noun or token.lemma_ == head_lemma:
                continue
            if token.pos_ in available_concept_pos: # extract keyword using spacy
                modifier_tokens.append(token)

        modifier = " ".join(token.text for token in modifier_tokens)
    result = dict(
        noun_phrase=noun_phrase,
        modifier=modifier
    )
        
    return result

def print_stat(df, eval_model_id="gpt-4o", baselines=[], type="emergence"):
    baselines = [baseline[:2] for baseline in baselines] if len(baselines) else [(col.split("_")[0], col.split("_")[1]) for col in df.columns if (f'{model_name}_{method}_0_emergence' in col)]
    baselines = list(set(baselines))
    print(f"CCPT Concept Max: {100 * df[f'meta.{eval_model_id}_indiv_max'].mean():.1f}")
    print(f"CCPT Combination: {100 * df[f'meta.combination_{eval_model_id}_relevance'].mean():.1f}")
    print(f"CCPT {type.capitalize()}: {100 * df[f'meta.{eval_model_id}_{type}'].mean():.1f}")

    for baseline in baselines:
        model_name, method = baseline
        # try:
        if method == "multi":
            method = "best+multi"
        print(f"{model_name}_{method} Concept Max: {100 * np.mean([df[f'{model_name}_{method}_0_indiv_max'].mean(), df[f'{model_name}_{method}_1_indiv_max'].mean(), df[f'{model_name}_{method}_2_indiv_max'].mean()]):.1f} ± {100 * np.std([df[f'{model_name}_{method}_0_indiv_max'].mean(), df[f'{model_name}_{method}_1_indiv_max'].mean(), df[f'{model_name}_{method}_2_indiv_max'].mean()]):.1f}")
        print(f"{model_name}_{method} Noun Phrase: {100 * np.mean([df[f'{model_name}_{method}_0_combination_relevance'].mean(), df[f'{model_name}_{method}_1_combination_relevance'].mean(), df[f'{model_name}_{method}_2_combination_relevance'].mean()]):.1f} ± {100 * np.std([df[f'{model_name}_{method}_0_combination_relevance'].mean(), df[f'{model_name}_{method}_1_combination_relevance'].mean(), df[f'{model_name}_{method}_2_combination_relevance'].mean()]):.1f}")
        print(f"{model_name}_{method} Score: {100 * np.mean([df[f'{model_name}_{method}_0_{type}'].mean(), df[f'{model_name}_{method}_1_{type}'].mean(), df[f'{model_name}_{method}_2_{type}'].mean()]):.1f} ± {100 * np.std([df[f'{model_name}_{method}_0_{type}'].mean(), df[f'{model_name}_{method}_1_{type}'].mean(), df[f'{model_name}_{method}_2_{type}'].mean()]):.1f}")
        # except:
        #     continue

def retry(n, output=None):
    def decorator(f):
        def wrapper(*args, **kwargs):
            for i in range(n):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    print(f"Error: {e}")
            if output is None:
                raise Exception(f"Failed after {n} retries")
            else:
                return output
        return wrapper
    return decorator

parser = argparse.ArgumentParser()
parser.add_argument("--property_type", type=str, required=False)
parser.add_argument("--local_host", type=str, required=False)
args = parser.parse_args()


def argparser():
    return args

import copy
def get_previous_concepts(C):
    new_C = copy.deepcopy(C)
    for k, v in C.items():
        if len(v) == 0:
            new_C[k] = new_C[k-1]
    return new_C