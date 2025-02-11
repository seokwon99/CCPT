import os
import pandas as pd
import ast
import numpy as np
from run.model.parallel import ParallelGPT
from run.format_checker import checker, get_error_pass
from run.prompts import eval_prompt, print_stat
import time
from data.pipeline.get_nounphrase import nlp
from tqdm import tqdm
from data.utils import merge_phrases, make_openai_batch, read_jsonlines
from multiprocessing import Pool, cpu_count
tqdm.pandas()

def get_root_and_modifier(line):
    noun_phrase = line['noun_phrase']
    
    token = merge_phrases(nlp(noun_phrase))[0]
    root = token._.root
    token_concepts = token._.C
    if root not in token_concepts or len(token_concepts) != 2:
        return None, None
    
    modifier = token_concepts[1] if token_concepts[0] == root else token_concepts[0]

    return root, modifier

available_concept_pos = ["ADJ", "ADV", "NOUN", "VERB", "PROPN"]

if __name__ == "__main__":
        
    # fpath = "data/pipeline/241212_vera_processed.jsonl"
    fpath = "data/pipeline/241227_expanded.csv"
    
    if fpath.endswith("jsonl"):
        df = pd.read_json(fpath, lines=True)
        fpath = fpath.replace("jsonl", "csv")
    else:
        df = pd.read_csv(fpath)
    
    df, baselines = checker(df, columns=["noun_phrase", "modifier"])

    # def parallel_apply(df, func):
    #     with Pool(cpu_count()) as pool:
    #         result = list(tqdm(pool.imap(func, [row for _, row in df.iterrows()]), total=df.shape[0]))
    #     return result
    # df['root_and_modifier'] = parallel_apply(df, get_root_and_modifier)
    # df['root'] = df['root_and_modifier'].apply(lambda x: x[0])
    # df['modifier'] = df['root_and_modifier'].apply(lambda x: x[1])
    # df.drop(columns=['root_and_modifier'], inplace=True)
    # df.dropna(subset=['root', 'modifier'], inplace=True)
    # df.reset_index(drop=True, inplace=True)
    
    # import pdb; pdb.set_trace()
    
    # df['root_rel_prompt'] = df.apply(lambda x: eval_prompt.format(
    #     concept=x['root'].capitalize(),
    #     property=x['property']
    # ), axis=1)
    # make_openai_batch(df['root_rel_prompt'].tolist(), "root")
    # del df['root_rel_prompt']
    
    # df['modifier_rel_prompt'] = df.apply(lambda x: eval_prompt.format(
    #     concept=x['modifier'].capitalize(),
    #     property=x['property']
    # ), axis=1)
    # make_openai_batch(df['modifier_rel_prompt'].tolist(), "modifier")
    # del df['modifier_rel_prompt']
    
    # df['noun_phrase_rel_prompt'] = df.apply(lambda x: eval_prompt.format(
    #     concept=x['noun_phrase'].capitalize(),
    #     property=x['property']
    # ), axis=1)
    # make_openai_batch(df['noun_phrase_rel_prompt'].tolist(), "noun_phrase")
    # del df['noun_phrase_rel_prompt']
    
    dir_path = "data/pipeline/250123_rel_result"
    
    df = pd.read_csv(fpath)
    _result = read_jsonlines(dir_path)
    result = {line['custom_id']: [choice['message']['content'] for choice in line['response']['body']['choices']] for line in _result}
    
    model = ParallelGPT(model_id='gpt-4o')
    
    def get_relevance_score(col_name, output=None):
        if output is None:
            model_output = {i: 0 for i in range(df.shape[0])}
        else:
            model_output = dict()
            for key in output:
                try:
                    o = output[key]
                    for sample in o:
                        ast.literal_eval(sample.lower())['relevance']
                    model_output[key] = o
                except:
                    model_output[key] = 0

        while True:
            model_inputs = []
            not_solved = []
            for i, row in df.iterrows():
                if isinstance(model_output[i], int):
                    model_inputs.append(eval_prompt.format(
                        concept=row[col_name].capitalize(),
                        property=row['property']
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
                        model_output[i] = ["{\"relevance\": 1}"]
                    else:
                        model_output[i] += 1
                    pass
            
        model_output = [value for key, value in sorted(model_output.items())]
        return [np.mean([int(ast.literal_eval(sample.lower())['relevance'])-1 for sample in o])/9 for o in model_output]
    
    df['root_rel'] = df.index.to_series().apply(lambda x: result.get(f"root-{x+1}", [None]))
    df['modifier_rel'] = df.index.to_series().apply(lambda x: result.get(f"modifier-{x+1}", [None]))
    df['noun_phrase_rel'] = df.index.to_series().apply(lambda x: result.get(f"noun_phrase-{x+1}", [None]))

    df['root_rel'] = get_relevance_score("root", {i: val for i, val in df['root_rel'].items()})
    df['modifier_rel'] = get_relevance_score("modifier", {i: val for i, val in df['modifier_rel'].items()})
    df['noun_phrase_rel'] = get_relevance_score("noun_phrase", {i: val for i, val in df['noun_phrase_rel'].items()})
    
    df['individual_max'] = df.apply(lambda x: max(x['root_rel'], x['modifier_rel']), axis=1)
    df['ccpt_score'] = df.apply(lambda x: max(0, x['noun_phrase_rel'] - max(x['root_rel'], x['modifier_rel'])), axis=1)
    df.to_csv(fpath, index=False)