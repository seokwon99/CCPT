import pandas as pd
import numpy as np
import spacy

from experiment.model.openai import ParallelGPT as MODEL
from experiment.format_checker import checker, get_error_pass
from experiment.utils import get_relevance_score, np_to_dict, print_stat, argparser

nlp = spacy.load("en_core_web_sm")
available_concept_pos = ["ADJ", "ADV", "NOUN", "VERB", "PROPN"]

args = argparser()

if __name__ == "__main__":
    
    task_type = args.property_type
    model_id = "Meta-Llama-3.1-70B-Instruct"
    method = "sa-llm&conceptnet" # naive, cot, sa-llm, sa-conceptnet, sa-llm&conceptnet
    fpath = f"results/npc_{task_type}_{model_id}_{method}.csv"

    eval_model_id = "gpt-4o"

    if fpath.endswith("jsonl"):
        df = pd.read_json(fpath, lines=True)
        fpath = fpath.replace("jsonl", "csv")
    else:
        df = pd.read_csv(fpath)

    df, baselines = checker(df, columns=["combination", "modifier"])

    model = MODEL(model_id=eval_model_id)
    
    if not f'meta.{eval_model_id}_emergence' in df:
        df[f'meta.combination_{eval_model_id}_relevance'] = get_relevance_score(df, model, "combination")
        df[f'meta.root_{eval_model_id}_relevance'] = get_relevance_score(df, model, "root")
        df[f'meta.modifier_{eval_model_id}_relevance'] = get_relevance_score(df, model, "modifier")
        df[f'meta.{eval_model_id}_indiv_max'] = df.apply(lambda x: max(x[f'meta.root_{eval_model_id}_relevance'], x[f'meta.modifier_{eval_model_id}_relevance']), axis=1)
        df[f'meta.{eval_model_id}_emergence'] = df.apply(lambda x: max(0, x[f'meta.combination_{eval_model_id}_relevance'] - x[f'meta.{eval_model_id}_indiv_max']), axis=1)
        df.to_csv(fpath, index=False)

    baselines = list(set(baselines))
    for baseline in baselines:
        model_name, method, seed = baseline
        if int(seed) > 2:
            continue
        
        
        if 'multi' in method:
            try:
                N = int(method.split("-")[-1])
            except:
                N = 5
            for i in range(N):
                if f'{model_name}_{method}_{seed}_{i}_emergence' in df:
                    continue
                print(baseline)
                df[f'{model_name}_{method}_{seed}_{i}_generated'] = df[f'{model_name}_{method}_{seed}_generated_'].apply(lambda x: get_error_pass(x.replace("\\n", "").replace("']", "}']").lower()))
                df[f'{model_name}_{method}_{seed}_{i}_generated'] = df[f'{model_name}_{method}_{seed}_{i}_generated'].apply(lambda x: x[i] if (len(x)-1 >= i) else x[-1])
                df[f'{model_name}_{method}_{seed}_{i}_combination'] = df[f'{model_name}_{method}_{seed}_{i}_generated'].apply(lambda x: x['combination'])
                
                df[f'{model_name}_{method}_{seed}_{i}_modifier'] = df[f'{model_name}_{method}_{seed}_{i}_generated'].apply(lambda x: x['modifier'] if 'modifier' in x else "")
                df[f'{model_name}_{method}_{seed}_{i}_modifier'] = df[f'{model_name}_{method}_{seed}_{i}_modifier'].apply(lambda x: [token.lemma_ for token in nlp(x) if token.pos_ in available_concept_pos][0] if len([token.lemma_ for token in nlp(x) if token.pos_ in available_concept_pos]) else None)                   
                df[f'{model_name}_{method}_{seed}_{i}_modifier'] = df.apply(lambda x: None if x['root'] == x[f'{model_name}_{method}_{seed}_{i}_modifier'] else x[f'{model_name}_{method}_{seed}_{i}_modifier'], axis=1)
                df[f'{model_name}_{method}_{seed}_{i}_modifier'] = df.apply(lambda x: x[f'{model_name}_{method}_{seed}_{i}_modifier'] if x[f'{model_name}_{method}_{seed}_{i}_modifier'] is not None else np_to_dict(x[f'{model_name}_{method}_{seed}_{i}_combination'], x['root'])['modifier'], axis=1)
                
                df[f'{model_name}_{method}_{seed}_{i}_combination_relevance'] = get_relevance_score(df, model, f'{model_name}_{method}_{seed}_{i}_combination')
                df[f'{model_name}_{method}_{seed}_{i}_modifier_relevance'] = get_relevance_score(df, model, f'{model_name}_{method}_{seed}_{i}_modifier')
                df[f'{model_name}_{method}_{seed}_{i}_indiv_max'] = df.apply(lambda x: max(x[f'meta.root_{eval_model_id}_relevance'], x[f'{model_name}_{method}_{seed}_{i}_modifier_relevance']), axis=1)
                df[f'{model_name}_{method}_{seed}_{i}_emergence'] = df.apply(lambda x: max(0, x[f'{model_name}_{method}_{seed}_{i}_combination_relevance'] - x[f'{model_name}_{method}_{seed}_{i}_indiv_max']), axis=1)
                df.to_csv(fpath, index=False)

            # Find best answer
            df[f'{model_name}_mean+{method}_{seed}_modifier_relevance'] = df.apply(lambda x: np.mean([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and ('modifier_relevance' in col)]), axis=1)
            df[f'{model_name}_best+{method}_{seed}_modifier_relevance'] = df.apply(lambda x: min([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and ('modifier_relevance' in col)]), axis=1)
            df[f'{model_name}_mean+{method}_{seed}_combination_relevance'] = df.apply(lambda x: np.mean([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and ('_combination_relevance' in col)]), axis=1)
            df[f'{model_name}_best+{method}_{seed}_combination_relevance'] = df.apply(lambda x: max([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and ('_combination_relevance' in col)]), axis=1)
            df[f'{model_name}_mean+{method}_{seed}_indiv_max'] = df.apply(lambda x: np.mean([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and ('_indiv_max' in col)]), axis=1)
            df[f'{model_name}_best+{method}_{seed}_indiv_max'] = df.apply(lambda x: min([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and ('_indiv_max' in col)]), axis=1)
            df[f'{model_name}_mean+{method}_{seed}_emergence'] = df.apply(lambda x: np.mean([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and ('_emergence' in col)]), axis=1)
            df[f'{model_name}_best+{method}_{seed}_emergence'] = df.apply(lambda x: max([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and ('_emergence' in col)]), axis=1)
            
        else:
            # if f'{model_name}_{method}_{seed}_emergence' in df:
                # continue
            print(baseline)
            df[f'{model_name}_{method}_{seed}_generated'] = df[f'{model_name}_{method}_{seed}_generated_'].apply(lambda x: get_error_pass(x.replace("\\n", "").replace("']", "}']").lower())[-1])
            df[f'{model_name}_{method}_{seed}_combination'] = df[f'{model_name}_{method}_{seed}_generated'].apply(lambda x: x['combination'])
            
            df[f'{model_name}_{method}_{seed}_modifier'] = df[f'{model_name}_{method}_{seed}_generated'].apply(lambda x: x['modifier'] if 'modifier' in x else "")
            df[f'{model_name}_{method}_{seed}_modifier'] = df[f'{model_name}_{method}_{seed}_modifier'].apply(lambda x: [token.lemma_ for token in nlp(x) if token.pos_ in available_concept_pos][0] if len([token.lemma_ for token in nlp(x) if token.pos_ in available_concept_pos]) else None)
            df[f'{model_name}_{method}_{seed}_modifier'] = df.apply(lambda x: None if x['root'] == x[f'{model_name}_{method}_{seed}_modifier'] else x[f'{model_name}_{method}_{seed}_modifier'], axis=1)
            df[f'{model_name}_{method}_{seed}_modifier'] = df.apply(lambda x: x[f'{model_name}_{method}_{seed}_modifier'] if x[f'{model_name}_{method}_{seed}_modifier'] is not None else np_to_dict(x[f'{model_name}_{method}_{seed}_combination'], x['root'])['modifier'], axis=1)

            df[f'{model_name}_{method}_{seed}_combination_relevance'] = get_relevance_score(df, model, f'{model_name}_{method}_{seed}_combination')
            df[f'{model_name}_{method}_{seed}_modifier_relevance'] = get_relevance_score(df, model, f'{model_name}_{method}_{seed}_modifier')
            df[f'{model_name}_{method}_{seed}_indiv_max'] = df.apply(lambda x: max(x[f'meta.root_{eval_model_id}_relevance'], x[f'{model_name}_{method}_{seed}_modifier_relevance']), axis=1)
            df[f'{model_name}_{method}_{seed}_emergence'] = df.apply(lambda x: max(0, x[f'{model_name}_{method}_{seed}_combination_relevance'] - x[f'{model_name}_{method}_{seed}_indiv_max']), axis=1)
        
        df.to_csv(fpath, index=False)

    print_stat(df, eval_model_id=eval_model_id, baselines=baselines, type="emergence")