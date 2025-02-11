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
    model_id = "gpt-4o"
    method = "sa-nf-conceptnet" # naive, cot, sa-llm, sa-conceptnet, sa-llm&conceptnet
    fpath = f"results/pi_{task_type}_{model_id}_{method}.csv"

    eval_model_id = "gpt-4o"
    score_type = "emergence" if task_type == "emergent" else "cancellation"
    
    if fpath.endswith("jsonl"):
        df = pd.read_json(fpath, lines=True)
        fpath = fpath.replace("jsonl", "csv")
    else:
        df = pd.read_csv(fpath)
    
    df, baselines = checker(df, columns=["property"])

    model = MODEL(model_id=eval_model_id)

    if not f'meta.{eval_model_id}_{score_type}' in df:
        df[f'meta.combination_{eval_model_id}_relevance'] = get_relevance_score(df, model, "combination")
        df[f'meta.root_{eval_model_id}_relevance'] = get_relevance_score(df, model, "root")
        df[f'meta.modifier_{eval_model_id}_relevance'] = get_relevance_score(df, model, "modifier")
        df[f'meta.{eval_model_id}_indiv_max'] = df.apply(lambda x: max(x[f'meta.root_{eval_model_id}_relevance'], x[f'meta.modifier_{eval_model_id}_relevance']), axis=1)
        if task_type == "emergent":
            df[f'meta.{eval_model_id}_{score_type}'] = df.apply(lambda x: max(0, x[f'meta.combination_{eval_model_id}_relevance'] - x[f'meta.{eval_model_id}_indiv_max']), axis=1)
        else:
            df[f'meta.{eval_model_id}_{score_type}'] = df.apply(lambda x: max(0, x[f'meta.{eval_model_id}_indiv_max'] - x[f'meta.combination_{eval_model_id}_relevance']), axis=1)
        df.to_csv(fpath, index=False)

    baselines = list(set(baselines))
    for baseline in baselines:
        model_name, method, seed = baseline
        if int(seed) > 2:
            continue
        print(baseline)
        
        if 'multi' in method:
            try:
                N = int(method.split("-")[-1])
            except:
                N = 5
            for i in range(N):
                if f'{model_name}_{method}_{seed}_{i}_{score_type}' in df:
                    continue
                print(f'{model_name}_{method}_{seed}_{i}')
                df[f'{model_name}_{method}_{seed}_{i}_generated'] = df[f'{model_name}_{method}_{seed}_generated_'].apply(lambda x: get_error_pass(x.replace("\\n", "").replace("']", "}']").lower()))
                df[f'{model_name}_{method}_{seed}_{i}_generated'] = df[f'{model_name}_{method}_{seed}_{i}_generated'].apply(lambda x: x[i] if (len(x)-1 >= i) else x[-1])
                df[f'{model_name}_{method}_{seed}_{i}_property'] = df[f'{model_name}_{method}_{seed}_{i}_generated'].apply(lambda x: x['property'])

                df[f'{model_name}_{method}_{seed}_{i}_combination_relevance'] = get_relevance_score(df, model, f'combination', model_name=model_name, method=method, seed=seed, t=i)
                df[f'{model_name}_{method}_{seed}_{i}_root_relevance'] = get_relevance_score(df, model, f'root', model_name=model_name, method=method, seed=seed, t=i)
                df[f'{model_name}_{method}_{seed}_{i}_modifier_relevance'] = get_relevance_score(df, model, f'modifier', model_name=model_name, method=method, seed=seed, t=i)
                df[f'{model_name}_{method}_{seed}_{i}_indiv_max'] = df.apply(lambda x: max(x[f'{model_name}_{method}_{seed}_{i}_root_relevance'], x[f'{model_name}_{method}_{seed}_{i}_modifier_relevance']), axis=1)
                df[f'{model_name}_{method}_{seed}_{i}_{score_type}'] = df.apply(lambda x: max(0,  x[f'{model_name}_{method}_{seed}_{i}_combination_relevance'] - x[f'{model_name}_{method}_{seed}_{i}_indiv_max'] if task_type == "emergent" else x[f'{model_name}_{method}_{seed}_{i}_indiv_max'] - x[f'{model_name}_{method}_{seed}_{i}_combination_relevance']), axis=1)
                df.to_csv(fpath, index=False)
            df[f'{model_name}_mean+{method}_{seed}_combination_relevance'] = df.apply(lambda x: np.mean([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and ('_combination_relevance' in col)]), axis=1)
            df[f'{model_name}_best+{method}_{seed}_combination_relevance'] = df.apply(lambda x: (max if task_type == "emergent" else min)([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and ('_combination_relevance' in col)]), axis=1)
            df[f'{model_name}_mean+{method}_{seed}_indiv_max'] = df.apply(lambda x: np.mean([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and ('_indiv_max' in col)]), axis=1)
            df[f'{model_name}_best+{method}_{seed}_indiv_max'] = df.apply(lambda x: (min if task_type == "emergent" else max)([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and ('_indiv_max' in col)]), axis=1)
            df[f'{model_name}_mean+{method}_{seed}_{score_type}'] = df.apply(lambda x: np.mean([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and (f'_{score_type}' in col)]), axis=1)
            df[f'{model_name}_best+{method}_{seed}_{score_type}'] = df.apply(lambda x: max([x[col] for col in df.columns if (f'{model_name}_{method}_{seed}' in col) and (f'_{score_type}' in col)]), axis=1)
        else:
            # if not f'{model_name}_{method}_{seed}_{score_type}' in df:
            if True:
                df[f'{model_name}_{method}_{seed}_generated'] = df[f'{model_name}_{method}_{seed}_generated_'].apply(lambda x: get_error_pass(x.replace("\\n", "").replace("']", "}']").lower())[-1])
                df[f'{model_name}_{method}_{seed}_property'] = df[f'{model_name}_{method}_{seed}_generated'].apply(lambda x: x['property'])

                df[f'{model_name}_{method}_{seed}_combination_relevance'] = get_relevance_score(df, model, f'combination', model_name=model_name, method=method, seed=seed)
                df[f'{model_name}_{method}_{seed}_root_relevance'] = get_relevance_score(df, model, f'root', model_name=model_name, method=method, seed=seed)
                df[f'{model_name}_{method}_{seed}_modifier_relevance'] = get_relevance_score(df, model, f'modifier', model_name=model_name, method=method, seed=seed)
                df[f'{model_name}_{method}_{seed}_indiv_max'] = df.apply(lambda x: max(x[f'{model_name}_{method}_{seed}_root_relevance'], x[f'{model_name}_{method}_{seed}_modifier_relevance']), axis=1)
                df[f'{model_name}_{method}_{seed}_{score_type}'] = df.apply(lambda x: max(0,  x[f'{model_name}_{method}_{seed}_combination_relevance'] - x[f'{model_name}_{method}_{seed}_indiv_max'] if task_type == "emergent" else x[f'{model_name}_{method}_{seed}_indiv_max'] - x[f'{model_name}_{method}_{seed}_combination_relevance']), axis=1)
                df.to_csv(fpath, index=False)
        
        df.to_csv(fpath, index=False)
    
    print_stat(df, eval_model_id=eval_model_id, baselines=baselines, type=score_type)