import os
from tqdm import tqdm
import pandas as pd
import csv
import multiprocessing
tqdm.pandas()
import spacy
nlp = spacy.load("en_core_web_sm")

def get_cooccurence(data):
    lod, file_path = data
    df = pd.read_csv(file_path, compression='gzip', on_bad_lines = "skip", sep="\t", header=None, quoting=csv.QUOTE_NONE)
    
    result = []
    for d in lod:
        w1 = d['root_org']
        w2 = d['modifier_org']
        w1_occur_df = df[(df[0] == w1) | (df[1] == w1)]
        w2_occur_df = df[(df[0] == w2) | (df[1] == w2)]
        co_occur_df = w1_occur_df[(w1_occur_df[0] == w2) | (w1_occur_df[1] == w2)]
        result.append(dict(
            tot = df[2].sum(),
            w1_occur = w1_occur_df[2].sum(),
            w2_occur = w2_occur_df[2].sum(),
            co_occur = co_occur_df[2].sum()
        ))
    
    return result


def get_frequency_multiprocessing(df, dir="downloads/google_ngrams/5_cooccurrence"):    
    
    df['modifier_org'] = df.apply(lambda x: [token.text for token in nlp(x['noun_phrase']) if token.lemma_ == x['modifier']][0], axis=1)
    df['root_org'] = df.apply(lambda x: [token.text for token in nlp(x['noun_phrase']) if token.lemma_ == x['root']][0], axis=1)
    
    file_paths = [(df[['modifier_org', 'root_org']].to_dict("records"), os.path.join(dir, file)) for file in os.listdir(dir)]
    
    with multiprocessing.Pool(50) as p:
        results = list(tqdm(p.imap(get_cooccurence, file_paths), total=len(file_paths)))
        p.close()
        p.join()
    new_result = [{"tot": 0, "w1_occur": 0, "w2_occur": 0, "co_occur": 0} for _ in results[0]]
    
    for result in results:
        for i, r in enumerate(result):
            new_result[i]['tot'] += r['tot']
            new_result[i]['w1_occur'] += r['w1_occur']
            new_result[i]['w2_occur'] += r['w2_occur']
            new_result[i]['co_occur'] += r['co_occur']
    return new_result

def get_cooccurence_multiprocessing(df, dir="downloads/google_ngrams/5_cooccurrence"):    
    
    file_paths = [(df[['modifier_org', 'root_org']].to_dict("records"), os.path.join(dir, file)) for file in os.listdir(dir)]
    
    with multiprocessing.Pool(50) as p:
        results = list(tqdm(p.imap(get_cooccurence, file_paths), total=len(file_paths)))
        p.close()
        p.join()
    new_result = [{"tot": 0, "w1_occur": 0, "w2_occur": 0, "co_occur": 0} for _ in results[0]]
    
    for result in results:
        for i, r in enumerate(result):
            new_result[i]['tot'] += r['tot']
            new_result[i]['w1_occur'] += r['w1_occur']
            new_result[i]['w2_occur'] += r['w2_occur']
            new_result[i]['co_occur'] += r['co_occur']
    return new_result

def get_df(file_path):
    return pd.read_csv(file_path, compression='gzip', on_bad_lines = "skip", sep="\t", header=None, quoting=csv.QUOTE_NONE)
    
def get_tot_df(dir="downloads/google_ngrams/5_cooccurrence"):
    file_paths = [os.path.join(dir, file) for file in os.listdir(dir)]
    with multiprocessing.Pool(50) as p:
        dfs = list(tqdm(p.imap(get_df, file_paths), total=len(file_paths)))
        p.close()
        p.join()
    return pd.concat(dfs)

def get_cooccurence_with_df(w1, w2, df):    
    w1_occur_df = df[(df[0] == w1) | (df[1] == w1)]
    w2_occur_df = df[(df[0] == w2) | (df[1] == w2)]
    co_occur_df = w1_occur_df[(w1_occur_df[0] == w2) | (w1_occur_df[1] == w2)]
    
    return df, dict(
        tot = df[2].sum(),
        w1_occur = w1_occur_df[2].sum(),
        w2_occur = w2_occur_df[2].sum(),
        co_occur = co_occur_df[2].sum()
    )

def get_topk(df, w1, k=10):
    df = df[(df[0] == w1) | (df[1] == w1)]
    import pdb; pdb.set_trace()
    df = df.sort_values(by=2, ascending=False)
    return df.iloc[:k]

if __name__ == "__main__":
    ## For demo
    # print(get_cooccurence_multiprocessing("apple", "red"))
    
    ## Frequency tagging
    fpath = "analysis/conceptnet_bigram.csv"
    spath = "analysis/conceptnet_bigram.csv"
    df = pd.read_csv(fpath)
    
    df['modifier_org'] = df.apply(lambda x: [token.text for token in nlp(x['noun_phrase']) if token.lemma_ == x['modifier']][0] if len([token.text for token in nlp(x['noun_phrase']) if token.lemma_ == x['modifier']]) else None, axis=1)
    df['root_org'] = df.apply(lambda x: [token.text for token in nlp(x['noun_phrase']) if token.lemma_ == x['root']][0] if len([token.text for token in nlp(x['noun_phrase']) if token.lemma_ == x['root']]) else None, axis=1)
    df = df[df.apply(lambda x: x['modifier_org'] is not None and x['root_org'] is not None, axis=1)]
    
    # db = get_tot_df()
    
    # df = df.iloc[:30]
    df['occurence'] = get_cooccurence_multiprocessing(df)
    df.to_csv(spath, index=False)