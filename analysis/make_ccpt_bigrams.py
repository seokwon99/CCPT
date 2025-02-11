import pandas as pd
import json
import os
from tqdm import tqdm
from datasets import load_dataset
import spacy
nlp = spacy.load("en_core_web_sm")


fpath = "data/pipeline/conceptnet_entity_processed.json"

if not os.path.isfile(fpath):
    df = pd.read_csv("run/conceptnet5.csv")
    concepts = df['concept1'].tolist() + df['concept2'].tolist()
    unigram = [concept for concept in concepts if isinstance(concept, str) and concept.count("_") == 0]
    unigram = list(set([concept for concept in unigram if len(concept) > 2]))
    bigram = list(set([concept for concept in concepts if isinstance(concept, str) and concept.count("_") == 1]))
        
    with open(fpath, "w") as f:
        json.dump(dict(
            unigram=unigram,
            bigram=bigram
        ), f, indent=4, ensure_ascii=False)
        
    del df, concepts

with open(fpath) as f:
    data = json.load(f)
concept = data['unigram']

paths = [
    "ccpt_emergent.csv",
    "ccpt_canceled.csv"
]
ccpt = pd.concat([
    pd.read_csv(path)[['combination', 'root', 'modifier']] for path in paths
])

pass_words = set()
pass_words.update(['and', 'or', 'of', 'regular', 'real', 'normal', 'typical', 'actual', 'average', 'original', 'standard', 'fuck', 'reddit', 'barrel', 'box', 'shot', 'sack', 'piece', 'kind', 'bag', 'amount', 'most', 'any']) # not combination
pass_words.update(['barrel', 'box', 'shot', 'sack', 'piece', 'kind', 'bag', 'amount', 'type'])


bigram = ccpt.apply(lambda x: [x['root'], x['modifier']], axis=1).tolist()

def f(bi):
    return nlp(" ".join(bi))

bigram = [f(bi) for bi in tqdm(bigram)]

bigram = [{"combination": bi.text, "root": bi[0].head.head.head.text, "modifier": [c for c in bi.text.split() if c in concept]} for bi in bigram if (bi[0].head.head.head.pos_ == "NOUN")]


output = []
for bi in bigram:
    try:
        output.append({"combination": bi['combination'], "root": bi['root'], "modifier": bi['modifier'][1] if bi['root'] == bi['modifier'][0] else bi['modifier'][0]})
    except:
        pass

bigram = [bi for bi in tqdm(output) if (bi['root'] in concept) and (bi['modifier'] in concept)]
bigram_df = pd.DataFrame(bigram)
bigram_df = bigram_df.groupby(["root", "modifier"]).apply(lambda x: x.iloc[0])

bigram_df.to_csv("analysis/ccpt_bigram.csv", index=False)