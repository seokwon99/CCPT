from wordfreq import word_frequency
from tqdm import tqdm
import json
import spacy
import multiprocessing
import pandas as pd

available_concept_pos = ["ADJ", "ADV", "NOUN", "PROPN", "VERB"]


nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("results/conceptnet5.csv")
concepts = df['concept1'].tolist() + df['concept2'].tolist()
concepts = list(set(concepts))
concepts = [concept for concept in concepts if isinstance(concept, str) and concept.count("_") == 0]

def get_frequency(sample):
    doc = nlp(sample)
    if len(doc) > 1:
        return None
    token = doc[0]
    if token.pos_ not in available_concept_pos:
        return None
    sample = token.text
    return {sample: word_frequency(sample, 'en')}

if __name__ == "__main__":
    # data = data[:100000]
    pool = multiprocessing.Pool()
    with multiprocessing.Pool(50) as p:
      data = list(tqdm(p.imap(get_frequency, concepts), total=len(concepts)))
    pool.close()
    pool.join()
    
    new_data = dict()
    for sample in data:
        if sample is not None and list(sample.values())[0] != 0:
            new_data.update(sample)
    new_data = dict(sorted(new_data.items(), key=lambda item: item[1], reverse=True))
    with open("dataset/processed/conceptnet_entity.json", "w") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)