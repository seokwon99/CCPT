# from mediawiki import MediaWiki
import spacy
import json
from tqdm import tqdm
import os.path
import multiprocessing
import itertools
import pandas as pd
from spacy.matcher import Matcher
from data.utils import merge_phrases, load_json_files_from_folder


nlp = spacy.load("en_core_web_sm")

pattern_as = [
    [{"LOWER": "as"}, {"POS": "ADV"}, {"LOWER": "as"}, {"_": {"NP": True}}],
    [{"LOWER": "as"}, {"POS": "ADJ"}, {"LOWER": "as"}, {"_": {"NP": True}}],
]
pattern_like = [
    [{"POS": "ADV"}, {"LOWER": "like", "POS": "ADP"}, {"_": {"NP": True}}],
    [{"POS": "ADJ"}, {"LOWER": "like", "POS": "ADP"}, {"_": {"NP": True}}]
]

# Initialize the Matcher with the provided patterns
matcher = Matcher(nlp.vocab)

matcher.add("simile_as", pattern_as)
matcher.add("simile_like", pattern_like)

fpath = "data/pipeline/conceptnet_entity_processed.json"

if os.path.isfile(fpath):
    with open(fpath) as f:
        data = json.load(f)
    unigram = data['unigram']
    bigram = data['bigram']
else:
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

def get_concept(token):
    return [t for t in token.lemma_.split() if t in unigram]

def get_is_noun_phrase(token):
    lemma = token.lemma_.split()
    if token.pos_ != "NOUN":
        return False
    elif len(lemma) != 2:
        return False
    elif any([l for l in lemma if l not in unigram]):
        return False
    elif "_".join(lemma) in bigram:
        return False
    else:
        return True

spacy.tokens.Token.set_extension("C", getter=get_concept) # turn off if not necessary
spacy.tokens.Token.set_extension("NP", getter=get_is_noun_phrase)

def get_sents(data_point):
    
    sent, file_path = data_point
    
    doc = nlp(sent)
    
    phrase_sents = []
    for sent in doc.sents:
        sent = sent.as_doc()
        
        # Filter-out short sentences (high probability of being less informative)
        if len(sent) < 20:
            continue
        
        sent = merge_phrases(sent)
        matches = matcher(sent)
        if matches:
            for match_id, start, end in matches:

                span = sent[start:end]
                string_match_id = nlp.vocab.strings[match_id]
                if string_match_id == "simile_as":
                    noun_phrase = span[-1]
                    property = span[1]
                    if property.text in ['far', 'long', 'even', 'well']:
                        continue
                elif string_match_id == "simile_like":
                    noun_phrase = span[-1]
                    property = span[0]
                
                # if noun phrase contain digits
                if any([c for c in noun_phrase.text if c.isdigit()]):
                    continue
                
                
                # if noun phrase contain context dependent words such as pronouns                       
                if noun_phrase._.pron:
                    continue
                
                phrase_sents.append((noun_phrase.text, sent.text, file_path))
        
    return phrase_sents

if __name__ == "__main__":
    
    # Corpus
    dir_path = "dataset/raw/todo"
    dataset_name = "reddit/writingprompts_comments"
    
    data = load_json_files_from_folder(f"conceptual_combination/dataset/raw/todo/{dataset_name}")

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        data = list(tqdm(p.imap(get_sents, data), total=len(data)))
        p.close()
        p.join()
    
    phrase_sents = list(itertools.chain.from_iterable(data))
    
    dataset_name = dataset_name.replace("/", "_")
    with open(f"dataset/pipeline/results/extraction_{dataset_name}.json", "w") as f:
        json.dump(phrase_sents, f, indent=4, ensure_ascii=False)