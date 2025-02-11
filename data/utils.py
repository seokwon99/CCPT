import jsonlines

import re
import spacy
import json
import numpy as np
from collections import Counter
from spacy.tokens import Span
import os
from spacy.tokens import Token
Token.set_extension("pron", default=False)
Token.set_extension("root", default="")

nlp = spacy.load("en_core_web_sm")
available_concept_pos = ["ADJ", "ADV", "NOUN", "VERB", "PROPN"]

deictic_expressions = set([
    # Pronouns
    # "I", "you", "he", "she", "it", "we", "they",
    # "mine", "yours", "his", "hers", "ours", "theirs",

    # Demonstratives
    "this", "that", "these", "those",

    # Adverbs of Place
    "here", "there", "above", "below", "nearby",

    # Adverbs of Time
    "now", "then", "today", "tomorrow", "yesterday", "later", "soon", "recently",

    # Determiners
    "some", "many", "few", "all", "none", "other", "another", "each", "every",
])

def preprocessing(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s\,\.\?\!\']', '', text)
    # cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

def make_whitespaces_single(text):
    # a linebreak to a space
    cleaned_text = re.sub(r'\s+', ' ', text)
    
    return cleaned_text


def children_indexes(token, depth=0):
    if depth > 10:
        return []
    indexes = [token.i]
    for child in list(token.children):
        indexes.extend(children_indexes(child, depth=depth+1))
    indexes = list(set(indexes))
    indexes.sort()
    
    return indexes

def get_phrase(doc, token, lemma=True):
    indexes = children_indexes(token)
    return " ".join([doc[i].lemma_ if lemma else doc[i] for i in indexes])

def get_except_phrase(doc, token, children=True):
    if children:
        indexes = children_indexes(token)
    else:
        indexes = [token.i]
    return " ".join([token.text for token in doc if token.i not in indexes])

def get_predicate(token):
    if token.pos_ == "AUX" or token.pos_ == "VERB":
        return token
    if token == token.head:
        return None
    return get_predicate(token.head)

MAX_VAL=100000000000

def merge_phrases(doc):
    if isinstance(doc, str):
        doc = nlp(doc)
    
    with doc.retokenize() as retokenizer:
        cur_idx = -1
        for noun_phrase in list(doc.noun_chunks):
            full_idx = children_indexes(noun_phrase.root)
            
            ## concat adp
            right_idx = noun_phrase.root.i + 1
            if len(doc) > right_idx and doc[right_idx].pos_ == "ADP":
                full_idx = list(set(full_idx + children_indexes(doc[right_idx])))
                full_idx.sort()
            
            ## concat verb
            left_idx = noun_phrase.root.i - 1
            if left_idx > 0 and doc[left_idx].pos_ == "VERB" and (doc[left_idx].text.endswith("ing") or doc[left_idx].text.endswith("ed")):
                full_idx = list(set(full_idx + children_indexes(doc[left_idx])))
                full_idx.sort()
            elif len(doc) > right_idx and doc[right_idx].pos_ == "VERB" and (doc[right_idx].text.endswith("ing") or doc[right_idx].text.endswith("ed")):
                full_idx = list(set(full_idx + children_indexes(doc[right_idx])))
                full_idx.sort()
            else:
                pass
            
            if np.min(full_idx) <= cur_idx:
                continue
            
            # if any([i for i in full_idx if doc[i].pos_ == "PRON"]):
            #     continue
            
            cur_idx = np.max(full_idx)
            attrs = {
                "LEMMA": " ".join([doc[i].lemma_ for i in full_idx if doc[i].pos_ in available_concept_pos]),
                "POS": noun_phrase.root.pos_,
                "_": {
                    "pron": any([doc[i] for i in full_idx if (doc[i].pos_ == "PRON" or doc[i].lemma_ in deictic_expressions)]),
                    "root": noun_phrase.root.lemma_
                }
            }
            retokenizer.merge(Span(doc, start=full_idx[0], end=full_idx[-1]+1), attrs=attrs)
    return doc

def merge_punct(doc):
    spans = []
    for word in doc[:-1]:
        if word.is_punct or not word.nbor(1).is_punct:
            continue
        start = word.i
        end = word.i + 1
        while end < len(doc) and doc[end].is_punct:
            end += 1
        span = doc[start:end]
        spans.append((span, word.tag_, word.lemma_, word.ent_type_))
    with doc.retokenize() as retokenizer:
        for span, tag, lemma, ent_type in spans:
            attrs = {"tag": tag, "lemma": lemma, "ent_type": ent_type}
            retokenizer.merge(span, attrs=attrs)
    return doc
    
def read_jsonlines(path:str):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_main(word):
    doc = nlp(word)
    doc = merge_phrases(doc)
    return doc[0].norm_

def extract_lemma(word):
    doc = nlp(word)
    return " ".join([token.lemma_ for token in doc if token.pos_ != "DET"])

bypass_words = [
    "like to",
    "like case",
    "all like",
    "part and part like",
    "like enough",
    "like as not",
    "like another",
    "like any other",
    "more like",
    "littel like",
    "much like",
    "less like",
    "like new",
    "like so",
    "like that",
    "as soon as",
    "as long as",
    "as far as",
    "as much as"
]

def select_elements_equally(lists):
    N = len(lists)
    
    # 최소한 하나의 리스트가 존재해야 함
    if N == 0:
        return []
    
    # 각 리스트에서 몇 개의 요소를 뽑아야 하는지 계산
    num_elements_per_list = 5 // N
    extra_elements = 5 % N
    
    selected_elements = []
    
    # 각 리스트에서 num_elements_per_list 개수만큼 요소를 뽑아 추가
    for i in range(num_elements_per_list):
        for lst in lists:
            if len(lst) > i:
                selected_elements.append(lst[i])
    
    # 남은 요소를 각 리스트에서 하나씩 추가 (총 5개가 되도록)
    for i in range(extra_elements):
        if len(lists[i]) > num_elements_per_list:
            selected_elements.append(lists[i][num_elements_per_list])
    
    return selected_elements

def pos_count(word):
    doc = nlp(word)
    pos = [token.pos_ for token in doc]
    return dict(Counter(pos))

def dep_count(word):
    doc = nlp(word)
    dep = [token.dep_ for token in doc]
    return dict(Counter(dep))

def is_ancestor(target, source):
    if source.head == source:
        return False
    elif source.head == target:
        return True
    else:
        return is_ancestor(target, source.head)
    
def load_json_files_from_folder(folder_path):
    json_data = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data = [(sent, file_path) for sent in data]
                        json_data.extend(data)
                except Exception as e:
                    print(f"Error loading JSON file {file_path}: {e}")
    
    return json_data

def read_jsonlines(path):
    data = []
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.jsonl'):
                    with jsonlines.open(os.path.join(root, file)) as read_file:
                        for line in read_file.iter():
                            data.append(line)
    elif os.path.isfile(path) and path.endswith('.jsonl'):
        with jsonlines.open(path) as read_file:
            for line in read_file.iter():
                data.append(line)
    else:
        raise ValueError("Invalid path")
    return data

def make_openai_batch(inputs, header):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    prompts = [dict(
        custom_id = f"{header}-{i+1}",
        method = "POST",
        url = "/v1/chat/completions",
        body = dict(
            model = "gpt-4o",
            messages = [
                {"role": "user", "content": input}
            ],
            max_tokens = 100,
            n=10,
        )
    ) for i, input in enumerate(inputs)]
    
    os.makedirs("data/temp", exist_ok=True)
    save_path = "data/temp/gpt4o_{}.jsonl"

    for i in range(0, len(prompts), 50000):
        save_path_chunk = save_path.format(i)
        with open(save_path_chunk, 'w') as outfile:
            for entry in prompts[i:i+50000]:
                json.dump(entry, outfile)
                outfile.write('\n')

        batch_input_file = client.files.create(
            file=open(save_path_chunk, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": header
            }
        )
    temp_dir = "data/temp"
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            os.remove(os.path.join(root, file))    
    os.rmdir(temp_dir)