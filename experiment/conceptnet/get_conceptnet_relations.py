# from datasets import load_dataset
import json
import requests
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from experiment.utils import retry
from multiprocessing import Pool
tqdm.pandas()

relation_template = {
    "AtLocation": "located or found at/in/on",
    "LocatedNear": "is located near or close to",
    "UsedFor": "used for",
    "DerivedFrom": "originates from or is derived from",
    "PartOf": "is a part of",
    "HasProperty": "can be characterized by being/having",
    "HasA": "has, possesses or contains",
    "CapableOf": "is/are capable of",
    "MadeOf": "is made of",
    "MotivatedByGoal": "is a step towards accomplishing the goal",
    "CausesDesire": "makes someone want",
    "HasPrerequisite": "to do this, one requires",
    "HasSubEvent": "includes the event/action",
    "HasFirstSubevent": "BEGINS with the event/action",
    "HasLastSubevent": "ENDS with the event/action",
    "RelatedTo": "is related to or connected with",
    "IsA": "is a type or example of",
    "DistinctFrom": "is distinct from or different than",
    "Antonym": "is the opposite of",
    "HasContext": "appears or occurs in the context of",
    "Causes": "causes",
    "CreatedBy": "is created by",
    "Desires": "desires",
    "HinderedBy": "can be hindered by",
    "InstanceOf": "is an example/instance of",
    "isAfter": "happens after",
    "isBefore": "happens before",
    "isFilledBy": "blank can be filled by",
    "HasA": "made (up) of",
    "NotDesires": "do(es) NOT desire",
    "ReceivesAction": "can receive or be affected by the action",
    "MotivatedByGoal": "because",
    "FormOf": "is a form of",
    "ObstructedBy": "is obstructed or blocked by",
    "Synonym": "is a synonym of",
    "SymbolOf": "is a symbol of or represents",
    "DefinedAs": "is defined as",
    "MannerOf": "is a manner or way of",
    "SimilarTo": "is similar to"
}

# @retry(3, output=0)
# def get_edge_weight(concept1, concept2, rel):
#     url = f"https://api.conceptnet.io/query?node=/c/en/{concept1}&other=/c/en/{concept2}&rel={rel}"

#     query_params = {
#         "limit": 10,
#     }
#     response = requests.get(url=url, params=query_params)
#     response = response.json()
#     edges = response.get('edges', [])
#     if not edges:
#         return None
#     edge_weights = [edge['weight'] for edge in edges if edge['start']['label'] == concept1 and edge['end']['label'] == concept2]
#     if not edge_weights:
#         return None
#     return max(edge_weights)

def get_raw_relations(concept):
    url = f"https://api.conceptnet.io/query?node=/c/en/{concept}"
    
    # url = f"http://api.conceptnet.io/c/en/{concept}" 
    response = requests.get(url=url)
    response = response.json()
    response = pd.DataFrame(response['edges'])
    if response.shape[0] == 0:
        return None
    response = response.sort_values(by=['weight'], ascending=False)
    response = response[response.apply(lambda x: 'language' in x['start'] and x['start']['language'] == "en" and 'language' in x['end'] and x['end']['language'] == "en", axis=1)]        
    response['start'] = response['start'].apply(lambda x: x['@id'].split("/")[-1].replace("_", " "))
    response['end'] = response['end'].apply(lambda x: x['@id'].split("/")[-1].replace("_", " "))
    response['rel'] = response['rel'].apply(lambda x: x['@id'].split("/")[-1].replace("_", " "))
    response = response[response.apply(lambda x: (x['start'] == concept) or (x['end'] == concept), axis=1)]

    return response

# @retry(3, output=None)
def get_relations(concept):
    concept = concept.replace(" ", "_")
    response = get_raw_relations(concept)
    if response is None:
        return []
    new_concept = []
    
    syn = list(set([c for c in new_concept if c != concept]))
    for c in syn:
        add = get_raw_relations(c)
        if add is not None:
            response = pd.concat([response, add])
    
    syn.append(concept)
    if response.shape[0] == 0:
        return []
    response['counter'] = response.apply(lambda x: x['end'] if x['start'] == concept else x['start'], axis=1)
    response = response.groupby("counter").apply(lambda x: x.iloc[0])
    response = response.sort_values(by=['weight'], ascending=False)
    
    response['natural_texts'] = response.apply(lambda x: x['start'] + " " + relation_template[x['rel']] + " " + x['end'] if x['rel'] in relation_template else "", axis=1)
    response = response[response['natural_texts'] != ""]
    response = response[response['counter'].apply(lambda x: len(x) > 2)]
    if response.shape[0] == 0:
        return []
    return response[['natural_texts', 'counter', 'weight']].to_dict("records")

if __name__ == "__main__":
    df = pd.read_csv("experiment/conceptnet/conceptnet5.csv")
    def apply_get_edge_weight(row):
        return get_edge_weight(row['concept1'], row['concept2'], row['rel'])

    with Pool(processes=Pool()._processes) as pool:
        df['weight'] = list(tqdm(pool.imap(apply_get_edge_weight, [row for _, row in df.iterrows()]), total=len(df)))
    df.to_csv("experiment/conceptnet/conceptnet5.csv", index=False)