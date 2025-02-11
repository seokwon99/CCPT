import pandas as pd
from tqdm import tqdm
tqdm.pandas()


import ast


# df = pd.read_csv('conceptnet-assertions-5.7.0.csv', header=None, sep='\t')
# df = df[df[0].progress_apply(lambda x: x.count("/en/") == 2)]


# df = df[[0, 4]]
# df[4] = df[4].progress_apply(lambda x: ast.literal_eval(x))
# df['weight'] = df[4].progress_apply(lambda x: x['weight'])
# df["triplet"] = df[0].progress_apply(lambda x: x.split("[")[-1].split("]")[0].split(","))

# import pdb; pdb.set_trace()

df = pd.read_csv("conceptnet-assertions-5.7.0-filtered.csv")

import networkx as nx
G = nx.Graph()
# Add edges to the graph
for index, row in tqdm(df.iterrows()):
    try:        
        row['triplet'] = ast.literal_eval(row['triplet'])
        
        relation = row['triplet'][0].split("/")[2]
        concept1 = row['triplet'][1].split("/")[3]
        concept2 = row['triplet'][2].split("/")[3]
        
        weight = 1/row['weight'] if row['weight'] > 0 else 100
        # You can include the relation as edge attributes
        G.add_edge(concept1, concept2, relation=relation, weight=weight)
    except:
        pass

# nx.write_adjlist(G, "results/ConceptNetGraph.adjlist.gz")


import networkx as nx
import numpy as np
# G = nx.read_adjlist("results/ConceptNetGraph.adjlist.gz")

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

import re
import time
import pandas as pd

def clean_text(text):
    # 특수문자를 제거하고 소문자로 변환
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return cleaned_text

db = pd.read_csv("conceptnet-assertions-5.7.0-filtered.csv")
def get_weight(c1, c2):
    rdf = db[(db['c1'] == c1) | (db['c2'] == c1)]
    rdf = rdf[(rdf['c1'] == c2) | (rdf['c2'] == c2)]
    if rdf.shape[0] == 0:
        return 10000
    return rdf['weight'].max()
    
import requests
def get_raw_relations(concept1, concept2):
    try:
        url = f"https://api.conceptnet.io/query?node=/c/en/{concept1}&other=/c/en/{concept2}"
        # url = f"http://api.conceptnet.io/c/en/{concept}"   
        query_params = {
            "limit": 100,
        }
        response = requests.get(url=url, params=query_params)
        response = response.json()
        response = pd.DataFrame(response['edges'])
        if response.shape[0] == 0:
            return None
        response = response.sort_values(by=['weight'], ascending=False)
        response = response[response.apply(lambda x: 'language' in x['start'] and x['start']['language'] == "en" and 'language' in x['end'] and x['end']['language'] == "en", axis=1)]        
        response['start'] = response['start'].apply(lambda x: x['@id'].split("/")[3].replace("_", " "))
        response['end'] = response['end'].apply(lambda x: x['@id'].split("/")[3].replace("_", " "))
        response['rel'] = response['rel'].apply(lambda x: x['@id'].split("/")[-1].replace("_", " "))
        return response
    except:
        time.sleep(5)
        print("retry")
        return get_raw_relations(concept1, concept2)

def get_relations(concept1, concept2):
    concept2 = clean_text(concept2)
    concept2 = concept2.replace(" ", "_")
    response = get_raw_relations(concept1, concept2)
    if response is None:
        return []
    response['natural_texts'] = response.apply(lambda x: x['start'] + " " + relation_template[x['rel']] + " " + x['end'] if x['rel'] in relation_template else "", axis=1)
    response = response[response['natural_texts'] != ""]
    if response.shape[0] == 0:
        return []
    return response[['natural_texts', 'weight']].to_dict("records")
    
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

topath = dict()
def find_shortest_path(graph, start_concept, end_concept):
    try:
        results = []
        for path in nx.all_simple_paths(graph, source=start_concept, target=end_concept, cutoff=3):
            total = 0
            for u, v in zip(path[:-1], path[1:]):
                sorted_vars = sorted([u, v])
                key = f"{sorted_vars[0]}_{sorted_vars[1]}"
                if key in topath:
                    total += topath[key]
                else:
                    weight = G.edges[u,v]['weight']
                    topath[key] = 1 / weight
                    total += topath[key]
            results.append((" - ".join(path), total))
        return results
    except nx.NetworkXNoPath:
        print(f"No path found between '{start_concept}' and '{end_concept}'.")
        return None
    except nx.NodeNotFound as e:
        print(e)
        return None

df = pd.read_csv("results/tag_result/0916_1007_tagged_task2_canceled_with_relations.csv")
# df['single_word_property'] = df['single_word_property'].apply(clean_text)
df['relations'] = df.progress_apply(lambda x: find_shortest_path(G, x['root'], x['modifier']), axis=1)
df.to_csv("results/tag_result/0916_1007_tagged_task2_canceled_with_relations.csv", index=False)