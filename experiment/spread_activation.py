import ast
import re
import copy
import spacy
nlp = spacy.load("en_core_web_sm")
from tqdm import tqdm
import random
from experiment.utils import retry
from experiment.conceptnet.get_conceptnet_relations import get_relations
from os.path import isfile
import json

activated_concepts = dict()

# Load saved data
conceptnet_path = "experiment/conceptnet/conceptnet_adjacent.json"
conceptnet_data = {}
if isfile(conceptnet_path):
    conceptnet_data = json.load(open(conceptnet_path))
    for key in conceptnet_data:
        if key not in activated_concepts:
            activated_concepts[key] = dict()
        activated_concepts[key]['conceptnet'] = conceptnet_data[key]

# @retry(3)
def activate(model, concepts, method):
    concepts = [c for c in concepts if not ((c in activated_concepts) and method in activated_concepts[c])]

    if method == "llm":
        prompt = """Instruction:
1. Your task is to find relevant concepts related to a given concept
2. Explore adjacent concepts from various and novel perspectives on given concept.
3. Use the previous examples to learn how to do this.
4. Answer in list format: ["{{adjacent_concept_1}}", "{{adjacent_concept_2}}", ...]. \
Do not include other formatting.

<Example 1>
Concept: Apple
Answer: ["fruit", "stem", "red", "round", "tree", "adam"]

<Example 2>
Concept: Castle
Answer: ["building", "house", "king", "moat", "fortress"]

Your turn:
Concept: {c}
Answer: """

        model_inputs = [prompt.format(c=c) for c in concepts]
        output = model.generate(model_inputs)['responses']
        pattern = r'\[[^\]]*\]'
        A_c = [re.findall(pattern, o[0])[0] for o in output]
        A_c = [ast.literal_eval(a) for a in A_c]
    elif method == "conceptnet":
        A_c = []
        for c in tqdm(concepts):
            results = get_relations(c)
            if results != None:
                results = [result['counter'] for result in get_relations(c)]
                # Save the data
                conceptnet_data[c] = results
                A_c.append(results)
                with open(conceptnet_path, "w") as f:
                    json.dump(conceptnet_data, f, indent=4, ensure_ascii=False)
            
        
    else:
        raise ValueError("Invalid method")

    for c, adj in zip(concepts, A_c):
        if c not in activated_concepts:
            activated_concepts[c] = dict()
        if adj != None:
            activated_concepts[c][method] = adj


@retry(3)
def filter(model, C_t_, O):
    prompt = """Instruction:
1. Identify the relevant nodes related to both given concepts.
2. Remove nodes that are redundant to another node or related to only one of the concepts.
3. Improve accuracy by learning from previous examples.
4. Come up with your reasoning process before giving your final answer.
4. Final answer should follow dictionary format: \
["{{node_1}}", "{{node_2}}", ...]. Do not use other formatting.

<Example 1>
Node set: ["crispy", "stem", "red", "crunchy"]
Concepts: "peeled", "apple"
Answer: Let's consider each concept. The crispy is the texture of apple, but it is not directly related to whether apple is peeled or not. \
Stem is the part of apple and removed when apple is peeled, related to either concepts. \
Apple is red and the color of apple is affected by whether it is peeled or not, related to either concepts. \
Crunchy is similar to crispy, which is a redundant node. \
So the answer is ["stem", "red"]

<Example 2>
Node set: ["house", "king", "royal", "abstract"]
Concepts: "imaginary", "castle"
Answer: Let's consider each concept. \
The castle is a house of king or royal, but the house is not related to 'imaginary'. \
The castle is related to king and king may be imaginary because of power. \
Royal is a similar concept to king, which is a redundant node. \
Abstract is related to imaginary, but not related to the castle. \    
So the answer is ["king", "novel"]

Your turn:
Node set: {C_t}
Concepts: {O}
Answer: """
    max_length = 5
    # model_inputs = [prompt.format(C_t=C_t, O=O[i]) for i, C_t in enumerate(C_t_)]
    model_inputs = [prompt.format(C_t=random.sample(C_t[:max_length], min(len(C_t), max_length)), O=O[i]) for i, C_t in enumerate(C_t_)]
    output = model.generate(model_inputs)['responses']
    pattern = r'\[[^\]]*\]'
    C_t_ = [re.findall(pattern, o[0])[0] if re.findall(pattern, o[0]) else "[]" for o in output]
    C_t_ = [ast.literal_eval(C_t) for C_t in C_t_]
    return C_t_

def difference_between_set(C_t, C_t1):
    try:
        C_t = [" ".join([token.lemma_ for token in nlp(c)]) for c in C_t]
        C_t = [" ".join([token.lemma_ for token in nlp(c)]) for c in C_t1]

        intersect_size = len(set(C_t).intersection(C_t1))
        union_size = len(set(C_t).union(C_t1))
        
        return 1 - intersect_size/union_size
    except:
        return 0

def spread_activation(model, C_0, O, method="conceptnet", T=1, e=0.1, no_filter=False):
    C = dict()
    C[0] = C_0
    stops = set()

    for t in tqdm(range(T)):
        
        # Do not repeat the instances in stops
        C_t_ = copy.deepcopy(C[t])
        C_t_ = [C_t for i, C_t in enumerate(C_t_) if i not in stops]
        C_t1_ = copy.deepcopy(C_t_)

        if "llm" in method:
            activate(model, [c for C_t in C_t_ for c in C_t], "llm")
        if "conceptnet" in method:
            activate(model, [c for C_t in C_t_ for c in C_t], "conceptnet")

        for i, C_t in enumerate(C_t_):
            for c in C_t:
                A_c = activated_concepts[c]
                C_t1_[i] += sum(A_c.values(), [])

        C_t1_ = [random.sample(C_t1, min(len(C_t1), 10)) for C_t1 in C_t1_]
        if not no_filter:
            C_t1_ = filter(model, C_t1_, O)            

        # Update C[t+1]
        C[t+1] = copy.deepcopy(C[t])
        start_idx = 0
        for i, C_t in enumerate(C[t]):
            if i not in stops:
                C[t+1][i] = C_t1_[start_idx]
                start_idx += 1
        C[t+1] = [list(set(C_t1)) for C_t1 in C[t+1]]

        diff = [difference_between_set(C_t, C_t1) for C_t, C_t1 in zip(C[t], C[t+1])]

        for i, d in enumerate(diff):
            if d < e:
                stops.add(i)

    return C