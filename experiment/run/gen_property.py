import os
import pandas as pd
import ast
import json

from experiment.prompts import task_instructions
from experiment.prompts import baseline_instruction
from experiment.utils import get_backend, argparser, get_previous_concepts
from experiment.spread_activation import spread_activation

args = argparser()

add_instruction = dict(
    emergent_naive = """Instructions:
1. You are given a combination of concepts. \
Your task is to generate emergent property of a combination.
2. Find a property that does not belong to any of the individual \
component in the combination but emerges when the words are combined.
3. Use the previous examples to learn how to do this.
4. Answer in dictionary format: {{"property": "{{generated_property}}"}}. \
Do not include other formatting.""",
    emergent_cot = """Instructions:
1. You are given a combination of concepts. \
Your task is to generate emergent property of a combination.
2. Find a property that does not belong to any of the individual \
component in the combination but emerges when the words are combined.
3. Use the previous examples to learn how to do this.
4. Come up with your reasoning process before giving your final answer.
5. Answer in dictionary format: {{"property": "{{generated_property}}"}}. \
Do not include other formatting.""",
    emergent_sa = """Instructions:
1. You are given a combination of concepts and a set of relevant concepts to solve a task. \
Your task is to generate emergent property of a combination.
2. Find a property that does not belong to any of the individual \
component in the combination but emerges when the words are combined.
3. Come up with your reasoning process before giving your final answer.
4. Final answer should follow dictionary format: {{"property": "{{generated_property}}"}}. \
Do not include other formatting.""",
    emergent_examples = """
    
<Example 1>
- Combination: Brown apple
- Correct answer: {example_1}

Above answer is correct because property "unappetizing" does not belong to either "brown" and "apple", but belong to brown apple

<Example 2>
- Combination: burned banknote
- Wrong answer: {example_2}

Above answer is wrong because "burned" something can directly belong to property "useless". Ensure that the emergent property you generate does not directly describe any of the individual words but is a characteristic of the combination as a whole.""",
    emergent_examples_sa = """
    
<Example 1>
- Combination: Brown apple
- Relevant concepts: ['fruit', 'apple', 'core', 'cider']
- Correct answer: {example_1}

Above answer is correct because property "unappetizing" does not belong to either "brown" and "apple", but belong to brown apple

<Example 2>
- Combination: burned banknote
- Relevant concepts: ['paper', 'ash', 'money', 'value']
- Wrong answer: {example_2}

Above answer is wrong because "burned" something can directly belong to property "useless". Ensure that the emergent property you generate does not directly describe any of the individual words but is a characteristic of the combination as a whole.""",
    canceled_naive = """Instructions:
1. You are given a combination of concepts. \
Your task is to generate canceled property of a combination.
2. Find a property that belongs to one of the individual \
components in the combination but doesn't belong to combination as a whole.
3. Use the previous examples to learn how to do this.
4. Answer in dictionary format: {{"property": "{{generated_property}}"}}. \
Do not include other formatting.""",
    canceled_cot = """Instructions:
1. You are given a combination of concepts. \
Your task is to generate canceled property of a combination.
2. Find a property that belongs to one of the individual \
components in the combination but doesn't belong to combination as a whole.
3. Use the previous examples to learn how to do this.
4. Answer in dictionary format: {{"property": "{{generated_property}}"}}. \
Do not include other formatting.""",
    canceled_sa = """Instructions:
1. You are given a combination of concepts and a set of relevant concepts to solve a task. \
Your task is to generate canceled property of a combination.
2. Find a property that belongs to one of the individual \
components in the combination but doesn't belong to combination as a whole.
3. Come up with your reasoning process before giving your final answer.
4. Final answer should follow dictionary format: {{"property": "{{generated_property}}"}}. \
Do not include other formatting.""",
    canceled_examples_sa = """
    
<Example 1>
- Combination: Brown apple
- Relevant concepts: ['fruit', 'apple', 'core', 'cider', 'withered']
- Correct answer: {example_1}

Above answer is correct because the property "appetizing" applies to the word "apple" individually but does not apply to the combination "brown apple" as a whole.

<Example 2>
- Combination: Peeled apple
- Relevant concepts: ['apple', 'round', 'skin', 'knife']
- Wrong answer: {example_2}

Above answer is wrong because the property "round" applies to both "apple" and "peeled apple".""",
    canceled_examples = """
    
<Example 1>
- Combination: Brown apple
- Correct answer: {example_1}

Above answer is correct because the property "appetizing" applies to the word "apple" individually but does not apply to the combination "brown apple" as a whole.

<Example 2>
- Combination: Peeled apple
- Wrong answer: {example_2}

Above answer is wrong because the property "round" applies to both "apple" and "peeled apple".""")

if __name__ == "__main__":
    ## (model_name, method, backend)
    task_type = args.property_type

    baselines = [
        ## naive prompting
        # ("gpt-4o", "naive", "openai"),
        # ("meta-llama/Meta-Llama-3.1-70B-Instruct", "naive", "transformers"),
        # ("Qwen/Qwen2.5-72B-Instruct", "naive", "transformers"),
        # ("claude-3-5-sonnet-20241022", "naive", "anthropic"),
        # ("o1-mini-2024-09-12", "naive", "openai_o1"),
        # ("o1-2024-12-17", "naive", "openai_o1"),
        ## CoT prompting
        # ("gpt-4o", "cot", "openai"),
        # ("meta-llama/Meta-Llama-3.1-70B-Instruct", "cot", "transformers"),
        # ("Qwen/Qwen2.5-72B-Instruct", "cot", "transformers"),
        ## SA prompting with llm only
        ("gpt-4o", "sa-llm", "openai"),
        # ("meta-llama/Meta-Llama-3.1-70B-Instruct", "sa-llm", "transformers"),
        # ("Qwen/Qwen2.5-72B-Instruct", "sa-llm", "transformers"),
        # ("ollama/qwen2.5:72b", "sa-llm", "ollama"),
        ## SA prompting with ConceptNet
        # ("gpt-4o", "sa-conceptnet", "openai"),
        # ("gpt-4o", "sa-nf-conceptnet", "openai"),
        # ("meta-llama/Meta-Llama-3.1-70B-Instruct", "sa-conceptnet", "transformers"),
        # ("Qwen/Qwen2.5-72B-Instruct", "sa-conceptnet", "transformers"),
        # ("ollama/qwen2.5:72b", "sa-conceptnet", "ollama"),
        ## SA prompting with hybrid
        # ("gpt-4o", "sa-llm&conceptnet", "openai"),
        # ("meta-llama/Meta-Llama-3.1-70B-Instruct", "sa-llm&conceptnet", "transformers"),
        # ("Qwen/Qwen2.5-72B-Instruct", "sa-llm&conceptnet", "transformers"),
        # ("ollama/qwen2.5:72b", "sa-llm&conceptnet", "ollama"),
        ## Multi for upper bound
        # ("gpt-4o", "multi", "openai"),
        # ("meta-llama/Meta-Llama-3.1-70B-Instruct", "multi", "transformers"),
        # ("Qwen/Qwen2.5-72B-Instruct", "multi", "transformers"),
        # ("claude-3-5-sonnet-20241022", "multi", "anthropic"),
        # ("o1-mini-2024-09-12", "multi", "openai_o1"),
        # ("o1-2024-12-17", "multi", "openai_o1"),
    ]
    #####################################################

    if task_type == "emergent":
        from experiment.utils import EME_DATA_PATH as DATA_PATH
    else:
        from experiment.utils import CAN_DATA_PATH as DATA_PATH

    assert task_type in ["emergent", "canceled"], "task_type should be either emergent or canceled"   
    
    prev_model_name = None
    for triplet in baselines:
        model_name, method, backend = triplet

        save_path = f"results/pi_{task_type}_{model_name.split('/')[-1]}_{method}.csv"
        print(save_path)

        if task_type in ["emergent", "canceled"]:
            df = pd.read_csv(DATA_PATH)
            # df = df[df['meta.human_labels'].apply(lambda x: ('other' not in x) and ('component' not in x))]
            test_index = json.load(open("test_index.json"))[f'pi_{task_type}']
            df = df[df.index.isin(test_index)]

        else:
            raise Exception("Wrong Path")

        if model_name != prev_model_name:
            MODEL = get_backend(backend, args)
            model = MODEL(model_id=model_name)

        prev_model_name = model_name
        prompt = baseline_instruction
        if 'sa' in method:
            from experiment.prompts import input_format_pi_sa as input_format
            
            # df = pd.DataFrame([
            #     {"root": "apple", "modifier": "brown"},
            #     {"root": "apple", "modifier": "peeled"},
            # ])
            
            if task_type == "emergent":
                example_1 = """The goal is to find an emergent property of "brown apple" that does not exist in "brown" or "apple" individually. \
"Fruit" and "apple" describe general attributes, so they are excluded. \
"Core" and "Cider" are unrelated to "brown apple." \
"Withered" relates to a decayed state, which applies to "brown apple" but not to "brown" or "apple" alone. \
To interpret "withered" naturally, we select "unappetizing" as the emergent property. \
So the answer is {{"property": "unappetizing"}}"""
                example_2 = """The goal is to find an emergent property of a “burned banknote” that does not exist in either “burned” or “banknote” individually. \
The attributes “paper” and “money” describe general properties of a banknote, so they are excluded.\
Likewise, “ash” describes a general property of something that is burned, so it is also excluded. \
While a banknote possesses “value,” this characteristic disappears once the banknote is burned. \
Consequently, to convey the idea of “valueless” naturally, we choose “useless” as the emergent property. \
So the answer is is: {{"property": "useless"}}"""
            else:
                example_1 = """The goal is to find an canceled property of "brown apple" that exists in "brown" or "apple" but not in "brown apple." \
"Fruit" and "apple" describe general attributes, so they are excluded. \
"Core" is unrelated to "apple." \
"Cider" is produced from apples. But it is hard to say that "cider" is a property of "apple." \
"Withered" relates to a decayed state, which applies to "brown apple" but not to "brown" or "apple" alone. \
To find canceled property with "withered", we select antonym of "withered" as "appetizing" as the canceled property. \
So the answer is {{"property": "appetizing"}}"""
                example_2 = """The goal is to find an canceled property of "peeled apple" that exists in "peeled" or "apple" but not in "peeled apple." \
['apple', 'round', 'skin', 'knife']
"Apple" describes general attributes, so they are excluded. \
"Round" is attribute of "apple" but not of "peeled apple." \
"Skin" describes general attributes of "peeled", so it is excluded. \
"Knife" make "peeled apple" but it is hard to say that "knife" is a property of "peeled apple." \
So the answer is {{"property": "round"}}"""

            prompt = prompt.format(
                format="{{\"property\": \"{{generated_property}}\"}",
                add_instruction=add_instruction[f"{task_type}_sa"] + add_instruction[f"{task_type}_examples_sa"].format(
                    example_1=example_1,
                    example_2=example_2
                )
            )
            import pdb; pdb.set_trace()
            C_0 = df[['root', 'modifier']].values.tolist()
            O = df.apply(lambda x:
                f"'{x['root']}', '{x['modifier']}'", 
                axis=1
            ).values.tolist()
            T = 5
            no_filter = True if 'nf' in method else False
            C = spread_activation(model, C_0, O, method=method.split("-")[-1], T=T, no_filter=no_filter)
            
            df['C'] = [{key: C[key][i] for key in C} for i in range(df.shape[0])]
            df['C'] = df['C'].apply(get_previous_concepts)
            df['C_T'] = df['C'].apply(lambda x: list(set(x[T]) - set(x[0])))
        else:
            from experiment.prompts import input_format_pi as input_format
            if 'cot' in method:
                if task_type == "emergent":
                    example_1 = """Let's think step-by-step. "Brown" signifies a color change due to aging, oxidation, or decay, indicating the apple is no longer fresh. "Apple" is a fruit that is typically appealing when fresh, but browning suggests overripeness or spoilage. Concepts like spectrum (color change) and growth (life cycle of the fruit) reinforce this transition. So the answer is {{"property": "unappetizing"}}"""
                    example_2 = """Let's think step-by-step. Individually, "burned" does not mean something is useless. A "banknote" by itself is valuable and serves as a medium of exchange. However, when combined, a "burned banknote" implies that the note is damaged beyond recognition, making it invalid for transactions and effectively useless as currency. So the answer is  {{"property": "useless"}}"""
                else:
                    example_1 = """Let's think step-by-step. "Burned" indicates exposure to fire, leading to combustion and irreversible damage. "Banknote" represents currency, which loses its value and function when burned. Concepts like fire, combustion, and oxidation highlight the destructive nature of burning. So the answer is {{"property": "appetizing"}}"""
                    example_2 = """Let's think step-by-step. A shark is typically carnivorous, as it preys on other marine animals. However, a baby shark is often dependent on its mother or has not yet developed the ability to hunt effectively. So the answer is {{"property": "carnivorous"}}"""
            else:
                example_1 = """{{"property": "unappetizing"}}""" if task_type == "emergent" else """{{"property": "appetizing"}}"""
                example_2 = """{{"property": "useless"}}""" if task_type == "emergent" else """{{"property": "round"}}"""
            
            prompt = prompt.format(
                format="{{\"property\": \"{{generated_property}}\"}",
                add_instruction=add_instruction[f"{task_type}_{'cot' if method == 'cot' else 'naive'}"] + add_instruction[f"{task_type}_examples"].format(
                    example_1=example_1,
                    example_2=example_2
                )
            )
        
        
        model_inputs = []
        
        for i, row in df.iterrows():
            model_inputs.append(prompt.format(
                input_format=input_format.format(**row),                
            ))

        if 'multi' in method:
            try:
                N = int(method.split("-")[-1])
            except:
                N = 5
            output = model.generate(model_inputs, system_prompt=task_instructions, num_return_sequences=3*N)['responses']
        else:
            output = model.generate(model_inputs, system_prompt=task_instructions, num_return_sequences=3)['responses']
        
        total_length = int(len(output[0]) / 3)
        model_name = model_name.replace("_", "-")

        df[f"{model_name.split('/')[-1]}_{method}_0_generated_"] = [o[:total_length] for o in output]
        df[f"{model_name.split('/')[-1]}_{method}_1_generated_"] = [o[total_length:2*total_length] for o in output]
        df[f"{model_name.split('/')[-1]}_{method}_2_generated_"] = [o[2*total_length:3*total_length] for o in output]

        df.to_csv(save_path, index=False)