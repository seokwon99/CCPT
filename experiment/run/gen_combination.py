import os
import pandas as pd
import json
import ast
import copy

from experiment.prompts import task_instructions
from experiment.prompts import baseline_instruction
from experiment.utils import get_backend, argparser, get_previous_concepts
from experiment.utils import EME_DATA_PATH as DATA_PATH
from experiment.spread_activation import spread_activation

args = argparser()

add_instruction = dict(
    emergent_naive = """Instructions:
1. You are given a head noun and emergent property. \
Your task is to generate a conceptual combination by adding one modifier.
2. You can use function word without any constraint.
3. Modifier should not have the given emergent property on its own, \
but the combination exhibits the emergent property.
4. Use the previous examples to learn the task.
5. Answer in dictionary format: \
{{"combination": "{{generated_combination}}", "modifier": "{{generated_modifier}}"}}. \
Do not include other formatting.""",
    emergent_cot = """Instructions:
1. You are given a head noun and emergent property. \
Your task is to generate a conceptual combination by adding one modifier.
2. You can use function word without any constraint.
3. Modifier should not have the given emergent property on its own, \
but the combination exhibits the emergent property.
4. Come up with your reasoning process before giving your final answer.
5. Use the previous examples to learn the task.
6. Answer in dictionary format: \
{{"combination": "{{generated_combination}}", "modifier": "{{generated_modifier}}"}}. \
Do not include other formatting.""",
    emergent_sa = """Instructions:
1. You are given a head noun, emergent property and a set of relevant concepts to solve a task. \
Your task is to generate a conceptual combination by adding one modifier.
2. You can use function word without any constraint.
3. Modifier should not have the given emergent property on its own, \
but the combination exhibits the emergent property.
4. Come up with your reasoning process before giving your final answer.
5. Final answer should follow dictionary format: \
{{"combination": "{{generated_combination}}", "modifier": "{{generated_modifier}}"}}. \
Do not use other formatting.""",
    examples = """
    
<Example 1>
- Head noun: apple
- Emergent property: unappetizing
- Correct answer: {example_1}

Above answer is correct because each component "brown" and "apple" do not possess "unappetizing" but "brown apple" does.

<Example 2>
- Head noun: banknote
- Emergent property: useless
- Wrong answer: {example_2}

Above answer is wrong because modifier "burned" directly elicit property "useless". Avoid modifier which has given property in itself.
""",
    examples_sa = """
    
<Example 1>
- Head noun: apple
- Emergent property: unappetizing
- Relevant concepts: ['bland', 'bitter', 'inedible', 'insipid', 'unappealing', 'unpalatable']
- Correct answer: {example_1}

Above answer is correct because each component "brown" and "apple" do not possess "unappetizing" but "brown apple" does.

<Example 2>
- Head noun: banknote
- Emergent property: useless
- Relevant concepts: ['counterfeit', 'worthless', 'ineffective', 'futile', 'meaningless', 'pointless']
- Wrong answer: {example_2}

Above answer is wrong because modifier "burned" directly elicit property "useless".  Avoid modifier which elicit given property in itself.
"""
)


if __name__ == "__main__":

    ################### Task settings ###################
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
        # ("gpt-4o", "sa-llm", "openai"),
        # ("meta-llama/Meta-Llama-3.1-70B-Instruct", "sa-llm", "transformers"),
        # ("Qwen/Qwen2.5-72B-Instruct", "sa-llm", "transformers"),
        # ("ollama/qwen2.5:72b", "sa-llm", "ollama"),
        ## SA prompting with ConceptNet
        ("gpt-4o", "sa-conceptnet", "openai"),
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

    assert task_type == "emergent", "Only emergent task is supported"
    
    prev_model_name = None
    for triplet in baselines:
        model_name, method, backend = triplet
        
        save_path = f"results/npc_{task_type}_{model_name.split('/')[-1]}_{method}.csv"
        print(save_path)

        if task_type == "emergent":
            df = pd.read_csv(DATA_PATH)
            # df = df[df['meta.human_labels'].apply(lambda x: ('other' not in x) and ('component' not in x))]
            test_index = json.load(open("test_index.json"))[f'npc_{task_type}']
            df = df[df.index.isin(test_index)]
        else:
            raise Exception("Wrong Path")
    
        if model_name != prev_model_name:
            MODEL = get_backend(backend, args)
            model = MODEL(model_id=model_name)

        prev_model_name = model_name
        prompt = baseline_instruction
        if 'sa' in method:
            from experiment.prompts import input_format_npc_sa as input_format

            # df = pd.DataFrame([
            #     {"root": "apple", "property": "unappetizing"},
            #     {"root": "banknote", "property": "useless"}
            # ])
            
            prompt = prompt.format(
                format="{{\"combination\": \"{{generated_combination}}\", \"modifier\": \"{{generated_modifier}}\"}}",
                add_instruction=add_instruction[f"{task_type}_sa"] + add_instruction["examples_sa"].format(
                    example_1="""The goal is to find a modifier that does not inherently have the emergent property "unappetizing," \
but do when combined with "apple". \
Related concepts such as bitter, inedible or unpalatable make apple unappetizing. \
To represent bitter apple, "yellow" can be used as a modifier. But yellow is somewhat related to bitter because of the color of lemons. \
To represent inedible or unpalatable apple, "plastic" or "brown" can be used as a modifier. However plastic is directly related to inedible. \
"Brown" as a modifier doesn't imply inedible on its own, but when paired with "apple," it suggest an inedible state. \
So the answer is {{"combination": "brown apple", "modifier": "brown"}}""",
                    example_2="""The goal is to find a modifier that does not inherently have the emergent property "useless," \
but do when combined with "banknote". \
Related concepts such as counterfeit or worthless make banknote useless. \
To represent counterfeit banknote, "fake" can be used as a modifier. But fake is somewhat related to useless because of the meaning. \
To represent worthless banknote, "burned" can be used as a modifier. \
So the answer is {{"combination": "burned banknote", "modifier": "burned"}}"""
                )
            )

            C_0 = df[['root', 'property']].values.tolist()
            
            O = df.apply(lambda x:
                f"\"{x['property']}\", \"{x['root']}\"", 
                axis=1
            ).values.tolist()
            T = 5
            no_filter = True if 'nf' in method else False
            C = spread_activation(model, C_0, O, method=method.split("-")[-1], T=T, no_filter=no_filter)

            df['C'] = [{key: C[key][i] for key in C} for i in range(df.shape[0])]
            df['C'] = df['C'].apply(get_previous_concepts)
            df['C_T'] = df['C'].apply(lambda x: list(set(x[T]) - set(x[0])))
        else:
            from experiment.prompts import input_format_npc as input_format
            
            prompt = prompt.format(
                format="{{\"combination\": \"{{generated_combination}}\", \"modifier\": \"{{generated_modifier}}\"}}",
                add_instruction=add_instruction[f"{task_type}_{'cot' if method == 'cot' else 'naive'}"] + add_instruction["examples"].format(
                    example_1="""Let's think step-by-step. A typical apple is fresh and appetizing, but certain modifications can make it unappetizing. Factors like discoloration, decay, or unusual texture can contribute to this perception. A brown apple, for instance, appears spoiled or oxidized, making it less appealing to eat. So the answer is {{"combination": "brown apple", "modifier": "brown"}}""" if method == 'cot' else """{{"combination": "brown apple", "modifier": "brown"}}""",
                    example_2="""Let's think step-by-step. A typical banknote has value and can be used for transactions, but certain modifications can make it useless. Burning a banknote destroys its structure, making it unrecognizable and invalid as currency. So the answer is {{"combination": "burned banknote", "modifier": "burned"}}""" if method == 'cot' else """{{"combination": "burned banknote", "modifier": "burned"}}"""
                )
            )
        model_inputs = []
        
        for i, row in df.iterrows():
            model_inputs.append(prompt.format(
                input_format=input_format.format(**row),
            ))
        
        if 'multi' in method:
            N = int(method.split("-")[-1]) if '-' in method else 5
            output = model.generate(model_inputs, system_prompt=task_instructions, num_return_sequences=3*N)['responses']
        else:
            output = model.generate(model_inputs, system_prompt=task_instructions, num_return_sequences=3)['responses']
            
        total_length = int(len(output[0]) / 3)
        model_name = model_name.replace("_", "-")

        df[f"{model_name.split('/')[-1]}_{method}_0_generated_"] = [o[:total_length] for o in output]
        df[f"{model_name.split('/')[-1]}_{method}_1_generated_"] = [o[total_length:2*total_length] for o in output]
        df[f"{model_name.split('/')[-1]}_{method}_2_generated_"] = [o[2*total_length:3*total_length] for o in output]
        
        df.to_csv(save_path, index=False)