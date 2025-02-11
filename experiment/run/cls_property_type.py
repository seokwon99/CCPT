import pandas as pd
import json

from experiment.prompts import task_instructions
from experiment.prompts import baseline_instruction
from experiment.utils import get_backend
from experiment.utils import EME_DATA_PATH, CAN_DATA_PATH, COM_DATA_PATH

add_instruction = dict(
    naive = """Instructions:
1. You are given a combination and property. \
Your task is to predict a type of property.
2. Definition of each property type is as follows:
    - Emergent: The property emerges from the combination of components.
    - Component: The property is inherited by component of the combination.
    - Canceled: The property is canceled out by the combination of components.
    - Others: The property is not related to the combination nor components.
3. Use the previous examples to learn the task.
4. Answer in dictionary format: \
{{"property_type": "{{property_type}}"}}. \
Do not include other formatting.""",
    examples = """
    
<Example 1>
- Combination: peeled apple
- Property: round
- Correct answer: {example_1}

Above answer is correct because property "round" is inherited by component "apple".

<Example 2>
- Combination: burned banknote
- Property: useless
- Wrong answer: {example_2}

Above answer is wrong because modifier "burned" directly elicit property "useless".
"""
)


if __name__ == "__main__":

    ################### Task settings ###################
    triplet = ("gpt-4o", "naive", "openai")
    #####################################################
    
    prev_model_name = None

    model_name, method, backend = triplet
    
    save_path = f"results/tp_{model_name.split('/')[-1]}_{method}.csv"
    print(save_path)
    
    df = pd.concat([
        pd.read_csv(EME_DATA_PATH)[['combination', 'property', 'human_label_majority']].sample(250),
        pd.read_csv(CAN_DATA_PATH)[['combination', 'property', 'human_label_majority']].sample(250),
        pd.read_csv(COM_DATA_PATH)[['combination', 'property', 'human_label_majority']].sample(250)
    ])
    
    # Generate others property
    others_df = pd.DataFrame(columns=['combination', 'property', 'human_label_majority'])
    others_list = []
    for i in range(250):
        others_list.append({
            'combination': df.combination.sample(1).values[0],
            'property': df.property.sample(1).values[0],
            'human_label_majority': "others"
        })
    
    df = pd.concat([df, pd.concat([others_df, pd.DataFrame(others_list)], ignore_index=True)])

    MODEL = get_backend(backend, None)
    model = MODEL(model_id=model_name)

    prompt = baseline_instruction
    
    from experiment.prompts import input_format_tp as input_format
    
    prompt = prompt.format(
        format="{{\"combination\": \"{{generated_combination}}\", \"modifier\": \"{{generated_modifier}}\"}}",
        add_instruction=add_instruction[f"naive"] + add_instruction["examples"].format(
            example_1="""{{"property_type": "component"}}""",
            example_2="""{{"property_type": "emergent"}}"""
        )
    )
    import pdb; pdb.set_trace()
    model_inputs = []
    
    for i, row in df.iterrows():
        model_inputs.append(prompt.format(
            input_format=input_format.format(**row),
        ))

    output = model.generate(model_inputs, system_prompt=task_instructions, num_return_sequences=1)['responses']
        
    total_length = 1
    model_name = model_name.replace("_", "-")

    df[f"{model_name.split('/')[-1]}_generated_"] = [o[:total_length] for o in output]
    
    df.to_csv(save_path, index=False)