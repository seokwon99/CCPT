eval_prompt = """Concepts are characterized by properties. For example, the concept "a chicken in front of a fox" strongly exhibits the property "in danger." When given a concept and a property, your task is to evaluate how much the concept has the property on a scale from 1 to 10. You should follow the format: {{"relevance": your_relevance_score}}

Use the following scoring criteria to assign a relevance score:
- {{"relevance": 1}}: The concept does not have the property at all.
- {{"relevance": 2-3}}: The concept rarely has the property.
- {{"relevance": 4-6}}: The concept sometimes has the property.
- {{"relevance": 7-8}}: The concept usually has the property, but not always.
- {{"relevance": 9}}: The concept almost always has the property.
- {{"relevance": 10}}: The concept always has the property.

---

Examples:

---

Concept: Rusty
Property: Useless
Relevance: {{"relevance": 7}}

---

Concept: A chicken in the cage
Property: In danger
Relevance: {{"relevance": 2}}

---

Concept: A chicken in front of a fox
Property: In danger
Relevance: {{"relevance": 9}}

---

Concept: {concept}
Property: {property}
Relevance:
"""

task_instructions = """Conceptual combination is a task that combines two concepts, which can result in new properties. It involves a head noun, a modifier, and corresponding properties. Here's the definition of each component:

1. Head Noun: The original concept in the conceptual combination.
2. Modifier: The word that you will generate to create a new conceptual combination with the head noun.
3. Component Property: A property inherent to individual concepts. (head noun or modifier)
4. Emergent Property: A new property that arises from the combination of the head noun and the modifier. This property does not exist in either concept individually (head noun or modifier) but emerge in conceptual combination.
5. Canceled Property: A property that is inherent to individual concept (head noun or modifier) and negated due to the combination.
"""


input_format_npc = """- Head noun: {root}
- Emergent property: {property}"""

input_format_npc_sa =  """- Head noun: {root}
- Emergent property: {property}
- Relevant concepts: {C_T}"""

input_format_pi = """- Combination: {combination}"""

input_format_pi_sa = """- Combination: {combination}
- Relevant concepts: {C_T}"""

input_format_tp = """- Combination: {combination}
- Property: {property}"""

baseline_instruction = "{add_instruction}\n\nThen let's begin:\n{{input_format}}\n- Answer:"

fewshot_instruction = ""