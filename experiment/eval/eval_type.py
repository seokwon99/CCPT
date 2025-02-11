import pandas as pd
from sklearn.metrics import confusion_matrix
import json
import ast
import re
import numpy as np

result = pd.read_csv("results/tp_gpt-4o_naive.csv")

types = [
    "emergent",
    "component",
    "canceled",
    "others"
]

# >> result.iloc[0]
# combination                              a bucket of piss
# property                                          useless
# human_label_majority                             emergent
# gpt-4o_generated_       ['{"property_type": "emergent"}']

# Extract gold labels and predictions
gold_labels = result['human_label_majority']
predictions = result['gpt-4o_generated_'].apply(lambda x: re.findall(r'\{.*?\}', x)[0])
predictions = predictions.apply(lambda x: ast.literal_eval(x)['property_type'])

conf_matrix = confusion_matrix(gold_labels, predictions, labels=types)
conf_matrix = 100 * conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

print(conf_matrix)