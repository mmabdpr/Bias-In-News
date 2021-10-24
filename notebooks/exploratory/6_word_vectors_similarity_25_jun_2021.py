# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

# %%
nlp = spacy.load("en_core_web_lg")

# %%
project_dir = Path(__file__).resolve().parents[2]
data_dir = os.path.join(project_dir, "data")
tweets = pd.read_json(os.path.join(data_dir, 'processed', 'tweets.json'), orient='records')

# %%
classes = tweets['bias'].unique().tolist()
similarities = dict()


def sim_key(x, y):
    return f'{min(x, y)} <> {max(x, y)}'


for c1 in classes:
    for c2 in classes:
        similarities[sim_key(c1, c2)] = (0, 0)  # count, sum

for i in range(10):
    for c1 in classes:
        for c2 in classes:
            t1 = tweets[tweets['bias'] == c1].sample(frac=0.01)
            t2 = tweets[tweets['bias'] == c2].sample(frac=0.01)

            t1_txt = ' '.join(t1['tweet'].tolist())
            t2_txt = ' '.join(t2['tweet'].tolist())

            doc1 = nlp(t1_txt, disable=["parser", "tagger", "ner", "lemmatizer"])
            doc2 = nlp(t2_txt, disable=["parser", "tagger", "ner", "lemmatizer"])

            s = doc1.similarity(doc2)

            similarities[sim_key(c1, c2)] = similarities[sim_key(c1, c2)][0] + 1, similarities[sim_key(c1, c2)][1] + s

sim_avg = dict()

for c1 in classes:
    for c2 in classes:
        sim_avg[sim_key(c1, c2)] = similarities[sim_key(c1, c2)][1] / similarities[sim_key(c1, c2)][0]

print(sim_avg)

# %%
for c1 in classes:
    for c2 in classes:
        print(f'{sim_key(c1, c2)}: {np.exp(sim_avg[sim_key(c1, c2)])}')

# %%
# [
#     {
#         "-1 <> -1": 0.9999261892681286,
#         "-1 <> 2": 0.996287149064243,
#         "-2 <> -1": 0.9969182508004899,
#         "-1 <> 1": 0.9992839887869283,
#         "-1 <> 0": 0.9989565965731311,
#         "2 <> 2": 0.99986649940142,
#         "-2 <> 2": 0.9986836804977568,
#         "1 <> 2": 0.9968332407384588,
#         "0 <> 2": 0.9950788329200891,
#         "-2 <> -2": 0.9998810564076527,
#         "-2 <> 1": 0.9965497498016613,
#         "-2 <> 0": 0.9952623289985569,
#         "1 <> 1": 0.9998487167451451,
#         "0 <> 1": 0.9988039108848452,
#         "0 <> 0": 0.99987976609093
#     },
#     {
#         "-2 <> -2": 2.717958525481146,
#         "-1 <> -1": 2.7180811974923045,
#         "0 <> 0": 2.7179550184560712,
#         "1 <> 1": 2.717870629041,
#         "2 <> 2": 2.7179189604299254,
#
#         "-2 <> -1": 2.70991766036398,
#         "-2 <> 0": 2.705433961980247,
#         "-2 <> 1": 2.7089192369706994,
#         "-2 <> 2": 2.7147060550221176,
#         "-1 <> 0": 2.7154470430536226,
#         "-1 <> 1": 2.7163362048168738,
#         "-1 <> 2": 2.708207966175669,
#         "0 <> 1": 2.715032464803738,
#         "0 <> 2": 2.704937571001962,
#         "1 <> 2": 2.709687299887191
#     }
# ]
