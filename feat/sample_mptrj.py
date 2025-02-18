import json
import random

with open('/home/lucky/Projects/ion_conductivity/feat/data/MPtrj_2022.9_full.json', 'rb') as f:
    mptrj = json.load(f)
# Randomly sample 10 structures
        
keys = [
    (mp_id, graph_id) for mp_id, dct in mptrj.items() for graph_id in dct
]
random.shuffle(keys)
keys = keys[:1580]
# save the sampled data while maintaining the original structure
sampled_mptrj = {}
for mp_id, graph_id in keys:
    if mp_id not in sampled_mptrj:
        sampled_mptrj[mp_id] = {}
    sampled_mptrj[mp_id][graph_id] = mptrj[mp_id][graph_id]

# Save the sampled data
with open('/home/lucky/Projects/ion_conductivity/feat/data/MPtrj_2022.9_full_sampled.json', 'w') as f:
    json.dump(sampled_mptrj, f)