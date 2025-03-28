import pandas as pd
from pfp_call.calculate_diffusivity_v2 import calculate_diffusivities

print("[INFO] Reading JSON file.")
df = pd.read_json('/home/sokim/ion_conductivity/top10.json')

print("[INFO] Calculating diffusivities.")
list_to_label = []
for i, row in df.iterrows():
    list_to_label.append((row['min_hydrostatic_pressure'], row['hydrostatic_pressures']))

result = calculate_diffusivities(list_to_label[:2])
print("[INFO] Diffusivities calculated.")
# save result to a txt file
with open('labeled_diffusivities.txt', 'w') as f:
    for i, r in enumerate(result):
        f.write(f"Index: {i+1}, Diffusivity: {r:.12f} cm^2/sec\n")