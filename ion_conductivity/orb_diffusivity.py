import pandas as pd
import os
import subprocess
import time

# start time
start_time = time.time()
df = pd.read_parquet('/home/lucky/Projects/ion_conductivity/data/MPContribs_armorphous_diffusivity.parquet')
df = df[df['data_properties_A_element'] == 'Li']  # only Li-ion conductivity
df = df[df['data_temperature_value']<5000]  # Remove 5000 K data

for i, row in df.iterrows():
    # check if the cif file is written already
    if not os.path.exists(f'orb_call/input/temp_{i}.cif'):
        with open(f'orb_call/input/temp_{i}.cif', 'w') as f:
            f.write(row['cif'])
    # print(run_md_simulation(row['data_temperature_value'], i))

print('CIFs are prepared')

chunk_len = 6  # number of GPUs


# Create individual Python scripts for each chunk
for i in range(chunk_len):
    script_content = f"""
from diffusivity_calc_debug import run_md_simulation
import pandas as pd
import os
import numpy as np

from joblib import Parallel, delayed


df = pd.read_parquet('/home/lucky/Projects/ion_conductivity/data/MPContribs_armorphous_diffusivity.parquet')
df = df[df['data_properties_A_element'] == 'Li']  # only Li-ion conductivity
df = df[df['data_temperature_value']<5000]  # Remove 5000 K data

# Split the dataframe into chunks using numpy split
chunk_len = {chunk_len}  # number of GPUs

chunks = np.array_split(df, chunk_len)
df = chunks[{i}]
print(len(df))
# Parallelize above
results = Parallel(n_jobs=10)(delayed(run_md_simulation)(row['data_temperature_value'], i, "orb", {i}, False) for i, row in df.iterrows())

# Save results
df['diffusivity_orb'] = results
df.to_parquet('/home/lucky/Projects/ion_conductivity/data/MPContribs_armorphous_diffusivity_orb_{i}.parquet')
"""
    script_path = f'/home/lucky/Projects/ion_conductivity/scripts/run_md_chunk_{i}.py'
    with open(script_path, 'w') as script_file:
        script_file.write(script_content)

# Execute each script
processes = []
for i in range(chunk_len):
    script_path = f'/home/lucky/Projects/ion_conductivity/scripts/run_md_chunk_{i}.py'
    processes.append(subprocess.Popen(['python', script_path]))

# Wait for all processes to complete
for p in processes:
    p.wait()

# Combine results
results = []
for i in range(chunk_len):
    result_df = pd.read_parquet(f'/home/lucky/Projects/ion_conductivity/data/MPContribs_armorphous_diffusivity_orb_{i}.parquet')
    results.append(result_df)

final_df = pd.concat(results)
final_df.to_parquet('/home/lucky/Projects/ion_conductivity/data/MPContribs_armorphous_diffusivity_orb_final.parquet')

print("Time taken: ", time.time() - start_time)


