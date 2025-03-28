
from diffusivity_calc_debug import run_md_simulation
import pandas as pd
import os
import numpy as np

from joblib import Parallel, delayed


df = pd.read_parquet('/home/lucky/Projects/ion_conductivity/data/MPContribs_armorphous_diffusivity.parquet')
df = df[df['data_properties_A_element'] == 'Li']  # only Li-ion conductivity
df = df[df['data_temperature_value']<5000]  # Remove 5000 K data

# Split the dataframe into chunks using numpy split
chunk_len = 6  # number of GPUs

chunks = np.array_split(df, chunk_len)
df = chunks[3]
print(len(df))
# Parallelize above
results = Parallel(n_jobs=10)(delayed(run_md_simulation)(row['data_temperature_value'], i, "orb", 3, False) for i, row in df.iterrows())

# Save results
df['diffusivity_orb'] = results
df.to_parquet('/home/lucky/Projects/ion_conductivity/data/MPContribs_armorphous_diffusivity_orb_3.parquet')
