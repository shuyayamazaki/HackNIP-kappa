import pandas as pd
import numpy as np
import argparse

def main(args):
    # read data
    df = pd.read_parquet(args.input_file)

    # filter 
    df = df[df['data_properties_A_element'] == 'Li']  # only Li-ion conductivity
    df = df[df['data_temperature_value']<5000]  # Remove 5000 K data
    df['y'] = df['data_properties_A_diffusivity_value'].apply(np.log10)
    print(len(df))

    # shuffle and split with fixed seed
    np.random.seed(42)
    df = df.sample(frac=1).reset_index(drop=True)
    df_train = df.iloc[:int(0.8*len(df))]
    df_val = df.iloc[int(0.8*len(df)):int(0.9*len(df))]
    df_test = df.iloc[int(0.9*len(df)):]
    df_train.to_parquet(args.output_train)
    df_val.to_parquet(args.output_val)
    df_test.to_parquet(args.output_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into train, validation, and test sets.')
    parser.add_argument('--input_file', type=str, default='MPContribs_armorphous_diffusivity.parquet', help='Path to the input parquet file')
    parser.add_argument('--output_train', type=str, help='Path to the output train parquet file', default='li-ion-conductivity_train.parquet')
    parser.add_argument('--output_val', type=str, help='Path to the output validation parquet file', default='li-ion-conductivity_val.parquet')
    parser.add_argument('--output_test', type=str, help='Path to the output test parquet file', default='li-ion-conductivity_test.parquet')
    args = parser.parse_args()
    
    main(args)