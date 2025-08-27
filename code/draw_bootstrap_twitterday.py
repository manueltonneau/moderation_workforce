import os
# from pathlib import path
import pandas as pd
import argparse
from glob import glob
import getpass
import numpy as np
import re
import emoji
import string
import joblib
from pathlib import Path
import pickle

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str)
    args = parser.parse_args()
    return args

def get_env_var(varname, default):
    if os.environ.get(varname) is not None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var

if __name__ == '__main__':
    args = get_args_from_command_line()
    # define vars
    path_to_data = f'/scratch/mt4493/mod_workforce/calibrated_scores/{args.lang}'
    output_path = f'/scratch/mt4493/mod_workforce/bootstrap_samples/{args.lang}'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)
    if args.lang in ['arb_Arab', 'arb_Latn']:
        fasttext_lang = args.lang
    elif args.lang == 'bul':
        fasttext_lang = 'bul_Cyrl'
    elif args.lang == 'heb':
        fasttext_lang = 'heb_Hebr'
    else:
        fasttext_lang = f'{args.lang}_Latn'
    if not os.path.exists(os.path.join(output_path, f'counts_{SLURM_ARRAY_TASK_ID}.pkl')):
        df = pd.concat([pd.read_parquet(path, columns=['has_no_text', f'calibrated_scores_{args.lang}']) for path in Path(path_to_data).glob('*.parquet')])
        df = df.loc[df['has_no_text'] == 0][[f'calibrated_scores_{args.lang}']]
        bootstrap_count = int(1000/SLURM_ARRAY_TASK_COUNT)
        # assert isinstance(bootstrap_count, int), "x must be an integer"
        count_list = list()
        print(args.lang)
        for i in range(bootstrap_count):
            print(i)
            count = df.sample(df.shape[0], replace=True)[f'calibrated_scores_{args.lang}'].sum()
            count_list.append(count)
        # Save to pickle file
        with open(os.path.join(output_path, f'counts_{SLURM_ARRAY_TASK_ID}.pkl'), 'wb') as f:
            pickle.dump(count_list, f)

