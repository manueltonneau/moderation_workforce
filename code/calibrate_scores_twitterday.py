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

def calibrate_scores(lang, raw_scores):
    """
    Loads a fitted IsotonicRegression model from a joblib file and calibrates raw scores.

    Parameters:
    - model_path (str): Path to the saved IsotonicRegression model (.joblib).
    - raw_scores (array-like): List or array of uncalibrated scores (floats).

    Returns:
    - np.ndarray: Calibrated scores.
    """
    model_path = f'/scratch/mt4493/mod_workforce/calibration_models/isotonic_regression_model_{lang}.joblib'
    # Load the fitted isotonic regression model
    iso_reg = joblib.load(model_path)

    # Ensure input is a numpy array
    raw_scores = np.asarray(raw_scores)

    # Apply calibration
    calibrated = iso_reg.transform(raw_scores)

    return calibrated

if __name__ == '__main__':
    args = get_args_from_command_line()
    # define vars
    path_to_data = '/scratch/mt4493/archive/just_another_day/data/processed/lang_detect_fasttext_latest_no_RT_more_mem'
    output_path = f'/scratch/mt4493/mod_workforce/calibrated_scores/{args.lang}'
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
    elif args.lang == 'eng_new':
        fasttext_lang = f'eng_Latn'
    else:
        fasttext_lang = f'{args.lang}_Latn'

    # label_to_lang = {label: label.replace("__label__", "").split("_")[0] for label in fasttext_labels}

    # get batch of files
    input_files_list = glob(os.path.join(path_to_data, '*.parquet'))
    print(input_files_list)
    paths = list(np.array_split(
            input_files_list,
            SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]
        )
    print('#files in paths_to_random', len(paths))
    # loop over files
    for file in paths:
        filename_without_extension = os.path.splitext(os.path.splitext(file.split('/')[-1])[0])[0]
        df = pd.read_parquet(file, columns = ['id', 'has_no_text', f'__label__{fasttext_lang}'])
        df[f'calibrated_scores_{args.lang}'] = calibrate_scores(lang=args.lang, raw_scores=df[f'__label__{fasttext_lang}'])
        # df[['tweet_lang_fasttext', 'tweet_lang_score']] = df['text'].apply(
        #     lambda x: pd.Series(get_language(text=x, model=model)))
        df.to_parquet(
            os.path.join(output_path,
                         filename_without_extension + str(getpass.getuser()) + '_random' + '-' + str(SLURM_JOB_ID)
                         + '.parquet'))
