import fasttext
from huggingface_hub import hf_hub_download
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


# def get_args_from_command_line():
#     """Parse the command line arguments."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_folder", type=str)
#     parser.add_argument("--input_path", type=str)
#     parser.add_argument("--output_path", type=str)
#     parser.add_argument("--drop_duplicates", type=bool, help="drop duplicated tweets from parquet files", default=False)
#     parser.add_argument("--resume", type=int, help="resuming a run, 0 or 1")
#     args = parser.parse_args()
#     return args


def get_language(text, model):
    if type(text) == str:
        if len(text) > 0:
            text = text.replace('\n', ' ')
            output = model.predict(text)
            lang = output[0][0].replace('__label__', '')
            score = output[1][0]
            return lang, score
        else:
            return None, None
    else:
        return None, None

def get_env_var(varname, default):
    if os.environ.get(varname) is not None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"RT\s+@\w+:\s*", "", text)     # remove RT prefix
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"@\w+", "", text)              # remove handles
    text = emoji.replace_emoji(text, replace="")  # remove emojis reliably
    text = text.replace('\n', ' ')
    return text.strip()


def extract_scores(text):
    labels, scores = model.predict(text, k=300)
    top_lang=labels[0]
    score_dict = dict(zip(labels, scores))

    # Keep scores for our selected languages
    selected_scores = {label: score_dict.get(label, 0.0) for label in fasttext_labels}

    return pd.Series({**selected_scores, "pred": top_lang})

if __name__ == '__main__':
    # args = get_args_from_command_line()
    # define vars
    path_to_data = '/scratch/mt4493/archive/just_another_day/data/processed/tweets_no_RT'
    output_path = '/scratch/mt4493/archive/just_another_day/data/processed/lang_detect_fasttext_latest_no_RT_more_mem'
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)
    # load model
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    model = fasttext.load_model(model_path)
    # List of labels you care about
    fasttext_labels = [
        "__label__bul_Cyrl",  # Bulgarian
        "__label__hrv_Latn",  # Croatian
        "__label__ces_Latn",  # Czech
        "__label__dan_Latn",  # Danish
        "__label__est_Latn",  # Estonian
        "__label__fin_Latn",  # Finnish
        "__label__ell_Grek",  # Greek
        "__label__hun_Latn",  # Hungarian
        "__label__lvs_Latn",  # Latvian
        "__label__lit_Latn",  # Lithuanian
        "__label__mlt_Latn",  # Maltese
        "__label__ron_Latn",  # Romanian
        "__label__slk_Latn",  # Slovak
        "__label__slv_Latn",  # Slovenian
        "__label__swe_Latn",  # Swedish
        # Additional common languages
        "__label__nld_Latn",  # Dutch
        "__label__eng_Latn",  # English
        "__label__fra_Latn",  # French
        "__label__deu_Latn",  # German
        "__label__ita_Latn",  # Italian
        "__label__pol_Latn",  # Polish
        "__label__por_Latn",  # Portuguese
        "__label__spa_Latn",  # Spanish
        "__label__arb_Arab",  # Arabic
        "__label__arb_Latn",  # Arabic in Latin alphabet
        "__label__heb_Hebr",  # Hebrew
    ]

    label_to_lang = {label: label.replace("__label__", "").split("_")[0] for label in fasttext_labels}

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
        df = pd.read_parquet(file, columns = ['id', 'author_id', 'text'])
        df['text'] = df['text'].apply(clean_text)

        # Define a regex pattern that matches only digits, whitespace, or punctuation
        pattern = rf"^[\d\s{re.escape(string.punctuation)}]*$"
        # Create the new column
        df["has_no_text"] = df["text"].apply(
            lambda x: 1 if isinstance(x, str) and re.fullmatch(pattern, x) else 0
        )
        df = df.join(df['text'].apply(extract_scores))
        # df[['tweet_lang_fasttext', 'tweet_lang_score']] = df['text'].apply(
        #     lambda x: pd.Series(get_language(text=x, model=model)))
        df.to_parquet(
            os.path.join(output_path,
                         filename_without_extension + str(getpass.getuser()) + '_random' + '-' + str(SLURM_JOB_ID)
                         + '.parquet'))
