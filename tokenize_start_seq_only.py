from datasets import load_from_disk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle


"""This script is used to tokenize the dataset for training the LM model.
It is only for the case when we only use the first input_length days of the access data.
It is copied from a jupyter notebook. Please set the paths accordingly when using."""

ds_path = "/share/dean/arxiv-data/cumulative_cnts_with_metadata_citation_offset_v4/train_day5_access_dropped_filtered_shorter_than_5_years_tokenized"
ds = load_from_disk(ds_path)
ds.save_to_disk("/share/dean/tmp/ds_copy")
ds_copy = load_from_disk("/share/dean/tmp/ds_copy")

ds.set_format("numpy")
input_length = 365


# citaton_inputs, download_inputs, citatoin_outputs
def get_one_chunk(seq):
    return seq[:input_length]


def get_outputs(seq):
    HORIZON = 365 * 5
    return seq[HORIZON - input_length : HORIZON]


def get_chunks(row):
    citation_tokens = row["citation_tokens"]
    access_tokens = row["access_tokens"]
    citation_inputs = get_one_chunk(citation_tokens)
    access_inputs = get_one_chunk(access_tokens)
    citation_outputs = get_outputs(citation_tokens)
    return {
        "citation_inputs": ",".join(citation_inputs),
        "access_inputs": ",".join(access_inputs),
        "citation_outputs": ",".join(citation_outputs),
    }


ds_filter = ds_copy.map(get_chunks)

citation_input_str = []
access_input_str = []
citation_output_str = []

from tqdm import tqdm

for row in tqdm(ds_filter):
    citation_input_str.append(row["citation_inputs"])
    access_input_str.append(row["access_inputs"])
    citation_output_str.append(row["citation_outputs"])

    citation_input_str = "\n".join(citation_input_str)
    access_input_str = "\n".join(access_input_str)
    citation_output_str = "\n".join(citation_output_str)


save_root = "/share/dean/arxiv-data/tokenized_strings_start_only_365_v4"
os.makedirs(save_root, exist_ok=True)
with open(os.path.join(save_root, "citation_input_train.txt"), "w") as f:
    f.write(citation_input_str)
with open(os.path.join(save_root, "citation_target_train.txt"), "w") as f:
    f.write(citation_output_str)
with open(os.path.join(save_root, "access_input_train.txt"), "w") as f:
    f.write(access_input_str)
