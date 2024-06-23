"""
Script to create train_x_lbs_st.tsv and train_x_rrs_st.tsv
from buffer.tsv and train_x_st.tsv and then update buffer.tsv

Usage: python sample.py <path_to_train_x_st> <path_to_buffer> <path_to_new_buffer> <data_aug_dir>
"""
import pandas as pd
import sys
import os
import numpy as np
from data_augmentation import add_columns, augment

# Parameter defining the fraction of samples taken by LBS and RRS from combined pool
LBS_SAMPLER_FRACTION = 0.8
RRS_SAMPLER_FRACTION = 0.6

def print_lang_freq(data, name):
    lang_freq = data.groupby('tgt_lang').size()
    print(f"\nFrequency of languages in {name}:\n", lang_freq)
    print(f"Total number of samples: {len(data)}\n")

def lbs(x_train, dataset, data_augmentation_dir, curr_task):
    lang_freq = dataset.groupby('tgt_lang').size()
    weights = {}
    for i in lang_freq.index:
        weights[i] = (1 - lang_freq[i]/(len(dataset) - 1))

    dataset['weights'] = dataset['tgt_lang'].map(weights)
    dataset['weights'] = dataset['weights'] / dataset['weights'].sum()

    train_x_lbs = dataset.sample(frac=LBS_SAMPLER_FRACTION, replace=True, weights='weights')

    dataset = dataset.drop('weights', axis=1)
    train_x_lbs = train_x_lbs.drop('weights', axis=1)

    lang_freq_lbs = train_x_lbs.groupby('tgt_lang').size()

    train_x_data = x_train.sample(n=lang_freq_lbs[curr_task])

    replay_data = train_x_lbs[train_x_lbs['tgt_lang'] != curr_task]

    train_x_lbs = pd.concat([train_x_data, replay_data], ignore_index=True)



    # Print lang frequency
    print_lang_freq(train_x_lbs, "language balanced data")

    #  Data Augmentation on duplicate samples

    duplicate_mask = train_x_lbs.duplicated()
    unique_samples = train_x_lbs[~duplicate_mask]
    duplicate_samples = train_x_lbs[duplicate_mask]

    print_lang_freq(duplicate_samples, 'duplicated samples')
    print_lang_freq(unique_samples, 'unique samples')

    duplicate_samples = augment(duplicate_samples, data_augmentation_dir)
    

    train_x_lbs = pd.concat([unique_samples, duplicate_samples], ignore_index=True)

    shuffled_idx = np.random.permutation(train_x_lbs.index)
    train_x_lbs = train_x_lbs.loc[shuffled_idx].reset_index(drop=True)



    return train_x_lbs

def rrs(dataset):
    train_x_rrs = dataset.sample(frac=RRS_SAMPLER_FRACTION)

    # Print lang frequency
    print_lang_freq(train_x_rrs, "randomly sampled data")

    return train_x_rrs

def update_buffer(buffer, x_train, split):
    print_lang_freq(buffer, "old buffer")

    # N is no of languages in old buffer
    N = len(buffer.groupby('tgt_lang').size())
    # Fraction of samples to keep in new buffer
    frac_to_keep = N / (N+1)
    # Calculate no of samples of new task to add to buffer
    no_of_samples_to_add = len(buffer) // (N + 1)
    buffer = buffer.sample(frac = frac_to_keep)
    # Add new samples to buffer with added columns for audio metadata
    x_train = x_train.drop_duplicates(subset='tgt_text', keep=False)
    buffer = pd.concat([buffer, add_columns(x_train.sample(n=no_of_samples_to_add), split, x_train)])
    print_lang_freq(buffer, "new buffer")

    return buffer

if __name__ == "__main__":
    # TODO Better arg handling
    # Process args
    x_train_file = sys.argv[1]
    buffer_file = sys.argv[2]
    new_buffer_file = sys.argv[3]
    data_augmentation_dir = sys.argv[4]

    curr_task = os.path.basename(x_train_file).split('_')[1]

    # Read tsv files
    print("-----Reading train data.....")
    x_train = pd.read_csv(x_train_file, sep='\t', on_bad_lines='warn', quotechar='	')
    print("-----Reading buffer data.....")
    buffer = pd.read_csv(buffer_file, sep='\t')

    # Dataset is combination of lang x training data and buffer
    dataset = pd.concat([x_train, buffer])

    # Print language frequency in dataset
    print_lang_freq(dataset, "dataset")

    # Get balanced and random data from their respective samplers
    print("\n-----Generating language balanced data.....")
    train_x_lbs = lbs(x_train, dataset, data_augmentation_dir, curr_task)
    print("\n-----Generating randomly sampled data.....")
    train_x_rrs = rrs(dataset)

    # Add new samples to buffer
    print("\n-----Updating buffer.....")
    split = os.path.basename(x_train_file).split('_')[0]
    buffer = update_buffer(buffer, x_train, split)

    # Write to files
    print("\n-----Writing data to files.....")
    with open(x_train_file[:-6] + 'lbs_st.tsv', 'w') as train_x_lbs_st_file:
        train_x_lbs_st_file.write(train_x_lbs.to_csv(sep='\t', index=False))
    with open(x_train_file[:-6] + 'rrs_st.tsv', 'w') as train_x_rrs_st_file:
        train_x_rrs_st_file.write(train_x_rrs.to_csv(sep='\t', index=False))
    with open(new_buffer_file, 'w') as buffer_filehandler:
        buffer_filehandler.write(buffer.to_csv(sep='\t', index=False))
