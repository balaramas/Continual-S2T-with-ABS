"""
Script to create buffer.tsv
from train_x_st.tsv and required buffer size
with added columns for audio file metadata

Usage: python create_buffer.py <path_to_train_x_st> <path_to_buffer> <buffer_size>
"""
import pandas as pd
import sys
import os

from data_augmentation import add_columns


if __name__ == "__main__":
    x_train_file = sys.argv[1]
    buffer_file = sys.argv[2]
    buffer_size = sys.argv[3]

    split = os.path.basename(x_train_file).split('_')[0]

    # Read tsv file
    print("-----Reading train data.....")
    x_train = pd.read_csv(x_train_file, sep='\t', dtype='str', quoting=3)

    x_train = x_train.drop_duplicates(subset='tgt_text', keep=False)

    buffer = x_train.sample(n=int(buffer_size))

    print("-----Adding columns for data augmentation")
    buffer = add_columns(buffer, split, x_train)

    print("\n-----Writing data to files.....")
    with open(buffer_file, 'w') as buffer_filehandler:
        buffer_filehandler.write(buffer.to_csv(sep='\t', index=False))