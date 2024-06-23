"""
Script to run inference on all chckpoints in given directory
and save results in given tsv file

Usage:  python inference_script.py <path_to_checkpoints_directory> \
        <path_to_result_tsv_file> <mustc_root> <name_of_tst-COMMON_file_excluding_.tsv>
"""
import sys
import os
import subprocess
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    # Process args
    checkpoint_dir = sys.argv[1]
    results_file = sys.argv[2]
    mustc_root = sys.argv[3]
    tst_file = sys.argv[4]

    blue_data = pd.DataFrame(columns=['index', 'checkpoint_file_name', 'blue_score'])
    with open(results_file, 'w') as blue_tsv:
        blue_tsv.write(blue_data.to_csv(sep='\t', index=False))

    # Get list of files in the dir
    checkpoint_files = os.listdir(checkpoint_dir)
    checkpoint_files.remove('checkpoint_last.pt')
    if 'checkpoint_best.py' in checkpoint_files:
        checkpoint_files.remove('checkpoint_best.pt')
    checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x[10:-3]))
    for i in tqdm(range(len(checkpoint_files))):
        command = ["fairseq-generate", mustc_root, "--config-yaml", "config_st.yaml", "--gen-subset", tst_file, "--task", "speech_to_text", "--prefix-size", "1", "--path", (checkpoint_dir + '/' + checkpoint_files[i]), "--max-tokens", "50000", "--beam", "5", "--scoring", "sacrebleu", "--quiet"]
        blue_line = subprocess.run(command, capture_output=True, text=True).stdout.split("\n")[-2]
        blue_score_lst = blue_line.split(" ")[6:8]
        blue_score = (float(blue_score_lst[0]) + float(blue_score_lst[1].split('/')[0]) + float(blue_score_lst[1].split('/')[1]) + float(blue_score_lst[1].split('/')[2]) + float(blue_score_lst[1].split('/')[3]))/5
        print(blue_score)
        pd.DataFrame({'index': checkpoint_files[i][10:-3], 'checkpoint_file_name': [checkpoint_files[i]], 'blue_score':[blue_score]}).to_csv(results_file, sep='\t', index=False, header=False, mode='a')

    
