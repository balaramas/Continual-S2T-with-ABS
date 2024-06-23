from pydub import AudioSegment
from pathlib import Path
import os
import ruamel.yaml
import yaml
import pandas as pd
from tqdm import tqdm
import random

def add_tags(file, tgtlang):
    l = []
    with open(file, 'r') as f:
        l = f.readlines()
        l[0] = l[0][:-1] + '\tsrc_lang\ttgt_lang\n'
        for i in range (1, len(l)):
            l[i] = l[i][:-1]+'\ten\t' + tgtlang + '\n'

    with open(file, 'w') as f:
        f.writelines(l)

def find_line_no(uid, txt_fh, tsv_file):
    tgt_txt = tsv_file.loc[tsv_file['id'] == uid]['tgt_text'].values[0]
    line_no = None
    for line_number, line in enumerate(txt_fh, start=0):
        current_sentence = line.strip()
        if current_sentence == tgt_txt:
            line_no = line_number
            break
    if line_no == None:
        raise Exception("Target Text not found.\n")
    return line_no

def assign_values(row, split, tsv_file, lines_dict):
    # audio_path column
    file_name = '_'.join(row['id'].split('_')[:-1])+ '.wav'
    data_path = os.path.dirname(row['audio'].split(':')[0]) + "/data/" + split + "/wav/" + file_name
    row['audio_path'] = data_path

    # start point and end point
    txt_file_path = os.path.dirname(row['audio'].split(':')[0]) + "/data/" + split + "/txt/" + split + '.' + row['tgt_lang']
    yaml_file_path = os.path.dirname(row['audio'].split(':')[0]) + "/data/" + split + "/txt/" + split + '.yaml'


    with open(txt_file_path, 'r') as txt_fh:
        line_no = find_line_no(row['id'], txt_fh, tsv_file)


    if (yaml_file_path in lines_dict):
        lines = lines_dict[yaml_file_path]
    else:
        with open(yaml_file_path, 'r') as file:
            yaml_text = file.read()
            lines = yaml_text.split('\n')
            lines_dict[yaml_file_path] = lines
    
    s=lines[line_no]
   
    yaml_line = yaml.safe_load(s)[0]

    row['start_point'] = yaml_line['offset']
    row['end_point'] = yaml_line['offset'] + yaml_line['duration']
    row['yaml_line'] = yaml_file_path + ':' + str(line_no)

    return row


def add_columns(buffer, split, x_train_tsv_file):

    buffer['yaml_line'] =''

    tqdm.pandas()

    lines_dict = {}
    
    buffer = buffer.progress_apply(assign_values, args=(split, x_train_tsv_file, lines_dict), axis=1)

    return buffer

def extract_audio_segment(input_path, output_path, start_time, end_time):
    # Load the input audio file
    audio = AudioSegment.from_file(input_path)

    # Convert start and end times from seconds to milliseconds
    start_time_ms = start_time * 1000
    end_time_ms = end_time * 1000

    # Extract the specified portion of the audio
    extracted_audio = audio[start_time_ms:end_time_ms]

    # Export the extracted audio to the output file
    extracted_audio.export(output_path, format="wav")


def augment_sample(ori_sample_path, aug_sample_path, duration):

    filename = ori_sample_path
    sound = AudioSegment.from_file(filename, format=filename[-3:])

    # octaves = [0.2, 0.25, 0.3, 0.35, 0.4, -0.2, -0.25, -0.3, -0.35]
    # octaves = [0.17, 0.22, 0.27, 0.32, 0.37, -0.23, -0.28, -0.33, -0.38]
    octaves = [0.23, 0.28, 0.33, 0.38, 0.43, -0.17, -0.22, -0.27, -0.32]

    if (duration < 2):
        # octaves = [-0.2, -0.25, -0.3, -0.35]
        # octaves = [-0.23, -0.28, -0.33, -0.38]
        octaves = [-0.17, -0.22, -0.27, -0.32]
        
    new_sample_rate = int(sound.frame_rate * (2.0 ** random.choice(octaves)))
    hipitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})

    #pitched changed audio
    hipitch_sound = hipitch_sound.set_frame_rate(16000)

    hipitch_sound.export(aug_sample_path, format="wav")

    length_ms = len(AudioSegment.from_file(aug_sample_path))
    duration = length_ms / 1000
    return duration


    
    
def augment(samples, aug_dir):
    tgt_languages = samples['tgt_lang'].unique().tolist()
    aug_path = Path(aug_dir)
    # Create required directory structure

    # dictionary to store opened file handlers
    txt_files_dict = {} # not closed

    for i in tgt_languages:
        wav_path = aug_path.joinpath('en-'+i+'/data/train/wav_orignal')
        wav_aug = aug_path.joinpath('en-'+i+'/data/train/wav')

        txt_path = aug_path.joinpath('en-'+i+'/data/train/txt')
        wav_path.mkdir(parents=True, exist_ok=True)
        wav_aug.mkdir(parents=True, exist_ok=True)

        txt_path.mkdir(parents=True, exist_ok=True)

        with open(str(txt_path.joinpath('train.' + i)), "w") as _:
            pass
        with open(str(txt_path.joinpath('train.yaml')), "w") as _:
            pass

        txt_files_dict[txt_path.joinpath('train.'+i)] = open(str(txt_path.joinpath('train.' + i)), 'a')
        txt_files_dict[txt_path.joinpath('train.yaml')] = open(str(txt_path.joinpath('train.yaml')), 'a')

    
    # dictionary to store existing yaml text
    yaml_text_dict = {}

    # dictionary to store new yaml text
    new_yaml_dict = {}


    # Extract audio samples to wav directory, add lines to yaml file and tgt txt file
    for _, row in tqdm(samples.iterrows(), total=len(samples), desc="Augmenting"):
        txt_path = aug_path.joinpath('en-'+row['tgt_lang']+'/data/train/txt')

        # Extract audio segment
        extract_audio_segment(row['audio_path'], str(aug_path.joinpath('en-'+row['tgt_lang']+'/data/train/wav_orignal/'+row['id']+'.wav')), row['start_point'],row['end_point'])

        # add train.<tgt_lang> lines
        txt_files_dict[txt_path.joinpath('train.'+row['tgt_lang'])].write(row['tgt_text'] + '\n')

        # yaml lines
        yaml_path, line_no = row['yaml_line'].split(':')[0], int(row['yaml_line'].split(':')[1])
        
        if yaml_path not in yaml_text_dict:
            with open(yaml_path, 'r') as file:
                yaml_text_dict[yaml_path] = file.read().split('\n')
            

        new_record = yaml.safe_load(yaml_text_dict[yaml_path][line_no])[0]

        new_length = augment_sample(str(aug_path.joinpath('en-'+row['tgt_lang']+'/data/train/wav_orignal/'+row['id']+'.wav')), str(aug_path.joinpath('en-'+row['tgt_lang']+'/data/train/wav/'+row['id']+'.wav')), new_record['duration'])

        new_record['offset'] = 0
        new_record['duration'] = new_length
        new_record['wav'] = row['id']+'.wav'


        if (txt_path.joinpath('train.yaml') in new_yaml_dict):
            new_yaml_dict[txt_path.joinpath('train.yaml')].append(new_record)
        else:
            new_yaml_dict[txt_path.joinpath('train.yaml')] = [new_record,]    

    
    for i in new_yaml_dict:
        with open(i, 'w') as file:
            ruamel.yaml.dump(new_yaml_dict[i], file)
    
    for i in txt_files_dict:
        txt_files_dict[i].close()



    os.system(f"python ./prep_mustc_data_aug.py \
    --data-root {aug_dir} --task st \
    --vocab-type unigram --vocab-size 8000")


    augmented_samples = pd.DataFrame()
    for i in tgt_languages:
        add_tags(str(aug_path.joinpath(f'en-{i}/train_{i}_st.tsv')), i)
        augmented_samples = pd.concat([augmented_samples, pd.read_csv(str(aug_path.joinpath(f'en-{i}/train_{i}_st.tsv')), sep='\t')], ignore_index=True)
    

    return augmented_samples