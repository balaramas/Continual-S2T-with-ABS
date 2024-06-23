# Contiual Sppech to text translation using augmented bi sampler

This repository is for extra scripts required to run the continual learning model for speech to text task from FAIRSEQ repository. So this repository requires you already have fairseq installed and have all the files of speech-to-text task.

The data pre process is as same as mentioned in fairseq 



## Data Preparation
[Download](https://ict.fbk.eu/must-c) and unpack MuST-C data to a path
`${MUSTC_ROOT}/en-${TARGET_LANG_ID}`, then preprocess it with
```bash
# additional Python packages for S2T data processing/model training
pip install pandas torchaudio soundfile sentencepiece

# Generate TSV manifests, features, vocabulary
# and configuration for each language
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr \
  --vocab-type unigram --vocab-size 5000
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type unigram --vocab-size 8000

# Add vocabulary and configuration for joint data
# (based on the manifests and features generated above)
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr --joint \
  --vocab-type unigram --vocab-size 10000
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st --joint \
  --vocab-type unigram --vocab-size 10000
```
The generated files (manifest, features, vocabulary and data configuration) will be added to
`${MUSTC_ROOT}/en-${TARGET_LANG_ID}` (per-language data) and `MUSTC_ROOT` (joint data).

- Here have to first use the script add_lang_tags.py to add language tags to the manifest data in the tsv files. Example add "fr" for french and "de" for German language.


## Next step is to run fairseq s2t for first data pair, in the usual way.

## After that create a buffer for this language, use the "create_buffer.py" file, which will store a fraction of the samples from the tsv manifest file.

## Run "sample_aug_entire_buffer.py" to run two samplers , Random Sampler and Language Proportional Sampler and create the train splits or the tsv files to feed to the transformer model. 

- It takes the buffer.tsv as input as well as the new language pair train split, example 'fr_train.tsv' and creates the two samples for RS and PLS. 
- It also updates the buffer after the train split is created automatically by putting samples of the new language also.
- While sampling 'prep_mustc_data_aug.py' is used to randomly perform augmentation to the samples to balance the minority samples.

##  Next step is to train the s2t model together with the command below

```bash
fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_st.yaml \
  --train-subset train_RS_st, train_PLS_st \ 
  --valid-subset dev_x1, dev_x2 \
  --save-dir ${MULTILINGUAL_ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --ignore-prefix-size 1 --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${JOINT_ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
```
where `ST_SAVE_DIR` (`MULTILINGUAL_ST_SAVE_DIR`) is the checkpoint root path. The ST encoder is pre-trained by ASR
for faster training and better performance: `--load-pretrained-encoder-from <(JOINT_)ASR checkpoint path>`. We set
`--update-freq 8` to simulate 8 GPUs with 1 GPU. You may want to update it accordingly when using more than 1 GPU.
For multilingual models, we prepend target language ID token as target BOS, which should be excluded from
the training loss via `--ignore-prefix-size 1`.

##  Model inference

```bash
--prefix-size 1 has to be given to consider the lang tag

fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --gen-subset tst-COMMON_st --task speech_to_text \
  --prefix-size 1 --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu

```

# The model can be trained as many times using the steps mentioned above by updating the buffer and training with the new language as well as the buffer to maintain the accuracy of the model for all languages.






 
