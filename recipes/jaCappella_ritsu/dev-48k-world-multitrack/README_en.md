# What is this recipe?
This is a recipe for training a multi-singer singing voice synthesis model using the jaCappella corpus and the "Namine Ritsu" singing voice database Ver2.

## 0: Data Preprocessing
The following data is required for training:
- Singing voice audio data
- HTS-style full-context labels aligned with the singing voice
- HTS-style full-context labels based the musical score

Refer to `recipes/_common/db/musical_scores/README.md` for instructions on generating HTS-style full-context labels from musical scores.

Place the data as follows:
```
downloads/hoge/{speaker_id}/{eng_songname}_aligned.lab
downloads/hoge/{speaker_id}/{eng_songname}_score.lab
downloads/hoge/{speaker_id}/{eng_songname}.wav
```

Data splitting for multi-track training:
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 20 --stop-stage 20
```

## 1: Feature Extraction
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 21 --stop-stage 21
```

## 2: Multi-track Training of the Time-lag Model
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 22 --stop-stage 22
```

## 3: Multi-track Training of the Duration Model
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 23 --stop-stage 23
```

## 4: Multi-track Training of the Acoustic Model
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 24 --stop-stage 24
```

## 5: Generation of Timing Features
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 25 --stop-stage 25
```

## 6: Synthesis of Singing Voice
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 26 --stop-stage 26
```

## Option:
When changing the speaker, modify `spk` and `db_root` in the config, and delete the contents of `data/`.

For multi-speaker training, the following settings are required:
- In `config.yaml`, set `spk` to "Multi" and configure `spk_list`.
- In `jaCappella_multi/conf/train_acoustic/model/multi_speaker_acoustic_nnsvs_world_multi_ar_f0.yaml`, set `speaker_embedding: num_embeddings:` to an appropriate value.
- In the dataloader for the acoustic model, set `multi_speaker` to True and configure `spk_list` with speaker IDs.
