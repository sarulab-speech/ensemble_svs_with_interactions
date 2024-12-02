# このレシピは何？
jaCappella corpus と 「波音リツ」歌声データベースVer2 で multi-singer歌声合成モデルを学習するレシピ

## 0: データの前処理
学習には以下のデータが必要．
- 歌声の音声データ
- 歌声に対してアライメントされた HTS-style full-context label
- 楽譜由来の HTS-style full-context label

楽譜から HTS-style full-context label を生成する方法は`recipes/_common/db/musical_scores/README.md`を参照のこと．

データを以下のように配置する.
`downloads/hoge/{speaker_id}/{eng_songname}_aliened.lab`
`downloads/hoge/{speaker_id}/{eng_songname}_score.lab`
`downloads/hoge/{speaker_id}/{eng_songname}.wav`

Multi-track学習のためのデータ分割
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 20 --stop-stage 20
```

## 3: 特徴量の抽出
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 21 --stop-stage 21
```

## 4: Time-lag model のMulti-track学習
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 22 --stop-stage 22
```

## 5: Duration model のMulti-track学習
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 23 --stop-stage 23
```

## 6: Acoustic model のMulti-track学習
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 24 --stop-stage 24
```

## 8: Acoustic model のMulti-track学習
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 24 --stop-stage 24
```

## 8: Acoustic model のMulti-track学習
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 24 --stop-stage 24
```

## option:
話者を変える際は、configのspk, db_rootを変更し、data/の中身を削除
Multi-speaker trainingでは，以下の設定が必要
- config.yaml, spk: "Multi", spk_listを設定
- jaCappella_multi/conf/train_acoustic/model/multi_speaker_acoustic_nnsvs_world_multi_ar_f0.yaml の speaker_embedding:　num_embeddings: を適切な値に
- 音響モデルのdataloader, multi_speaker: True に設定, spk_listに話者IDを設定