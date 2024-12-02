# Token alignment for singing voice synthesis

## -1: 前準備
- `musical_scores`以下に`NEUTRINO`フォルダを作り，そこにNEUTRINO一式(Electron v1.3.x, Online版)を置く．symbolic link でもOK．仕様上，`./NEUTRINO/settings/dic/*.utf_8.*`は上書きされるので注意． ラベル生成と合わせて，実行ログも同じディレクトリに保存される．
- config.yaml の中身を確認．以降の shell は，この設定を元に実行される．
- 楽譜データを `./scores_for_SVS` に保存．(このディレクトリは，config.yaml で定義している．)
- (`$ chmod 755 ./*.sh`)

## 0: 楽譜をパートごとに分ける
```
$ ./0_score_separation.sh
```
`../scores_for_SVS` の楽譜をパートごとに分解して `./scores_parts` に保存．

## 1: 歌声合成用のラベルを生成
```
$ ./1_label_synthesis.sh 
```
`./scores_parts` の musicXML からラベルを生成し，`./label` に保存．

## 2: NEUTRINO を使って，ラベルからNEUTRINO歌声を合成 (歌声にalignされたラベルがない場合)
```
$ ./2_neutrino_synthesis.sh
```
結構時間がかかるので注意．合成歌声を `./syn_wav` に保存．

## 3: NEUTRINO歌声と，アカペラ歌声のアライメント
```
$ ./3_label_alignment.sh
```
アカペラ歌声の時刻を `./aligned_label` に保存．このラベルファイルは，audacity で読み込み可能．