Here is the English translation of the provided Markdown:

# Token alignment for singing voice synthesis

## -1: Preparation
- Create a `NEUTRINO` folder under `musical_scores` and place the NEUTRINO package (Electron v1.3.x, Online version) there. A symbolic link is also acceptable. Note that `./NEUTRINO/settings/dic/*.utf_8.*` will be overwritten. Along with label generation, execution logs will also be saved in the same directory.
- Check the contents of `config.yaml`. The subsequent shell commands will be executed based on this configuration.
- Save the score data in `./scores_for_SVS` (this directory is defined in `config.yaml`).
- (`$ chmod 755 ./*.sh`)

## 0: Separate the score by parts
```
$ ./0_score_separation.sh
```
Separate the scores in `../scores_for_SVS` by parts and save them in `./scores_parts`.

## 1: Generate labels for singing voice synthesis
```
$ ./1_label_synthesis.sh 
```
Generate labels from the musicXML files in `./scores_parts` and save them in `./label`.

## 2: Synthesize NEUTRINO singing voice from labels (if there are no labels aligned with the singing voice)
```
$ ./2_neutrino_synthesis.sh
```
This process takes quite some time. Save the synthesized singing voice in `./syn_wav`.

## 3: Align NEUTRINO singing voice with a cappella singing voice (if there are no labels aligned with the singing voice)
```
$ ./3_label_alignment.sh
```
Save the timing of the a cappella singing voice in `./aligned_label`. This label file can be loaded in Audacity.
