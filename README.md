# T-SNE for Unsupervised Features, testing gender, phonemes and language

## Setting up the conda environment

`conda env create --file environment.yaml`

## Preparing the CV sub dataset

You will need to download version 6.1 or more recent of the CommonVoice dataset
The list of utterances used in the subset are given in Dataset/cv_fr_wavlist.txt and Dataset/cv_en_wavlist.txt. *Fill with more information on the dataset and info on how to recreate it / link on how to create it*

You will then need to create a FEAT directory, containing the following files:

```bash 
FEATS
│
├─────── CV_en
│        ├── common_voice_en_7851095.txt
│        ├── common_voice_en_7851096.txt
│        ... 
│        └── common_voice_en_9823212.txt
└─────── CV_fr
         ├── common_voice_fr_7851095.txt
         ├── common_voice_fr_7851096.txt
         ... 
         └── common_voice_fr_9823212.txt
```

For each txt file, each line corresponds to one frame, with all values for each dimension separated by a space.
Here we use 10ms windows for frames.

E.g, for a wav file of 30ms with 5 dimensions output features, the txt file should look like:
```bash 
    1.342 4.231 -3.231e-03 5.42 1.21
    3.231 5.12  4.213      6.41 4.312
    2.12  -1.2  3.2343e-01 4.21 5.532
```



## Running the TSNE_Interspeech notebook

### Variables to fill in

You will need to fill in the correct path to the "VARIABLES TO FILL IN" cell. Here is a list below of what they should look like:

* FR_UTT2GENDER : Text file with the French test set utterance to gender mapping. Present in "Dataset".
* EN_UTT2GENDER : Text file with the English test set utterance to gender mapping. Present in "Dataset".
* FR_ALI_CTM : Frame to Phone alignment in CTM format for the French test set. Present in "Dataset"
* EN_ALI_CTM : Frame to Phone alignment in CTM format for the English test set. Present in "Dataset"


* FEATS : Location to the FEATS directory described above.
