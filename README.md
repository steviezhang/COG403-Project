# COG403-Project

To load model, download model trained under 11 epochs at this link (https://drive.google.com/drive/folders/1jmRsTGdCsVuYIRhgokGfYoErz8q-qSxO?usp=sharing) and put epoch11.model into the top level inside COG403-Project. Then the model will be loaded into the rnng notebook.

The rnng notebook base code was used from https://github.com/kmccurdy/rnng-notebook under Apache Licensing.
Additions were made for data analysis and stats for paper.

main.py file contains experiment code for pcfg induction and scoring code

Parsetree directory contains Adam's corpus in treebank fashion parsed for use in main.py

data directory contains training/testing/validation files for rnng model, as well as a vocab cluster file.

Adam directory contains the original data from he CHILDES corpus which we then processed for use.
