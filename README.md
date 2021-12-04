# Sentiment and Bias Analysis over News on Social Media

## Overview

The goal of this project is to construct a dataset consisting of posts written by english news channels such as CNN, BBC, Fox, Reuters, etc on Twitter alongside with their political bias label and perform some natural language processing tasks on the collected data.

## What I've done

- Constructed a novel dataset of famous news media data on Twitter labeled with political bias.
- Conducted multiple preprocessing and data analysis experiments on the collected data.
- Ran multiple NLP tasks including word similarity, NER, and DP.
- Trained deep language models, namely, GPT2 as a Causal LM and RoBERTa as a Masked LM.
- Trained deep classification models using word2vec, LSTM, BERT, and CNN.
- Gained experience with running models using CUDA-enabled GPU on my local machine.

## Notes

- The structure of the codebase is based on famous [cookiecutter datascience](https://drivendata.github.io/cookiecutter-data-science/).
- Codes for collecting and processing data are placed in `src/data` directory.
- Codes for designing and running models are placed in `src/models` directory.
- Reports of every phase is placed in `docs` directory.
- Run `src.data.make_dataset` module to download and build the dataset.
- Run `src.data.make_analysis_reports` module to extract figures and tables needed to compile latex reports.
- Compile `docs/phase_1_report/report.tex` to make `report.pdf` of phase 1. ([download](https://github.com/mohammadmahdiabdollahpour/Bias-In-News/raw/main/docs/phase_1_report/report.pdf))
- Compile `docs/phase_2_report/report.tex` to make `report.pdf` of phase 2. ([download](https://github.com/mohammadmahdiabdollahpour/Bias-In-News/raw/main/docs/phase_2_report/report.pdf))
