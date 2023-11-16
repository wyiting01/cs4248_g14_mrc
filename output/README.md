# Welcome to `/output`!

This folder consists of all the predictions from both the baseline and ensemble models in json files using `dev-v1.1.json`, and will be used for evaluation.

You can simply evaluate these output json files with the following command, assuming your current directory is in this folder:
```
python ../src/evaluate-v2.0.py ../data/raw/dev-v1.1.json pred.json
```
> Replace `pred.json` with the filename you are interested for evaluation
