# Welcome to `/output`!

This folder consists of all the predictions from both the baseline and ensemble models in json files using `dev-v1.1.json`, and will be used for evaluation with the official script `evaluate-v2.0.py`.

If interested in Exact Scores and F1 Scores of our outputs, you can simply evaluate them with the following command, or visit [here](https://github.com/wyiting01/cs4248_g14_mrc#experiment-results). 
Assuming your current directory is in this folder:
```
python ../src/evaluate-v2.0.py ../data/raw/dev-v1.1.json pred.json
```
> Replace `pred.json` with the filename you are interested to evaluate
