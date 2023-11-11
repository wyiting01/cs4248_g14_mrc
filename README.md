# CS4248 G14 Machine Reading Comprehension on SQuAD

## Dataset
Training Data: `data/raw/train-v1.1.json`  
Test Data: `data/raw/dev-v1.1.json`

### Data Preprocessing  
We preprocess the json files by separating context, question, question id, answer and answer span into individual file. This preprocessing is applied to both `train-v1.1.json` and `dev-v1.1.json`.

To use our preprocessing script, we will need to specify some parameters as shown below:
```
python src/preprocessing.py --file_path "data/raw/train-v1.1.json" --train
python src/preprocessing.py --file_path "data/raw/dev-v1.1.json" --test
python src/preprocessing.py --file_path "data/raw/train-v1.1.json" --train_val
```
The third line illustrates how we can further split the training data into train set and validation set. This is essential for our ensemble learning with unequal weightage.
> Note that these proprocessed files can also be found in `data/curated`

## Experiment Results
The table below reports the Exact and F1 scores evaluted from the official SQuAD evaluation script `evaluate-v2.0.py`.  

|  | Model | Exact | F1 |
| ---- | ----- | ----- | -- |
| BASELINE | BERT-SQuAD | 57.79564806054872 | 72.18288133527203 |
| | RoBERTa | 81.20151371807 | 88.41621816924753 |
| | XLNet | 75.34531693472091 | 84.06374401013184 |
| | biLSTM | TBC | TBC | 
| ENSEMBLE | Equal - Maximum | 78.89309366130558 | 86.70557785319596 |
| | Equal - Multiplicative | <ins>**82.19489120151371**</ins> | <ins>**88.9101732833653**</ins> |
| | Unequal - Optuna | 81.47587511825922 | 88.20920854244099

To run the evaluation script with model predictions `pred.json`, simply run the command below:
```
python evaluate-v2.0.py data/raw/dev-v1.1.json pred.json
```
> Note that `pred.json` can be replaced with any output json files containing the predictions of your target model found in `output/`

## Baseline Models

### 1. BERT-SQuAD
- Script: `src/bert.py`
- Model: [BERT-SQuAD](https://github.com/kamalkraj/BERT-SQuAD/tree/master)

```
# training:
python

# testing:
python
```

### 2. RoBERTa
- Script: `src/roberta.py`
- Model: [RobertaForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaForQuestionAnswering)

```
# training:
python src/roberta.py --train --data_path "data/curated/training_data" --model_path "model/xlnet.pt"

# testing:
python src/roberta.py --test --data_path "data/curated/test_data" --model_path "model/xlnet.pt" --output_path "output/roberta_pred.json"
```

### 3. XLNet
- Script: `src/xlnet.py`
- Model: [XLNetForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/xlnet#transformers.XLNetForQuestionAnswering)  

```
# training:
python src/xlnet.py --train --data_path "data/curated/training_data" --model_path "model/xlnet.pt"

# testing:
python src/xlnet.py --test --data_path "data/curated/test_data" --model_path "model/xlnet.pt" --output_path "output/xlnet_pred.json"
```

### 4. biLSTM
- Script: `src/bilstm_bert.py`
- Model: biLSTM-BERT

```
# training:
python src/bilstm_bert.py --train_path data/curated/training_data/ --model_path bilstm.pt

# testing:
python src/bilstm_bert.py --test_path data/curated/test_data/
```

## Ensemble Models

### Equal Weightage

**1. Maximum Score**  
This ensemble learning method will find common start and end indices for models participating in ensemble. We then extract the maximum score for each of these common indices, and output the best answer with the highest score calculated by adding the final start and end scores together.  

**2. Multiplicative Score**  
Similar to maximum score, but instead of extracting the max score for common indices, we multiply their scores together.

Run the following to get predictions from our ensemble models:
```
# for maximum score
python src/ensemble.py\
        --maximum\
        --data_path "data/curated/test_data"\
        --roberta_path "model/roberta.pt"\
        --xlnet_path "model/xlnet.pt"\
        --output_path "output/ensemble_max_pred.json"

# for multiplicative score
python src/ensemble.py\
        --multiplicative\
        --data_path "data/curated/test_data"\
        --roberta_path "model/roberta.pt"\
        --xlnet_path "model/xlnet.pt"\
        --output_path "output/ensemble_mul_pred.json"
```

### Unequal Weightage
**3. Weighting based on Optuna**  
In this approach, we will deploy the Optuna framework for hyperparameter optimsation.  
- **Preprocessing**: The training data `train-v1.1.json` will be split into train set and validation set (80:20), which can be obtained by running the third line illustrated under Data Preprocessing section.  
- **Part A**: The base models mentioned above will be trained using the train set.
- **Part B**: We then get candidates indices and their scores for the validation set and use Optuna framework to obtain the optimal weights by trial and error. (src/ensemble_optuna.ipynb)
- **Part C**: Lastly, we get candidates indices and their scores for the test set, and combine the scores for the common indices using weights determined in Part B.

```
# Part A: Training on train set (80% of train-v1.1.json)
python ensemble_unequal_optuna.py --train --roberta --data_path "../data/curated/ensemble_data/train" --xlnet_path "../model/xlnet.pt" --roberta_path "../model/roberta.pt" --xlnet_dict "../xlnet.json" --roberta_dict "../roberta.json"
python ensemble_unequal_optuna.py --train --xlnet --data_path "../data/curated/ensemble_data/train" --xlnet_path "../model/xlnet.pt" --roberta_path "../model/roberta.pt" --xlnet_dict "../xlnet.json" --roberta_dict "../roberta.json"

# Part B: Obtaining candidates for validation set (20% of train-v1.1.json)
python ensemble_unequal_optuna.py --get_candidates --xlnet --data_path "../data/curated/ensemble_data/val" --xlnet_path "../model/xlnet.pt" --roberta_path "../model/roberta.pt" --xlnet_dict "../ensemble/xlnet_val.json" --roberta_dict "../ensemble/roberta_val.json"
python ensemble_unequal_optuna.py --get_candidates --roberta --data_path "../data/curated/ensemble_data/val" --xlnet_path "../model/xlnet.pt" --roberta_path "../model/roberta.pt" --xlnet_dict "../ensemble/xlnet_val.json" --roberta_dict "../ensemble/roberta_val.json"

# Part C: Obtaining candidates for test set and perform weighting based on Optuna weights
python ensemble_unequal_optuna.py --get_candidates --xlnet --data_path "../data/curated/test_data" --xlnet_path "../model/xlnet.pt" --roberta_path "../model/roberta.pt" --xlnet_dict "../ensemble/xlnet_test.json" --roberta_dict "../ensemble/roberta_test.json"
python ensemble_unequal_optuna.py --get_candidates --roberta --data_path "../data/curated/test_data" --xlnet_path "../model/xlnet.pt" --roberta_path "../model/roberta.pt" --xlnet_dict "../ensemble/xlnet_test.json" --roberta_dict "../ensemble/roberta_test.json"
python ensemble_unequal_optuna.py --test --xlnet_dict "../ensemble/xlnet_test.json" --roberta_dict "../ensemble/roberta_test.json" --output_path "../ensemble/ensemble_optuna_pred.json" --xlnet_weight 0.41 --roberta_weight 0.59
```
> Note that training of both models (as well as getting candidates) cannot be performed at the same time due to the limitation of the cluster, hence running them separately is required.

## Directory Structure
To navigate around this repository, you can refer to the directory tree below:
```
.
├── data
|    ├── raw
|    |    ├── dev-v1.1.json
|    |    └── train-v1.1.json
|    └── curated
|         ├── ensemble_data
|         |    ├── train
|         |    |    ├── answer
|         |    |    ├── answer_span
|         |    |    ├── context
|         |    |    ├── question
|         |    |    └── question_id
|         |    └── val
|         |         ├── answer
|         |         ├── answer_span
|         |         ├── context
|         |         ├── question
|         |         └── question_id
|         ├── test_data
|         |    ├── answer
|         |    ├── answer_span
|         |    ├── context
|         |    ├── question
|         |    └── question_id
|         └── training_data
|              ├── answer
|              ├── answer_span
|              ├── context
|              ├── question
|              └── question_id
├── model
|    ├── bert_squad.pt
|    ├── bilstm.pt
|    ├── roberta.pt
|    └── xlnet.pt
├── output
|    ├── bert_squad_pred.json
|    ├── bilstm_pred.json
|    ├── ensemble_max_pred.json
|    ├── ensemble_mul_pred.json
|    ├── ensemble_optuna_pred.json
|    ├── ensemble_max_voting_pred.json
|    ├── robert_pred.json
|    └── xlnet_pred.json
├── src
|    ├── preprocessing.py
|    ├── evaluate-v2.0.py
|    ├── baseline
|    |    ├── bilstm_bert.py
|    |    ├── bert_squad.py
|    |    ├── roberta.py
|    |    └── xlnet.py
|    └── ensemble
|         ├── ensemble_equal_weighting.py
|         ├── ensemble_optuna.ipynb
|         ├── ensemble_unequal_optuna.py
|         └── max_vote.py
└── README.md
```

## Folder Contents:
1. **data/** : This folder consists of all the raw and processed data for training and evaluation
2. **model/** : This folder consists of all model weights for our models mentioned above
3. **output/** : This folder consists of all the predictions output by each model mentioned above
4. **src/** : This folder consists of the code needed for this entire project - preprocessing, individual model, ensemble model, and official evaluation script from SQuAD.
