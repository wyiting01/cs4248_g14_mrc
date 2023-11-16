# CS4248 G14 Machine Reading Comprehension on SQuAD

## Dataset
Training Data: `data/raw/train-v1.1.json`  
Test Data: `data/raw/dev-v1.1.json`

### Data Preprocessing  
We preprocessed the data json files by separating context, question, question id, answer and answer span into individual file. This preprocessing is applied to both `train-v1.1.json` and `dev-v1.1.json`.

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

|           | Model                  | Exact                            | F1                              |
| --------- | ---------------------- | -------------------------------- | ------------------------------- |
| BASELINE  | BERT-SQuAD             | 57.79564806054872                | 72.18288133527203               |
|           | RoBERTa                | 81.20151371807                   | 88.41621816924753               |
|           | XLNet                  | 75.34531693472091                | 84.06374401013184               |
|           | biLSTM                 | 74.41882991088725                | 82.48893567082695               | 
| ENSEMBLE  | Equal - Maximum        | 78.89309366130558                | 86.70557785319596               |
|           | Equal - Multiplicative | 82.19489120151371                | 88.9101732833653                |
|           | Unequal - Optuna       | 81.47587511825922                | 88.20920854244099               |
|           | Unequal - CAWPE (fixed)| 82.42194891201514                | 88.96254477083748               |
|           | Unequal - CAWPE (auto) | <ins>**82.69631031220435**</ins> | <ins>**89.11858601649928**</ins>|

To run the evaluation script with model predictions `pred.json`, simply run the command below:
```
python src/evaluate-v2.0.py data/raw/dev-v1.1.json output/pred.json
```
> Note that `pred.json` can be replaced with any output json files containing the predictions of your target model found in `output/`  
> Use of modified evaluation script `evaluate-v2.1.py` will handle any encoding/key-value errors

## Baseline Models

### 1. BERT-SQuAD
- Folder: `src/baseline/bert`
- Model: [BERT-SQuAD](https://github.com/kamalkraj/BERT-SQuAD/tree/master)

```
# training: Using pretrained weights

# testing:
python src/baseline/bert/test.py --test --question_input "data/curated/test_data/question" --context_input "data/curated/test_data/context" --output_file "output/bert_squad_pred.json"
```

### 2. RoBERTa
- Script: `src/baseline/roberta.py`
- Model: [RobertaForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaForQuestionAnswering)

```
# training:
python src/baseline/roberta.py --train --data_path "data/curated/training_data" --model_path "model/roberta.pt"

# testing:
python src/baseline/roberta.py --test --data_path "data/curated/test_data" --model_path "model/roberta.pt" --output_path "output/roberta_pred.json"
```

### 3. XLNet
- Script: `src/baseline/xlnet.py`
- Model: [XLNetForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/xlnet#transformers.XLNetForQuestionAnswering)  

```
# training:
python src/baseline/xlnet.py --train --data_path "data/curated/training_data" --model_path "model/xlnet.pt"

# testing:
python src/baseline/xlnet.py --test --data_path "data/curated/test_data" --model_path "model/xlnet.pt" --output_path "output/xlnet_pred_top.json"
```

### 4. biLSTM
- Script: `src/baseline/bilstm_bert.py`
- Model: biLSTM-BERT

```
# training:
python src/baseline/bilstm_bert.py --train --train_path "data/curated/training_data" --model_path "model/bilstm.pt"

# testing
python src/baseline/bilstm_bert.py --test --test_path "data/curated/test_data" --model_path "model/bilstm.pt" --output_path "output/bilstm_pred.json" --score_path "intermediate/bilstm_scores.json"
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
python src/ensemble/ensemble_equal.py\
        --maximum\
        --data_path "data/curated/test_data"\
        --roberta_path "model/roberta.pt"\
        --xlnet_path "model/xlnet.pt"\
        --output_path "output/ensemble_max_pred.json"

# for multiplicative score
python src/ensemble/ensemble_equal.py\
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
python src/ensemble/ensemble_unequal_optuna.py --train --roberta --data_path "data/curated/ensemble_data/train" --xlnet_path "model/xlnet_optuna.pt" --roberta_path "model/roberta_optuna.pt" --xlnet_dict "intermediate/xlnet.json" --roberta_dict "output/roberta.json"
python src/ensemble/ensemble_unequal_optuna.py --train --xlnet --data_path "data/curated/ensemble_data/train" --xlnet_path "model/xlnet_optuna.pt" --roberta_path "model/roberta_optuna.pt" --xlnet_dict "intermediate/xlnet.json" --roberta_dict "output/roberta.json"

# Part B: Obtaining candidates for validation set (20% of train-v1.1.json)
python src/ensemble/ensemble_unequal_optuna.py --get_candidates --xlnet --data_path "data/curated/ensemble_data/val" --xlnet_path "model/xlnet_optuna.pt" --roberta_path "model/roberta_optuna.pt" --xlnet_dict "intermediate/xlnet_val.json" --roberta_dict "intermediate/roberta_val.json"
python src/ensemble/ensemble_unequal_optuna.py --get_candidates --roberta --data_path "data/curated/ensemble_data/val" --xlnet_path "model/xlnet_optuna.pt" --roberta_path "model/roberta_optuna.pt" --xlnet_dict "intermediate/xlnet_val.json" --roberta_dict "intermediate/roberta_val.json"

# Part C: Obtaining candidates for test set and perform weighting based on Optuna weights
python src/ensemble/ensemble_unequal_optuna.py --get_candidates --xlnet --data_path "data/curated/test_data" --xlnet_path "model/xlnet_optuna.pt" --roberta_path "model/roberta_optuna.pt" --xlnet_dict "intermediate/xlnet_test.json" --roberta_dict "intermediate/roberta_test.json"
python src/ensemble/ensemble_unequal_optuna.py --get_candidates --roberta --data_path "data/curated/test_data" --xlnet_path "model/xlnet_optuna.pt" --roberta_path "model/roberta_optuna.pt" --xlnet_dict "intermediate/xlnet_test.json" --roberta_dict "intermediate/roberta_test.json"
python src/ensemble/ensemble_unequal_optuna.py --test --xlnet_dict "intermediate/xlnet_test.json" --roberta_dict "intermediate/roberta_test.json" --output_path "output/ensemble_optuna_pred.json" --xlnet_weight 0.41 --roberta_weight 0.59
```
> Note that training of both models (as well as getting candidates) cannot be performed at the same time due to the limitation of the cluster, hence running them separately is required.
> Note that Optuna trials are performed in `ensemble_optuna.ipynb`

**4. Weighting based on CAWPE**  
In this approach, we follow the unequal weighting scheme as proposed in the cross-validation accuracy weighted probabilistic ensemble (CAWPE). We used k-fold CV to obtain an averaged accuracy metric, which are then exponentiated by a chosen alpha value to magnify differences in competence of each model.

```
# perform kfold for participating baseline models
python src/baseline/roberta.py --train_kf --data_path "data/curated/training_data" --model_path "model/roberta_kf.pt" --metric_path "intermediate/roberta_kf_scores.json"
python src/baseline/xlnet.py --train_kf --data_path "data/curated/training_data" --model_path "model/xlnet_kf.pt" --metric_path "intermediate/xlnet_kf_scores.json"

# perform weighitng based on kfold average accuracy
python src/ensemble/ensemble_unequal.py --test --xlnet_dict "intermediate/xlnet_test.json" --roberta_dict "intermediate/roberta_test.json" --xlnet_acc "intermediate/xlnet_kf_scores.json" --roberta_acc "intermediate/roberta_kf_scores.json" --output_path "output"
```

## Directory Structure
To navigate around this repository, you can refer to the directory tree below:

```
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
├── intermediate
|    ├── xlnet_val.json
|    ├── xlnet_test.json
|    ├── roberta_val.json
|    ├── roberta_test.json
|    ├── bilstm_metrics.json
|    ├── bilstm_scores.json
|    ├── xlnet_kf_scores.json
|    └── roberta_kf_scores.json
├── model
|    ├── bert_squad.pt
|    ├── bilstm.pt
|    ├── roberta.pt
|    ├── xlnet.pt
|    ├── roberta_optuna.pt
|    ├── xlnet_optuna.pt
|    ├── roberta_kf.pt
|    └── xlnet_kf.pt
├── output
|    ├── bert_pred.json
|    ├── bilstm_pred.json
|    ├── ensemble_max_pred.json
|    ├── ensemble_mul_pred.json
|    ├── ensemble_optuna_pred.json
|    ├── unequal_weight_fixed_pred.json
|    ├── unequal_weight_auto_pred.json
|    ├── roberta_pred.json
|    └── xlnet_pred.json
├── src
|    ├── preprocessing.py
|    ├── evaluate-v2.0.py
|    ├── baseline
|    |    ├── bilstm_bert.py
|    |    ├── roberta.py
|    |    ├── xlnet.py
|    |    └── bert
|    |          └── test.py
|    └── ensemble
|         ├── ensemble_equal.py
|         ├── ensemble_optuna.ipynb
|         ├── ensemble_unequal_optuna.py
|         └── ensemble_unequal.py
└── README.md
```

## Folder Contents:
1. **data/** : This folder consists of all the raw and processed data for training and evaluation
2. **intermediate/** : This folder consists of all the intermediate outputs that are generated.
3. **model/** : This folder consists of all model weights for our models (baseline) mentioned above
4. **output/** : This folder consists of all the predictions output by each model (baseline and ensemble) mentioned above
5. **src/** : This folder consists of the code needed for this entire project - preprocessing, individual models, ensemble models, and official evaluation script from SQuAD.

## References:
1. [Hugging Face Preprocessing for Modeling](https://huggingface.co/docs/transformers/tasks/question_answering)
2. [BERT-SQuAD](https://github.com/kamalkraj/BERT-SQuAD/tree/master)
