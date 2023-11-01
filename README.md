# CS4248 G14 Machine Reading Comprehension on SQuAD

## Base Models

### Results
The table below is the Exact and F1 scores evaluted from the official SQuAD evaluation script `evaluate-v2.0.py`.  

| Model | Exact | F1 |
| ----- | ----- | -- |
| BERT  | 57.79564806054872 | 72.18288133527203 |
| XLNet | 75.34531693472091 | 84.06374401013184 |
| biLSTM | | | 

To run the evaluation script with model predictions `pred.json`, simply run the command below:
```
python evaluate-v2.0.py data/raw/dev-v1.1.json pred.json
```

### 1. BERT
- Script: `src/bert.py`
- Model: BERT-SQuAD

### 2. XLNet
- Script: `src/xlnet.py`
- Model: [XLNetForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/xlnet#transformers.XLNetForQuestionAnswering)

To train/test using XLNet:
```
# training:
python xlnet.py --train --data_path "../data/curated/training_data" --model_path "./xlnet.pt"

# testing:
python xlnet.py --test --data_path "../data/curated/test_data" --model_path "./xlnet.pt" --output_path "./pred.json"
```

### 3. biLSTM
- Script: ``
- Model:

## Ensemble Models
