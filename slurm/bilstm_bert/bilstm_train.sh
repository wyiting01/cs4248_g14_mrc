#!/bin/sh

python ../../src/bilstm_bert.py \
	--train \
	--train_path "../../data/curated/training_data"\
	--model_path "./bilstm.pt"

python ../../src/bilstm_bert.py \
	--test \
	--test_path "../../data/curated/test_data"\
	--model_path "./bilstm.pt"\
	--output_path "./bilstm_pred.json"\
	--score_path "./bilstm_scores.json"