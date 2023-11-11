#!/bin/sh

python ../../src/bilstm_bert.py \
	--train_kf\
	--train_path "../../data/curated/training_data"\
	--test_path "../../data/curated/test_data"\
	--model_path "./bilstm.pt"\
	--output_path "./bilstm_pred.json"\
	--score_path "./bilstm_scores.json"\
	--metric_path "./bilstm_metrics.json"