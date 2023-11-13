#!/bin/sh

python ../../src/max_vote.py \
	--data_path "../../data/curated/test_data" \
	--xlnet_model "../../src/model/xlnet.pt" \
	--model_path "../../src/model/bilstm.pt"
