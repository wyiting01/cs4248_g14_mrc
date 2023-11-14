#!/bin/sh

python ../../src/max_vote.py \
	--data_path "../../data/curated/test_data" \
	--bert_path "../../src/bert/model" \
	--xlnet_model "../../model/xlnet.pt" \
	--bilstm_model "../../model/bilstm.pt" \
	--output_path "allMaxVoteAns.json" \
	--final_output_path "maxVoteAns.json"
