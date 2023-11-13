#!/bin/sh

python ../../src/max_vote.py \
	--data_path "../../data/curated/test_data" \
	--xlnet_model "../../src/model/xlnet.pt" \
	--bilstm_model "../../src/model/bilstm.pt" \
	--output_path "allMaxVoteAns.json" \
	--final_output_path "maxVoteAns.json"
