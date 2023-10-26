#!/bin/sh

python xlnet.py \
    --test\
	--data_path "../data/curated/test_data"\
	--model_path "./xlnet.pt"\
    --output_path "./pred.json"