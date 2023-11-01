#!/bin/sh

python xlnet.py \
    --train\
	--data_path "../data/curated/training_data"\
	--model_path "./xlnet.pt"